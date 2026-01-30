import uuid
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.scraper import scrape_article
from backend.cleaner import clean_text, remove_promotional_content
from backend.chunker import chunk_text
from backend.extractor import batch_extract, merge_extractions
from backend.normalizer import normalize_pipeline
from backend.graph_builder import build_graph, enrich_graph, export_to_json
from backend.graph_storage import save_graph, load_graph


app = FastAPI(
    title="Knowledge Graph Builder API",
    description="REST API for building knowledge graphs from web articles",
    version="1.0.0"
)


origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScrapeRequest(BaseModel):
    url: HttpUrl
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.bbc.com/news/articles/example"
            }
        }


class ScrapeResponse(BaseModel):
    job_id: str
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending"
            }
        }


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = Field(ge=0, le=100)
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 45.5,
                "message": "Extracting entities from chunks"
            }
        }


class GraphResponse(BaseModel):
    nodes: list
    edges: list
    metadata: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "id": "node_1",
                        "label": "OpenAI",
                        "type": "ORGANIZATION",
                        "properties": {},
                        "metrics": {}
                    }
                ],
                "edges": [
                    {
                        "id": "edge_1",
                        "source": "node_1",
                        "target": "node_2",
                        "label": "WORKS_FOR",
                        "type": "WORKS_FOR",
                        "properties": {}
                    }
                ],
                "metadata": {
                    "node_count": 1,
                    "edge_count": 1,
                    "metrics": {}
                }
            }
        }


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2024-01-29T10:30:00",
                "version": "1.0.0"
            }
        }


jobs: Dict[str, Dict[str, Any]] = {}


async def process_pipeline(job_id: str, url: str):
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = "Starting pipeline"
        
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Scraping article"
        article = scrape_article(url)
        
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Cleaning text"
        cleaned = clean_text(article.get("text", ""))
        cleaned = remove_promotional_content(cleaned)
        
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Chunking text"
        chunks = chunk_text(cleaned, chunk_size=300, overlap=30)
        
        jobs[job_id]["progress"] = 40
        jobs[job_id]["message"] = "Extracting entities and relationships"
        extractions = batch_extract(chunks, batch_size=5)
        
        jobs[job_id]["progress"] = 60
        jobs[job_id]["message"] = "Merging extractions"
        merged = merge_extractions(extractions)
        
        jobs[job_id]["progress"] = 70
        jobs[job_id]["message"] = "Normalizing entities"
        entities, relationships = normalize_pipeline(
            merged["entities"],
            merged["relationships"],
            text=cleaned
        )
        
        jobs[job_id]["progress"] = 80
        jobs[job_id]["message"] = "Building knowledge graph"
        graph_data = build_graph(entities, relationships)
        
        jobs[job_id]["progress"] = 90
        jobs[job_id]["message"] = "Enriching graph"
        enriched = enrich_graph(graph_data)
        
        jobs[job_id]["progress"] = 95
        jobs[job_id]["message"] = "Saving graph"
        save_graph(enriched, f"job_{job_id}")
        
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["message"] = "Graph building completed successfully"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["message"] = f"Pipeline failed: {str(e)}"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Knowledge Graph Builder API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "scrape": "POST /api/scrape",
            "status": "GET /api/status/{job_id}",
            "graph": "GET /api/graph/{job_id}"
        }
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/api/scrape", response_model=ScrapeResponse, tags=["Pipeline"])
async def scrape_and_build_graph(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "url": str(request.url),
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Job created",
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(process_pipeline, job_id, str(request.url))
    
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING
    }


@app.get("/api/status/{job_id}", response_model=StatusResponse, tags=["Pipeline"])
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"]
    }


@app.get("/api/graph/{job_id}", response_model=GraphResponse, tags=["Pipeline"])
async def get_graph(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed. Current status: {job['status']}"
        )
    
    graph_data = load_graph(f"job_{job_id}")
    
    if graph_data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load graph for job {job_id}"
        )
    
    json_output = export_to_json(graph_data)
    
    return {
        "nodes": json_output["nodes"],
        "edges": json_output["edges"],
        "metadata": json_output["metadata"]
    }


@app.get("/api/jobs", tags=["Management"])
async def list_jobs():
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"],
                "url": job.get("url", "")
            }
            for job_id, job in jobs.items()
        ]
    }


@app.delete("/api/jobs/{job_id}", tags=["Management"])
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    del jobs[job_id]
    
    return {
        "message": f"Job {job_id} deleted successfully"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)