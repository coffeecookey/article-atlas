import uvicorn
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import logging

from backend.config import settings
from backend.scraper import scrape_article
from backend.cleaner import clean_text
from backend.chunker import chunk_text
from backend.extractor import extract_entities_and_relationships
from backend.normalizer import normalize_entities
from backend.graph_builder import build_graph


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# create logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Knowledge Graph API",
    description="Convert articles into knowledge graphs",
    version="1.0.0",
    debug=settings.DEBUG
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncProcessRequest(BaseModel):
    url: HttpUrl
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.bbc.com/news/articles/example"
            }
        }


class ProcessRequest(BaseModel):
    url: HttpUrl


class JobResponse(BaseModel):
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

# temporary DB for storing jobs
jobs: Dict[str, Dict[str, Any]] = {}


async def process_pipeline_async(job_id: str, url: str):
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = "Starting pipeline"
        
        logger.info(f"Job {job_id}: Scraping article")
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Scraping article"
        article = scrape_article(url)
        
        if not article or not article.get('text'):
            raise Exception("Could not extract article content")
        
        logger.info(f"Job {job_id}: Cleaning text")
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Cleaning text"
        cleaned_text = clean_text(article['text'])
        
        logger.info(f"Job {job_id}: Chunking text")
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Chunking text"
        chunks = chunk_text(cleaned_text, chunk_size=settings.MAX_CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        logger.info(f"Job {job_id}: Created {len(chunks)} chunks")
        
        logger.info(f"Job {job_id}: Extracting entities")
        jobs[job_id]["progress"] = 40
        jobs[job_id]["message"] = "Extracting entities and relationships"
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            result = extract_entities_and_relationships(chunk)

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            all_entities.extend(entities)
            all_relationships.extend(relationships)

            progress = 40 + (i + 1) / len(chunks) * 20
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = f"Processing chunk {i+1}/{len(chunks)}"
        
        logger.info(f"Job {job_id}: Found {len(all_entities)} raw entities")
        
        logger.info(f"Job {job_id}: Normalizing entities")
        jobs[job_id]["progress"] = 70
        jobs[job_id]["message"] = "Normalizing entities"
        normalized_entities = normalize_entities(all_entities)
        logger.info(f"Job {job_id}: Normalized to {len(normalized_entities)} unique entities")
        
        logger.info(f"Job {job_id}: Building knowledge graph")
        jobs[job_id]["progress"] = 85
        jobs[job_id]["message"] = "Building knowledge graph"
        graph_data = build_graph(normalized_entities, all_relationships)
        
        logger.info(f"Job {job_id}: Converting to JSON")
        jobs[job_id]["progress"] = 95
        jobs[job_id]["message"] = "Finalizing graph"
        
        logger.info(f"Job {job_id}: Complete")
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["message"] = "Graph building completed successfully"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {
            "graph": graph_data,
            "article_info": {
                'title': article.get('title', 'Untitled'),
                'url': url,
                'authors': article.get('authors', []),
                'publish_date': str(article.get('publish_date', ''))
            },
            "stats": {
                'chunks': len(chunks),
                'raw_entities': len(all_entities),
                'entities': len(normalized_entities),
                'relationships': len(all_relationships)
            }
        }
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed - {str(e)}", exc_info=True)
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["message"] = f"Pipeline failed: {str(e)}"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@app.get("/")
def root():
    return {
        "status": "Knowledge Graph API Running",
        "version": "1.0.0",
        "endpoints": {
            "sync": {
                "/scrape": "GET - Scrape article content",
                "/process": "POST - Full pipeline (synchronous, waits for completion)",
                "/clean": "POST - Test text cleaning",
                "/chunk": "POST - Test text chunking"
            },
            "async": {
                "/process-async": "POST - Start async processing (returns job ID)",
                "/status/{job_id}": "GET - Check job status and progress",
                "/result/{job_id}": "GET - Get completed graph result",
                "/jobs": "GET - List all jobs"
            },
            "health": {
                "/health": "GET - Health check"
            }
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "scraper": "operational",
            "nlp": "operational",
            "llm_enabled": settings.LLM_ENABLED
        },
        "config": {
            "storage_mode": settings.STORAGE_MODE,
            "chunk_size": settings.MAX_CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP
        }
    }


@app.get("/scrape")
def scrape(url: str):
    try:
        logger.info(f"Scraping URL: {url}")
        article = scrape_article(url)

        if not article:
            raise HTTPException(
                status_code=400,
                detail="Could not extract article content"
            )

        return {
            "success": True,
            "url": url,
            "title": article.get('title', 'No title'),
            "text": article.get('text', ''),
            "authors": article.get('authors', []),
            "publish_date": article.get('publish_date')
        }

    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/process")
async def process_article_sync(request: ProcessRequest):
    try:
        url = str(request.url)
        logger.info(f"Processing article synchronously: {url}")

        logger.info("Step 1: Scraping...")
        article = scrape_article(url)
        if not article or not article.get('text'):
            raise HTTPException(status_code=400, detail="Could not extract article")

        logger.info("Step 2: Cleaning text...")
        cleaned_text = clean_text(article['text'])

        logger.info("Step 3: Chunking text...")
        chunks = chunk_text(cleaned_text, chunk_size=settings.MAX_CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Step 4: Extracting entities...")
        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            result = extract_entities_and_relationships(chunk)

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        logger.info(f"Found {len(all_entities)} raw entities")

        logger.info("Step 5: Normalizing entities...")
        normalized_entities = normalize_entities(all_entities)
        logger.info(f"Normalized to {len(normalized_entities)} unique entities")

        logger.info("Step 6: Building knowledge graph...")
        graph_data = build_graph(normalized_entities, all_relationships)

        logger.info(f"Graph complete")

        return {
            "success": True,
            "graph": graph_data,
            "article_info": {
                'title': article.get('title', 'Untitled'),
                'url': url,
                'authors': article.get('authors', []),
                'publish_date': str(article.get('publish_date', ''))
            },
            "stats": {
                'chunks': len(chunks),
                'raw_entities': len(all_entities),
                'entities': len(normalized_entities),
                'relationships': len(all_relationships)
            },
            "message": "Graph generated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

@app.post("/process-async", response_model=JobResponse)
async def process_article_async(request: AsyncProcessRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    url = str(request.url)
    
    logger.info(f"Creating async job {job_id} for URL: {url}")
    
    jobs[job_id] = {
        "job_id": job_id,
        "url": url,
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Job created",
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(process_pipeline_async, job_id, url)
    
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING
    }


@app.get("/status/{job_id}", response_model=StatusResponse)
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


@app.get("/result/{job_id}")
async def get_job_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    if job["status"] == JobStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is pending. Current progress: {job['progress']}%"
        )
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is still processing. Current progress: {job['progress']}%"
        )
    
    if job["status"] == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} failed: {job.get('error', 'Unknown error')}"
        )
    
    return job.get("result", {})


@app.get("/jobs")
async def list_jobs():
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "url": job.get("url", ""),
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at"),
                "failed_at": job.get("failed_at")
            }
            for job_id, job in jobs.items()
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    del jobs[job_id]
    
    return {
        "success": True,
        "message": f"Job {job_id} deleted successfully"
    }


@app.post("/clean")
async def clean_text_endpoint(text: str):
    cleaned = clean_text(text)
    return {
        "original_length": len(text),
        "cleaned_length": len(cleaned),
        "cleaned_text": cleaned
    }


@app.post("/chunk")
async def chunk_text_endpoint(text: str):
    chunks = chunk_text(text, chunk_size=settings.MAX_CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    return {
        "num_chunks": len(chunks),
        "chunks": chunks
    }

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Starting Knowledge Graph API")
    logger.info("="*70)
    logger.info(f"Host: {settings.API_HOST}")
    logger.info(f"Port: {settings.API_PORT}")
    logger.info(f"Debug: {settings.DEBUG}")
    logger.info(f"LLM Enabled: {settings.LLM_ENABLED}")
    logger.info(f"Storage Mode: {settings.STORAGE_MODE}")
    logger.info(f"Chunk Size: {settings.MAX_CHUNK_SIZE}")
    logger.info(f"Chunk Overlap: {settings.CHUNK_OVERLAP}")
    logger.info("="*70)
    logger.info("Available Endpoints:")
    logger.info("  Sync:  POST /process (waits for completion)")
    logger.info("  Async: POST /process-async (returns job ID)")
    logger.info("         GET  /status/{job_id} (check progress)")
    logger.info("         GET  /result/{job_id} (get result)")
    logger.info("="*70)

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )