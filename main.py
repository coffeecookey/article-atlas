# import uvicorn
# import uuid
# from fastapi import FastAPI, HTTPException, BackgroundTasks, status
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, HttpUrl, Field
# from typing import List, Optional, Dict, Any
# from datetime import datetime
# from enum import Enum
# import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from backend.config import settings
from backend.scraper import scrape_article
from backend.cleaner import clean_text, remove_promotional_content
from backend.chunker import chunk_text
from backend.extractor import batch_extract, merge_extractions
from backend.normalizer import normalize_entities
from backend.graph_builder import build_graph

app = FastAPI(title="Knowledge Graph API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str


class GraphResponse(BaseModel):
    entities: list
    relationships: list
    entity_count: int
    relationship_count: int


@app.post("/api/process", response_model=dict)
async def process_article(request: URLRequest):
    """
    Process article URL through the complete pipeline.
    
    Steps:
    1. Scrape article from URL
    2. Clean the text
    3. Chunk into smaller pieces
    4. Extract entities and relationships using GPT-4
    5. Merge into knowledge graph
    6. Return JSON to frontend
    """
    try:
        url = request.url
        print(f"\n{'='*70}")
        print(f"Processing URL: {url}")
        print(f"{'='*70}\n")
        
        # Step 1: Scrape article
        print("Step 1: Scraping article...")
        article = scrape_article(url)
        raw_text = article.get('text', '')
        
        if not raw_text:
            raise HTTPException(status_code=400, detail="No text content found in article")
        
        print(f"✓ Scraped {len(raw_text)} characters")
        
        # Step 2: Clean text
        print("\nStep 2: Cleaning text...")
        cleaned = clean_text(raw_text)
        cleaned = remove_promotional_content(cleaned)
        print(f"✓ Cleaned to {len(cleaned)} characters")
        
        # Step 3: Chunk text
        print("\nStep 3: Chunking text...")
        chunks = chunk_text(cleaned, chunk_size=300, overlap=30)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Limit chunks for faster processing (remove this in production)
        chunks_to_process = chunks[:5]  # Process first 5 chunks
        print(f"⚠ Processing {len(chunks_to_process)} chunks (limited for speed)")
        
        # Step 4: Extract entities and relationships
        print("\nStep 4: Extracting entities and relationships...")
        extractions = batch_extract(chunks_to_process, batch_size=2)
        
        total_entities = sum(len(e.get('entities', [])) for e in extractions)
        total_relationships = sum(len(e.get('relationships', [])) for e in extractions)
        print(f"✓ Extracted {total_entities} entities and {total_relationships} relationships")
        
        # Step 5: Merge into knowledge graph
        print("\nStep 5: Merging into knowledge graph...")
        merged_graph = merge_extractions(extractions)
        print(f"✓ Final graph: {merged_graph['entity_count']} entities, {merged_graph['relationship_count']} relationships")
        
        print(f"\n{'='*70}")
        print("✓ Pipeline complete!")
        print(f"{'='*70}\n")
        
        # Return JSON response
        return {
            "entities": merged_graph['entities'],
            "relationships": merged_graph['relationships'],
            "entity_count": merged_graph['entity_count'],
            "relationship_count": merged_graph['relationship_count']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Knowledge Graph API",
        "endpoints": {
            "POST /api/process": "Process an article URL and return knowledge graph"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Starting FastAPI server...")
    print("API URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)