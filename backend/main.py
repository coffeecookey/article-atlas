import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
import logging
from backend.scraper import scrapeWeb
from backend.cleaner import TextCleaner
from backend.chunker import TextChunker
from backend.extractor import EntityExtractor
from backend.normalizer import EntityNormalizer
from backend.graph_builder import KnowledgeGraphBuilder
from backend.models import ArticleResponse, GraphResponse, ProcessRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Knowledge Graph API",
    description="Convert articles into knowledge graphs",
    version="1.0.0"
)

# CORS - Adding more origins 
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initializing the pipeline components
cleaner = TextCleaner()
chunker = TextChunker(chunk_size=500, overlap=50)
extractor = EntityExtractor()
normalizer = EntityNormalizer(similarity_threshold=85)
graph_builder = KnowledgeGraphBuilder()

@app.get("/")
def root():
    return {
        "status": "Backend running 🚀",
        "endpoints": {
            "/scrape": "GET - Scrape article content",
            "/process": "POST - Full pipeline: scrape → clean → chunk → extract → graph",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "scraper": "operational",
            "nlp": "operational"
        }
    }

@app.get("/scrape")
def scrape(url: str):
    """
    Basic scraping endpoint - just returns article content
    """
    try:
        logger.info(f"Scraping URL: {url}")
        article = scrapeWeb(url)

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

@app.post("/process", response_model=GraphResponse)
async def process_article(request: ProcessRequest):
    """
    Full pipeline: scrape → clean → chunk → extract entities → build graph
    """
    try:
        url = str(request.url)
        logger.info(f"Processing article: {url}")
        
        # Step 1: Scrape
        logger.info("Step 1: Scraping...")
        article = scrapeWeb(url)
        if not article or not article.get('text'):
            raise HTTPException(status_code=400, detail="Could not extract article")
        
        # Step 2: Clean
        logger.info("Step 2: Cleaning text...")
        clean_text = cleaner.clean(article['text'])
        
        # Step 3: Chunk
        logger.info("Step 3: Chunking text...")
        chunks = chunker.chunk(clean_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Extract entities and relationships
        logger.info("Step 4: Extracting entities...")
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            entities, relationships = extractor.extract_from_chunk(chunk['text'])
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        logger.info(f"Found {len(all_entities)} raw entities")
        
        # Step 5: Normalize entities
        logger.info("Step 5: Normalizing entities...")
        normalized_entities, entity_map = normalizer.normalize(all_entities)
        logger.info(f"Normalized to {len(normalized_entities)} unique entities")
        
        # Step 6: Build graph
        logger.info("Step 6: Building knowledge graph...")
        graph = graph_builder.build_graph(
            normalized_entities, 
            all_relationships, 
            entity_map
        )
        
        # Convert to JSON
        graph_json = graph_builder.to_json()
        
        logger.info(f"Graph complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return GraphResponse(
            success=True,
            graph=graph_json,
            article_info={
                'title': article.get('title', 'Untitled'),
                'url': url,
                'authors': article.get('authors', []),
                'publish_date': str(article.get('publish_date', ''))
            },
            stats={
                'chunks': len(chunks),
                'raw_entities': len(all_entities),
                'entities': len(normalized_entities),
                'relationships': len(all_relationships),
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges()
            },
            message="Graph generated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

@app.post("/clean")
async def clean_text(text: str):
    """
    Test endpoint for text cleaning
    """
    cleaned = cleaner.clean(text)
    return {
        "original_length": len(text),
        "cleaned_length": len(cleaned),
        "cleaned_text": cleaned
    }

@app.post("/chunk")
async def chunk_text(text: str):
    """
    Test endpoint for text chunking
    """
    chunks = chunker.chunk(text)
    return {
        "num_chunks": len(chunks),
        "chunks": chunks
    }

# Run with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )