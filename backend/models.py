from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional, Any

class ProcessRequest(BaseModel):
    url: HttpUrl
    options: Optional[Dict[str, Any]] = None

class ArticleInfo(BaseModel):
    title: str
    url: str
    authors: List[str]
    publish_date: Optional[str]

class Stats(BaseModel):
    chunks: int
    raw_entities: int
    entities: int
    relationships: int
    nodes: int
    edges: int

class GraphResponse(BaseModel):
    success: bool
    graph: Dict[str, Any]  # JSON graph data
    article_info: Dict[str, Any]
    stats: Dict[str, Any]
    message: Optional[str] = None

class ArticleResponse(BaseModel):
    success: bool
    url: str
    title: str
    text: str
    authors: List[str]
    publish_date: Optional[str]

"""
Pydantic Models for API Request/Response validation
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Any

# ==================== Request Models ====================

class ProcessRequest(BaseModel):
    """Request to process an article URL"""
    url: HttpUrl
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional processing parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/article",
                "options": {
                    "chunk_size": 500,
                    "similarity_threshold": 85
                }
            }
        }

class TextCleanRequest(BaseModel):
    """Request to clean text"""
    text: str = Field(..., min_length=1, description="Text to clean")

class TextChunkRequest(BaseModel):
    """Request to chunk text"""
    text: str = Field(..., min_length=1, description="Text to chunk")
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    overlap: Optional[int] = Field(default=50, ge=0, le=500)

# ==================== Response Models ====================

class ArticleInfo(BaseModel):
    """Article metadata"""
    title: str
    url: str
    authors: List[str]
    publish_date: Optional[str]

class Stats(BaseModel):
    """Processing statistics"""
    chunks: int
    raw_entities: int
    entities: int
    relationships: int
    nodes: int
    edges: int

class GraphNode(BaseModel):
    """Graph node representation"""
    id: str
    label: str
    type: str
    mentions: int = 1
    aliases: List[str] = []
    centrality: Optional[float] = None
    importance: Optional[float] = None

class GraphEdge(BaseModel):
    """Graph edge representation"""
    source: str
    target: str
    relation: str
    weight: int = 1
    confidence: float = 1.0

class GraphData(BaseModel):
    """Complete graph data"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class GraphResponse(BaseModel):
    """Main response for graph processing"""
    success: bool
    graph: Dict[str, Any]
    article_info: Dict[str, Any]
    stats: Dict[str, Any]
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "graph": {
                    "nodes": [
                        {
                            "id": "Apple Inc",
                            "type": "ORG",
                            "mentions": 5
                        }
                    ],
                    "edges": [
                        {
                            "source": "Apple Inc",
                            "target": "Steve Jobs",
                            "relation": "FOUNDED"
                        }
                    ]
                },
                "article_info": {
                    "title": "Tech Company History",
                    "url": "https://example.com"
                },
                "stats": {
                    "nodes": 10,
                    "edges": 15
                }
            }
        }

class ArticleResponse(BaseModel):
    """Response for article scraping"""
    success: bool
    url: str
    title: str
    text: str
    authors: List[str]
    publish_date: Optional[str]
    word_count: Optional[int] = None

class CleanResponse(BaseModel):
    """Response for text cleaning"""
    original_length: int
    cleaned_length: int
    chars_removed: int
    reduction_percent: float
    cleaned_text: str

class ChunkResponse(BaseModel):
    """Response for text chunking"""
    num_chunks: int
    chunks: List[Dict[str, Any]]
    stats: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, str]