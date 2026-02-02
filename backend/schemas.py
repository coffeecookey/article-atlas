
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


# Stores metadata extracted separately from article text
class ArticleMetadata(BaseModel):
    reading_time: Optional[int] = Field(None, ge=0)  # Estimated reading time (minutes)
    tags: List[str] = Field(default_factory=list)    # Article tags
    categories: List[str] = Field(default_factory=list)  # Article categories

    # Cleans tag/category lists by stripping whitespace and removing empties
    @field_validator('tags', 'categories')
    def clean_list_items(v: List[str]) -> List[str]:
        return [item.strip() for item in v if item and item.strip()]


# Represents cleaned article content after scraping
class ArticleContent(BaseModel):
    url: str                                      # Source URL
    title: str = Field(..., min_length=1)         # Article title
    text: str = Field(..., min_length=1)          # Cleaned article text
    author: Optional[str] = None                  # Author name (optional)
    publish_date: Optional[datetime] = None       # Publication date (optional)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra metadata

    # Ensures URL is not empty or whitespace
    @field_validator('url')
    def validate_url(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    # Ensures title and text are not empty
    @field_validator('title', 'text')
    def validate_non_empty(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Represents a semantic chunk of article text sent to the LLM
class TextChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))  # Unique chunk ID
    text: str = Field(..., min_length=1)                         # Chunk text
    position: int = Field(..., ge=0)                              # Order in article
    token_count: int = Field(..., ge=0)                           # Approx token count
    source_url: str                                               # Article URL

    # Ensures chunk text is valid
    @field_validator('text')
    def validate_text(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text chunk cannot be empty")
        return v.strip()


# Represents an entity (node) in the knowledge graph
class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))  # Unique entity ID
    name: str = Field(..., min_length=1)                   # Entity name
    type: str = Field(..., min_length=1)                   # Entity type
    properties: Dict[str, Any] = Field(default_factory=dict)  # Extra attributes
    mentions: List[str] = Field(default_factory=list)         # Text mentions

    # Ensures name and type are valid
    @field_validator('name', 'type')
    def validate_non_empty(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Represents a relationship (edge) between two entities
class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))  # Unique relationship ID
    source_entity_id: str = Field(..., min_length=1)       # Source entity ID
    target_entity_id: str = Field(..., min_length=1)       # Target entity ID
    relationship_type: str = Field(..., min_length=1)      # Relationship type
    properties: Dict[str, Any] = Field(default_factory=dict)  # Extra attributes

    # Ensures required fields are not empty
    @field_validator('source_entity_id', 'target_entity_id', 'relationship_type')
    def validate_non_empty(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Output of entity & relationship extraction from one text chunk
class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)         # Extracted entities
    relationships: List[Relationship] = Field(default_factory=list)  # Extracted edges
    chunk_id: str                                                # Source chunk ID


# Represents the full merged knowledge graph
class GraphData(BaseModel):
    nodes: List[Entity] = Field(default_factory=list)        # All graph nodes
    edges: List[Relationship] = Field(default_factory=list)  # All graph edges
    metadata: Dict[str, Any] = Field(default_factory=dict)   # Graph-level metadata

    # Ensures all node IDs are unique
    @field_validator('nodes')
    def validate_unique_nodes(v: List[Entity]) -> List[Entity]:
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs found")
        return v

    # Ensures edges reference valid nodes
    @field_validator('edges')
    def validate_edge_references(v: List[Relationship], info) -> List[Relationship]:
        if 'nodes' in info.data:
            node_ids = {n.id for n in info.data['nodes']}
            for edge in v:
                if edge.source_entity_id not in node_ids:
                    raise ValueError(f"Invalid source node: {edge.source_entity_id}")
                if edge.target_entity_id not in node_ids:
                    raise ValueError(f"Invalid target node: {edge.target_entity_id}")
        return v


# Controls what the LLM is allowed to extract
class GraphSchema(BaseModel):
    entity_types: List[str] = Field(default_factory=list)        # Allowed entity types
    relationship_types: List[str] = Field(default_factory=list) # Allowed relation types
    extraction_prompt: str = Field(..., min_length=1)           # LLM prompt template

    # Normalizes and validates type lists
    @field_validator('entity_types', 'relationship_types')
    def validate_unique_types(v: List[str]) -> List[str]:
        cleaned = [item.strip().upper() for item in v if item and item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Duplicate types found")
        return cleaned

    # Ensures prompt is not empty
    @field_validator('extraction_prompt')
    def validate_prompt(v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Extraction prompt cannot be empty")
        return v.strip()


# Logs scraping failures instead of crashing the pipeline
class ScrapingFailure(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)  # Failure time
    url: str                                                   # Failed URL
    domain: str                                                # Domain name
    failure_type: str                                          # Type of failure
    http_status: Optional[int] = None                          # HTTP status (if any)
    error_message: Optional[str] = None                        # Extra details


# Controls what gets imported when using `from schemas import *`
__all__ = [
    'ArticleMetadata',
    'ArticleContent',
    'TextChunk',
    'Entity',
    'Relationship',
    'GraphExtraction',
    'GraphData',
    'GraphSchema',
    'ScrapingFailure'
]
