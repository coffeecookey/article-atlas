import os
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
        
        self.NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
        self.NEO4J_USER: Optional[str] = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
        
        self.MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.RATE_LIMIT_DELAY: float = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
        
        self.STORAGE_MODE: str = os.getenv("STORAGE_MODE", "json")
        
        cors_origins_str = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://localhost:8080"
        )
        self.CORS_ORIGINS: List[str] = [
            origin.strip() for origin in cors_origins_str.split(",")
        ]
        
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
        
        self.SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "85.0"))
        
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.USER_AGENT: str = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        
        self.LLM_ENABLED: bool = bool(self.GOOGLE_API_KEY or self.OPENAI_API_KEY)
        
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
        
        self.STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./data/graphs")
        self.MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
        
        self.BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))


def load_settings() -> Settings:
    settings = Settings()
    
    validate_settings(settings)
    
    return settings


def validate_settings(settings: Settings) -> None:
    errors = []
    
    if not settings.LLM_ENABLED:
        errors.append("No LLM API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY")
    
    if settings.STORAGE_MODE not in ["neo4j", "json"]:
        errors.append(f"Invalid STORAGE_MODE: {settings.STORAGE_MODE}. Must be 'neo4j' or 'json'")
    
    if settings.STORAGE_MODE == "neo4j":
        if not settings.NEO4J_URI:
            errors.append("NEO4J_URI required when STORAGE_MODE is 'neo4j'")
        if not settings.NEO4J_USER:
            errors.append("NEO4J_USER required when STORAGE_MODE is 'neo4j'")
        if not settings.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD required when STORAGE_MODE is 'neo4j'")
    
    if settings.MAX_CHUNK_SIZE <= 0:
        errors.append("MAX_CHUNK_SIZE must be positive")
    
    if settings.CHUNK_OVERLAP < 0:
        errors.append("CHUNK_OVERLAP must be non-negative")
    
    if settings.CHUNK_OVERLAP >= settings.MAX_CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than MAX_CHUNK_SIZE")
    
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_message)


settings = load_settings()


API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
ALLOWED_ORIGINS = settings.CORS_ORIGINS
CHUNK_SIZE = settings.MAX_CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD
REQUEST_TIMEOUT = settings.REQUEST_TIMEOUT
USER_AGENT = settings.USER_AGENT
OPENAI_API_KEY = settings.OPENAI_API_KEY
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY
LLM_ENABLED = settings.LLM_ENABLED


def get_settings() -> Settings:
    return settings


def print_settings():
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"API Host: {settings.API_HOST}")
    print(f"API Port: {settings.API_PORT}")
    print(f"Storage Mode: {settings.STORAGE_MODE}")
    print(f"Max Chunk Size: {settings.MAX_CHUNK_SIZE}")
    print(f"Chunk Overlap: {settings.CHUNK_OVERLAP}")
    print(f"Rate Limit Delay: {settings.RATE_LIMIT_DELAY}s")
    print(f"Max Retries: {settings.MAX_RETRIES}")
    print(f"Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
    print(f"LLM Enabled: {settings.LLM_ENABLED}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Storage Directory: {settings.STORAGE_DIR}")
    print(f"CORS Origins: {', '.join(settings.CORS_ORIGINS)}")
    
    print("\nAPI Keys:")
    print(f"  OpenAI: {'Set' if settings.OPENAI_API_KEY else 'Not set'}")
    print(f"  Google: {'Set' if settings.GOOGLE_API_KEY else 'Not set'}")
    print(f"  Anthropic: {'Set' if settings.ANTHROPIC_API_KEY else 'Not set'}")
    
    if settings.STORAGE_MODE == "neo4j":
        print("\nNeo4j Configuration:")
        print(f"  URI: {settings.NEO4J_URI}")
        print(f"  User: {settings.NEO4J_USER}")
        print(f"  Password: {'Set' if settings.NEO4J_PASSWORD else 'Not set'}")
    
    print("="*70)


if __name__ == "__main__":
    print_settings()