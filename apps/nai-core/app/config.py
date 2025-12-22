"""
nAI Core Configuration
Environment-based configuration using pydantic-settings
"""

import os
from typing import Optional, List
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "nAI Core"
    app_version: str = "0.2.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins"
    )
    cors_allow_all: bool = Field(default=False, description="Allow all CORS origins (dev only)")
    
    # Paths
    data_dir: str = Field(default="data", description="Data directory path")
    docs_subdir: str = Field(default="docs", description="Documents subdirectory")
    index_subdir: str = Field(default="index", description="Index subdirectory")
    
    # Ingestion
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".txt", ".md", ".markdown", ".rst", ".html"],
        description="Allowed file extensions"
    )
    chunk_size: int = Field(default=1000, description="Default chunk size in characters")
    chunk_overlap: int = Field(default=150, description="Chunk overlap in characters")
    
    # BM25 Search
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    default_top_k: int = Field(default=5, description="Default number of results")
    
    # Embeddings
    embeddings_enabled: bool = Field(default=False, description="Enable embedding-based search")
    embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embeddings_dimension: int = Field(default=384, description="Embedding vector dimension")
    
    # Qdrant
    qdrant_enabled: bool = Field(default=False, description="Enable Qdrant vector store")
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection: str = Field(default="nai_documents", description="Qdrant collection name")
    
    # LLM
    llm_enabled: bool = Field(default=False, description="Enable LLM for answer generation")
    llm_provider: str = Field(default="ollama", description="LLM provider: ollama, openai, anthropic")
    llm_model: str = Field(default="llama3.2", description="LLM model name")
    llm_base_url: Optional[str] = Field(default="http://localhost:11434", description="LLM API base URL")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key (if required)")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=1024, description="Maximum tokens for LLM response")
    
    # Authentication
    auth_enabled: bool = Field(default=False, description="Enable authentication")
    auth_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET",
        description="JWT secret key"
    )
    auth_algorithm: str = Field(default="HS256", description="JWT algorithm")
    auth_token_expire_minutes: int = Field(default=1440, description="Token expiration in minutes")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")
    
    model_config = {
        "env_prefix": "NAI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
    
    @property
    def data_path(self) -> str:
        """Get absolute data directory path."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, self.data_dir)
    
    @property
    def docs_path(self) -> str:
        """Get absolute documents directory path."""
        return os.path.join(self.data_path, self.docs_subdir)
    
    @property
    def index_path(self) -> str:
        """Get absolute index directory path."""
        return os.path.join(self.data_path, self.index_subdir)
    
    @property
    def index_file(self) -> str:
        """Get absolute index file path."""
        return os.path.join(self.index_path, "index.jsonl")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for dependency injection
def get_config() -> Settings:
    """Get settings for FastAPI dependency injection."""
    return get_settings()

