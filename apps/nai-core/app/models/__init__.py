"""
nAI Core Models
Pydantic schemas and data models
"""

from .schemas import (
    # Health
    HealthResponse,
    # Documents
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    ChunkRecord,
    # Ingestion
    IngestFileResult,
    IngestResponse,
    # Search & Ask
    AskRequest,
    SearchRequest,
    Citation,
    SearchResult,
    AskResponse,
    SearchResponse,
    # Chat
    ChatMessage,
    ChatRequest,
    ChatResponse,
    # Auth
    UserCreate,
    UserLogin,
    User,
    Token,
    TokenData,
)

__all__ = [
    "HealthResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "DocumentDeleteResponse",
    "ChunkRecord",
    "IngestFileResult",
    "IngestResponse",
    "AskRequest",
    "SearchRequest",
    "Citation",
    "SearchResult",
    "AskResponse",
    "SearchResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "UserCreate",
    "UserLogin",
    "User",
    "Token",
    "TokenData",
]

