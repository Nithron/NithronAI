"""
nAI Core Pydantic Schemas
Request/Response models for API endpoints
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import hashlib


# ============================================================================
# Health
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok", "degraded", "error"] = "ok"
    time: str
    version: str
    components: Dict[str, bool] = Field(default_factory=dict)


# ============================================================================
# Documents
# ============================================================================

class ChunkRecord(BaseModel):
    """A single chunk in the index."""
    doc_id: str = Field(..., description="Unique document identifier (hash)")
    doc_path: str = Field(..., description="Relative path to document")
    chunk_id: int = Field(..., description="Chunk index within document")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    @classmethod
    def create(cls, doc_path: str, chunk_id: int, text: str, metadata: Optional[Dict] = None):
        """Create a chunk record with auto-generated doc_id."""
        doc_id = hashlib.sha256(doc_path.encode()).hexdigest()[:16]
        return cls(
            doc_id=doc_id,
            doc_path=doc_path,
            chunk_id=chunk_id,
            text=text,
            metadata=metadata or {}
        )


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    path: str = Field(..., description="Relative path in storage")
    chunk_count: int = Field(..., description="Number of chunks")
    total_chars: int = Field(..., description="Total character count")
    file_size: int = Field(0, description="File size in bytes")
    file_type: str = Field(..., description="File extension/type")
    indexed_at: str = Field(..., description="Indexing timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[DocumentInfo]
    total: int
    page: int = 1
    page_size: int = 50


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""
    success: bool
    doc_id: str
    chunks_removed: int
    message: str


# ============================================================================
# Ingestion
# ============================================================================

class IngestFileResult(BaseModel):
    """Result for a single ingested file."""
    filename: str
    doc_id: str
    chunks: int
    chars: int
    status: Literal["success", "skipped", "error"]
    message: Optional[str] = None


class IngestResponse(BaseModel):
    """Response for document ingestion."""
    added: List[IngestFileResult]
    total_chunks: int
    total_files: int
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# Search & Ask
# ============================================================================

class AskRequest(BaseModel):
    """Request for asking a question."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_llm: bool = Field(default=True, description="Use LLM for answer generation if available")
    include_sources: bool = Field(default=True, description="Include source chunks in response")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")
    
    @field_validator('question')
    @classmethod
    def clean_question(cls, v: str) -> str:
        return v.strip()


class SearchRequest(BaseModel):
    """Request for raw search (no answer generation)."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    method: Literal["bm25", "embedding", "hybrid"] = Field(default="bm25")
    filters: Optional[Dict[str, Any]] = Field(default=None)


class Citation(BaseModel):
    """A citation/source reference."""
    doc_id: str
    doc_path: str
    chunk_id: int
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result."""
    doc_id: str
    doc_path: str
    chunk_id: int
    text: str
    score: float
    highlight: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response for a question."""
    answer: str
    citations: List[Citation]
    method: str = Field(default="extractive", description="Answer generation method")
    model: Optional[str] = Field(default=None, description="LLM model used if applicable")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    tokens_used: Optional[int] = Field(default=None)


class SearchResponse(BaseModel):
    """Response for raw search."""
    results: List[SearchResult]
    total: int
    method: str
    query: str


# ============================================================================
# Chat (Multi-turn conversation)
# ============================================================================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    messages: List[ChatMessage]
    top_k: int = Field(default=5, ge=1, le=20)
    use_context: bool = Field(default=True, description="Include document context")
    stream: bool = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    """Response for chat endpoint."""
    message: ChatMessage
    citations: List[Citation] = Field(default_factory=list)
    conversation_id: Optional[str] = None


# ============================================================================
# Authentication
# ============================================================================

class UserCreate(BaseModel):
    """User creation request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class User(BaseModel):
    """User information (public)."""
    id: str
    username: str
    email: str
    is_active: bool = True
    is_admin: bool = False
    created_at: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    exp: Optional[int] = None

