"""
nAI Core - Local-first AI Document Q&A System

A privacy-focused, open-core AI stack for document ingestion,
search, and question answering.

Features:
- Document ingestion (PDF, Markdown, TXT, HTML)
- BM25 full-text search
- Optional embedding-based semantic search (Qdrant)
- Optional LLM-powered answer generation (Ollama, OpenAI, etc.)
- JWT authentication
- Multi-turn chat
- RESTful API with OpenAPI documentation

For configuration, set environment variables with NAI_ prefix.
See config.py for all available options.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware import setup_middleware
from .utils.logging import setup_logging, get_logger
from .routes import (
    health_router,
    ingest_router,
    ask_router,
    documents_router,
    chat_router,
    auth_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()
    logger = get_logger(__name__)
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Embeddings enabled: {settings.embeddings_enabled}")
    logger.info(f"LLM enabled: {settings.llm_enabled}")
    logger.info(f"Auth enabled: {settings.auth_enabled}")
    
    # Initialize services (they're singletons, so this warms them up)
    from .services import get_indexer, get_retriever, get_answerer, get_embedding_service
    
    indexer = get_indexer()
    retriever = get_retriever()
    answerer = get_answerer()
    
    # Pre-load index
    records = indexer.load_index()
    logger.info(f"Loaded {len(records)} chunks from index")
    
    # Pre-build BM25 index if we have data
    if records:
        retriever.ensure_index(records)
    
    # Initialize embeddings if enabled
    if settings.embeddings_enabled:
        embedding_service = get_embedding_service()
        if embedding_service.initialize():
            logger.info("Embedding service initialized")
        else:
            logger.warning("Embedding service failed to initialize")
    
    yield
    
    # Shutdown
    logger.info("Shutting down nAI Core")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_format=not settings.debug,
    )
    
    # Create app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=__doc__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS middleware
    origins = ["*"] if settings.cors_allow_all else settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    
    # Custom middleware
    setup_middleware(app)
    
    # Register routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(ask_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(auth_router)
    
    return app


# Create the application instance
app = create_app()


# For running directly with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
