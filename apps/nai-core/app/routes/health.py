"""
nAI Core Health Routes
Health check and system status endpoints
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends

from ..config import Settings, get_config
from ..models.schemas import HealthResponse
from ..services.indexer import IndexerService, get_indexer
from ..services.retriever import RetrieverService, get_retriever
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..services.answerer import AnswererService, get_answerer

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    answerer: AnswererService = Depends(get_answerer),
) -> HealthResponse:
    """
    Health check endpoint.
    Returns system status and component health.
    """
    components: Dict[str, bool] = {}
    
    # Check index
    try:
        indexer.load_index()
        components["index"] = True
    except Exception:
        components["index"] = False
    
    # Check embeddings (if enabled)
    if settings.embeddings_enabled:
        components["embeddings"] = embeddings.is_available()
    
    # Check LLM (if enabled)
    if settings.llm_enabled:
        components["llm"] = answerer.is_llm_available()
    
    # Determine overall status
    status = "ok"
    if False in components.values():
        status = "degraded"
    
    return HealthResponse(
        status=status,
        time=datetime.utcnow().isoformat() + "Z",
        version=settings.app_version,
        components=components,
    )


@router.get("/health/detailed")
async def detailed_health(
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    retriever: RetrieverService = Depends(get_retriever),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> Dict[str, Any]:
    """
    Detailed health and statistics endpoint.
    """
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "config": {
            "debug": settings.debug,
            "embeddings_enabled": settings.embeddings_enabled,
            "qdrant_enabled": settings.qdrant_enabled,
            "llm_enabled": settings.llm_enabled,
            "llm_provider": settings.llm_provider if settings.llm_enabled else None,
            "auth_enabled": settings.auth_enabled,
        },
        "index": indexer.get_stats(),
        "retriever": retriever.get_stats(),
        "embeddings": embeddings.get_stats() if settings.embeddings_enabled else None,
    }

