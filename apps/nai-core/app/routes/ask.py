"""
nAI Core Ask Routes
Question answering and search endpoints
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..config import Settings, get_config
from ..models.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
)
from ..services.indexer import IndexerService, get_indexer
from ..services.retriever import RetrieverService, get_retriever
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..services.answerer import AnswererService, get_answerer
from ..utils.logging import get_logger

router = APIRouter(tags=["Search & Ask"])
logger = get_logger(__name__)


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    retriever: RetrieverService = Depends(get_retriever),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    answerer: AnswererService = Depends(get_answerer),
) -> AskResponse:
    """
    Ask a question about your indexed documents.
    
    The system will:
    1. Search for relevant passages using BM25 (and embeddings if enabled)
    2. Generate an answer using LLM (if enabled) or provide extractive excerpts
    3. Return the answer with citations
    """
    # Load index
    records = indexer.load_index()
    
    if not records:
        return AskResponse(
            answer="No documents have been indexed yet. Please upload some documents first using the /ingest endpoint.",
            citations=[],
            method="none",
        )
    
    # Search for relevant passages
    if settings.embeddings_enabled and embeddings.is_available():
        # Hybrid search
        embedding_results = embeddings.search(request.question, top_k=request.top_k)
        results = retriever.search_hybrid(
            request.question,
            records,
            top_k=request.top_k,
            embedding_results=embedding_results,
        )
    else:
        # BM25 only
        results = retriever.search_bm25(request.question, records, top_k=request.top_k)
    
    # Convert to citations
    citations = retriever.results_to_citations(results)
    
    # Generate answer
    response = await answerer.answer(
        request.question,
        citations,
        use_llm=request.use_llm,
    )
    
    return response


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    retriever: RetrieverService = Depends(get_retriever),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> SearchResponse:
    """
    Search indexed documents without answer generation.
    
    Returns raw search results with relevance scores.
    Useful for exploring the corpus or building custom applications.
    """
    # Load index
    records = indexer.load_index()
    
    if not records:
        return SearchResponse(
            results=[],
            total=0,
            method="none",
            query=request.query,
        )
    
    # Search based on method
    if request.method == "embedding":
        if not (settings.embeddings_enabled and embeddings.is_available()):
            raise HTTPException(
                status_code=400,
                detail="Embedding search is not enabled. Set NAI_EMBEDDINGS_ENABLED=true"
            )
        results = embeddings.search(request.query, top_k=request.top_k)
        method = "embedding"
    
    elif request.method == "hybrid":
        if settings.embeddings_enabled and embeddings.is_available():
            embedding_results = embeddings.search(request.query, top_k=request.top_k)
            results = retriever.search_hybrid(
                request.query,
                records,
                top_k=request.top_k,
                embedding_results=embedding_results,
            )
            method = "hybrid"
        else:
            results = retriever.search_bm25(request.query, records, top_k=request.top_k)
            method = "bm25"
    
    else:
        # Default: BM25
        results = retriever.search_bm25(request.query, records, top_k=request.top_k)
        method = "bm25"
    
    return SearchResponse(
        results=results,
        total=len(results),
        method=method,
        query=request.query,
    )

