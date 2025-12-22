"""
nAI Core Document Routes
Document management endpoints
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import Settings, get_config
from ..models.schemas import (
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from ..services.indexer import IndexerService, get_indexer
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..utils.logging import get_logger

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    indexer: IndexerService = Depends(get_indexer),
) -> DocumentListResponse:
    """
    List all indexed documents.
    
    Returns document metadata including chunk counts and indexing timestamps.
    Supports pagination for large document collections.
    """
    all_docs = indexer.list_documents()
    
    # Sort by indexed_at descending (newest first)
    all_docs.sort(key=lambda d: d.indexed_at, reverse=True)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    docs_page = all_docs[start:end]
    
    return DocumentListResponse(
        documents=docs_page,
        total=len(all_docs),
        page=page,
        page_size=page_size,
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    indexer: IndexerService = Depends(get_indexer),
) -> DocumentInfo:
    """
    Get details for a specific document.
    """
    docs = indexer.list_documents()
    
    for doc in docs:
        if doc.doc_id == doc_id:
            return doc
    
    raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> DocumentDeleteResponse:
    """
    Delete a document from the index.
    
    This removes all chunks and optionally the source file.
    If embeddings are enabled, also removes vectors from Qdrant.
    """
    # Delete from index
    success, chunks_removed = indexer.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    
    # Delete from embeddings if enabled
    if settings.embeddings_enabled and embeddings.is_available():
        embeddings.delete_by_doc_id(doc_id)
    
    return DocumentDeleteResponse(
        success=True,
        doc_id=doc_id,
        chunks_removed=chunks_removed,
        message=f"Successfully deleted document and {chunks_removed} chunks",
    )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    indexer: IndexerService = Depends(get_indexer),
):
    """
    Get chunks for a specific document.
    
    Useful for inspecting how a document was chunked.
    """
    index = indexer.load_index()
    
    # Filter chunks for this document
    doc_chunks = [
        {
            "chunk_id": r.get("chunk_id", 0),
            "text": r.get("text", ""),
            "created_at": r.get("created_at", ""),
        }
        for r in index
        if (r.get("doc_id") or r.get("metadata", {}).get("doc_id", "")) == doc_id
    ]
    
    if not doc_chunks:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    
    # Sort by chunk_id
    doc_chunks.sort(key=lambda c: c["chunk_id"])
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "doc_id": doc_id,
        "chunks": doc_chunks[start:end],
        "total": len(doc_chunks),
        "page": page,
        "page_size": page_size,
    }


@router.get("/stats/summary")
async def get_stats(
    indexer: IndexerService = Depends(get_indexer),
):
    """
    Get summary statistics for the document index.
    """
    return indexer.get_stats()

