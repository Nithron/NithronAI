"""
nAI Core Ingest Routes
Document ingestion and indexing endpoints
"""

from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query

from ..config import Settings, get_config
from ..models.schemas import IngestResponse, IngestFileResult
from ..services.indexer import IndexerService, get_indexer
from ..services.embeddings import EmbeddingService, get_embedding_service
from ..utils.logging import get_logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)


@router.post("", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Files to ingest"),
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> IngestResponse:
    """
    Ingest one or more documents.
    
    Supported formats:
    - PDF (.pdf)
    - Markdown (.md, .markdown)
    - Plain text (.txt)
    - HTML (.html, .htm)
    - RST (.rst)
    
    Files are extracted, chunked, and indexed for search.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results: List[IngestFileResult] = []
    errors: List[str] = []
    total_chunks = 0
    
    for file in files:
        try:
            result = await indexer.ingest_file(file)
            results.append(result)
            
            if result.status == "success":
                total_chunks += result.chunks
                
                # Index embeddings if enabled
                if settings.embeddings_enabled and embeddings.is_available():
                    index = indexer.load_index()
                    # Get chunks for this document
                    doc_chunks = [
                        r for r in index
                        if r.get("metadata", {}).get("doc_id") == result.doc_id
                    ]
                    if doc_chunks:
                        embeddings.index_chunks(doc_chunks)
            
            elif result.status == "error":
                errors.append(f"{file.filename}: {result.message}")
        
        except Exception as e:
            logger.error(f"Error ingesting {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")
            results.append(IngestFileResult(
                filename=file.filename or "unknown",
                doc_id="",
                chunks=0,
                chars=0,
                status="error",
                message=str(e),
            ))
    
    return IngestResponse(
        added=results,
        total_chunks=total_chunks,
        total_files=len([r for r in results if r.status == "success"]),
        errors=errors,
    )


@router.post("/url")
async def ingest_from_url(
    url: str = Query(..., description="URL to fetch and ingest"),
    settings: Settings = Depends(get_config),
    indexer: IndexerService = Depends(get_indexer),
) -> IngestResponse:
    """
    Ingest a document from a URL.
    
    Currently supports:
    - Plain text URLs
    - HTML pages (will extract text)
    """
    import aiohttp
    import tempfile
    import os
    from urllib.parse import urlparse
    
    try:
        # Parse URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        # Fetch content
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch URL: HTTP {response.status}"
                    )
                
                content = await response.read()
                content_type = response.headers.get("content-type", "")
        
        # Determine extension
        if "html" in content_type:
            ext = ".html"
        elif "pdf" in content_type:
            ext = ".pdf"
        else:
            ext = ".txt"
        
        # Save to temp file
        filename = f"url_{parsed.netloc.replace('.', '_')}{ext}"
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Create UploadFile-like object
            class FakeUploadFile:
                def __init__(self, path: str, name: str):
                    self.filename = name
                    self._path = path
                    self._file = open(path, "rb")
                    self.file = self._file
                
                async def read(self):
                    return self._file.read()
                
                def close(self):
                    self._file.close()
            
            fake_file = FakeUploadFile(tmp_path, filename)
            result = await indexer.ingest_file(fake_file)
            fake_file.close()
            
            return IngestResponse(
                added=[result],
                total_chunks=result.chunks,
                total_files=1 if result.status == "success" else 0,
                errors=[result.message] if result.status == "error" else [],
            )
        finally:
            os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

