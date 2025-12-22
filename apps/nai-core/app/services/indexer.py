"""
nAI Core Indexer Service
Document indexing and management
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from functools import lru_cache

from fastapi import UploadFile

from ..config import Settings, get_settings
from ..models.schemas import (
    ChunkRecord,
    DocumentInfo,
    IngestFileResult,
)
from ..utils.text import chunk_text_semantic, normalize_text
from ..utils.extractors import extract_text, get_file_hash
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IndexerService:
    """
    Service for document indexing and management.
    Handles file storage, text extraction, chunking, and index management.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._ensure_directories()
        self._index_cache: Optional[List[Dict[str, Any]]] = None
        self._index_mtime: float = 0
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.settings.docs_path, exist_ok=True)
        os.makedirs(self.settings.index_path, exist_ok=True)
    
    def _get_doc_id(self, content: bytes) -> str:
        """Generate document ID from content hash."""
        return hashlib.sha256(content).hexdigest()[:16]
    
    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str, bytes]:
        """
        Save an uploaded file to disk.
        Returns (saved_path, original_filename, content_bytes).
        """
        content = await file.read()
        
        # Check file size
        if len(content) > self.settings.max_file_size_bytes:
            raise ValueError(
                f"File too large: {len(content)} bytes "
                f"(max: {self.settings.max_file_size_mb}MB)"
            )
        
        # Check extension
        original_name = file.filename or "unknown"
        ext = Path(original_name).suffix.lower()
        if ext not in self.settings.allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Allowed: {', '.join(self.settings.allowed_extensions)}"
            )
        
        # Generate unique filename
        doc_id = self._get_doc_id(content)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in original_name)
        saved_name = f"{timestamp}_{doc_id}_{safe_name}"
        saved_path = os.path.join(self.settings.docs_path, saved_name)
        
        # Save file
        with open(saved_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved file: {saved_name} ({len(content)} bytes)")
        return saved_path, original_name, content
    
    async def ingest_file(self, file: UploadFile) -> IngestFileResult:
        """
        Ingest a single file: save, extract, chunk, and index.
        """
        try:
            # Save file
            saved_path, original_name, content = await self.save_uploaded_file(file)
            doc_id = self._get_doc_id(content)
            
            # Check for duplicates
            if self._is_duplicate(doc_id):
                logger.info(f"Skipping duplicate: {original_name}")
                return IngestFileResult(
                    filename=original_name,
                    doc_id=doc_id,
                    chunks=0,
                    chars=0,
                    status="skipped",
                    message="Document already indexed"
                )
            
            # Extract text
            text, metadata = extract_text(saved_path)
            text = normalize_text(text)
            
            if not text.strip():
                return IngestFileResult(
                    filename=original_name,
                    doc_id=doc_id,
                    chunks=0,
                    chars=0,
                    status="skipped",
                    message="No text content extracted"
                )
            
            # Chunk text
            chunks = chunk_text_semantic(
                text,
                max_chars=self.settings.chunk_size,
                overlap_sentences=1
            )
            
            # Create chunk records
            relative_path = os.path.relpath(saved_path, self.settings.docs_path)
            records = []
            for i, chunk_text in enumerate(chunks):
                record = ChunkRecord.create(
                    doc_path=relative_path,
                    chunk_id=i,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "original_filename": original_name,
                        **{k: v for k, v in metadata.items() if k != "error"}
                    }
                )
                records.append(record)
            
            # Append to index
            self._append_to_index(records)
            
            # Invalidate cache
            self._index_cache = None
            
            logger.info(f"Indexed: {original_name} -> {len(chunks)} chunks")
            
            return IngestFileResult(
                filename=original_name,
                doc_id=doc_id,
                chunks=len(chunks),
                chars=len(text),
                status="success"
            )
        
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return IngestFileResult(
                filename=file.filename or "unknown",
                doc_id="",
                chunks=0,
                chars=0,
                status="error",
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return IngestFileResult(
                filename=file.filename or "unknown",
                doc_id="",
                chunks=0,
                chars=0,
                status="error",
                message=f"Processing error: {str(e)}"
            )
    
    def _is_duplicate(self, doc_id: str) -> bool:
        """Check if document is already indexed."""
        index = self.load_index()
        for record in index:
            if record.get("metadata", {}).get("doc_id") == doc_id:
                return True
        return False
    
    def _append_to_index(self, records: List[ChunkRecord]) -> None:
        """Append chunk records to the index file."""
        with open(self.settings.index_file, "a", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(record.model_dump(), ensure_ascii=False)
                f.write(line + "\n")
    
    def load_index(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Load index from disk with caching.
        Reloads if file has been modified.
        """
        if not os.path.exists(self.settings.index_file):
            return []
        
        # Check if reload needed
        mtime = os.path.getmtime(self.settings.index_file)
        if not force_reload and self._index_cache is not None and mtime <= self._index_mtime:
            return self._index_cache
        
        # Load from disk
        records = []
        with open(self.settings.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        self._index_cache = records
        self._index_mtime = mtime
        logger.debug(f"Loaded index: {len(records)} chunks")
        
        return records
    
    def invalidate_cache(self) -> None:
        """Invalidate the index cache."""
        self._index_cache = None
    
    def list_documents(self) -> List[DocumentInfo]:
        """
        List all indexed documents with aggregated info.
        """
        index = self.load_index()
        
        # Aggregate by document
        docs: Dict[str, Dict[str, Any]] = {}
        
        for record in index:
            doc_path = record.get("doc_path", "")
            doc_id = record.get("doc_id") or record.get("metadata", {}).get("doc_id", "")
            
            if doc_path not in docs:
                metadata = record.get("metadata", {})
                docs[doc_path] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("original_filename", os.path.basename(doc_path)),
                    "path": doc_path,
                    "chunks": 0,
                    "total_chars": 0,
                    "file_type": metadata.get("extension", Path(doc_path).suffix),
                    "indexed_at": record.get("created_at", ""),
                    "metadata": metadata,
                }
            
            docs[doc_path]["chunks"] += 1
            docs[doc_path]["total_chars"] += len(record.get("text", ""))
        
        # Convert to list of DocumentInfo
        return [
            DocumentInfo(
                doc_id=info["doc_id"],
                filename=info["filename"],
                path=info["path"],
                chunk_count=info["chunks"],
                total_chars=info["total_chars"],
                file_type=info["file_type"],
                indexed_at=info["indexed_at"],
                metadata=info["metadata"],
            )
            for info in docs.values()
        ]
    
    def delete_document(self, doc_id: str) -> Tuple[bool, int]:
        """
        Delete a document from the index by doc_id.
        Returns (success, chunks_removed).
        """
        if not os.path.exists(self.settings.index_file):
            return False, 0
        
        index = self.load_index()
        
        # Filter out chunks for this document
        remaining = []
        removed_count = 0
        doc_path_to_delete = None
        
        for record in index:
            record_doc_id = record.get("doc_id") or record.get("metadata", {}).get("doc_id", "")
            if record_doc_id == doc_id:
                removed_count += 1
                if doc_path_to_delete is None:
                    doc_path_to_delete = record.get("doc_path")
            else:
                remaining.append(record)
        
        if removed_count == 0:
            return False, 0
        
        # Rewrite index file
        with open(self.settings.index_file, "w", encoding="utf-8") as f:
            for record in remaining:
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
        
        # Optionally delete the source file
        if doc_path_to_delete:
            full_path = os.path.join(self.settings.docs_path, doc_path_to_delete)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    logger.info(f"Deleted source file: {doc_path_to_delete}")
                except Exception as e:
                    logger.warning(f"Could not delete source file: {e}")
        
        # Invalidate cache
        self._index_cache = None
        
        logger.info(f"Deleted document {doc_id}: {removed_count} chunks")
        return True, removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        index = self.load_index()
        docs = self.list_documents()
        
        total_chars = sum(len(r.get("text", "")) for r in index)
        
        return {
            "total_documents": len(docs),
            "total_chunks": len(index),
            "total_characters": total_chars,
            "index_file": self.settings.index_file,
            "docs_directory": self.settings.docs_path,
        }


# Singleton instance
_indexer_instance: Optional[IndexerService] = None


def get_indexer() -> IndexerService:
    """Get or create the indexer service singleton."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = IndexerService(get_settings())
    return _indexer_instance

