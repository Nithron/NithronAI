"""
nAI Core Retriever Service
BM25 and hybrid search with caching
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import threading

from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings
from ..models.schemas import SearchResult, Citation
from ..utils.text import tokenize, highlight_matches
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BM25Index:
    """
    Cached BM25 index with automatic invalidation.
    Thread-safe implementation.
    """
    
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._lock = threading.RLock()
        self._version: int = 0
    
    def build(self, records: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        """Build BM25 index from records."""
        with self._lock:
            if not records:
                self._bm25 = None
                self._corpus = []
                self._tokenized = []
                return
            
            self._corpus = records
            self._tokenized = [tokenize(r.get("text", "")) for r in records]
            self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)
            self._version += 1
            
            logger.debug(f"Built BM25 index: {len(records)} documents, version {self._version}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the index.
        Returns list of (index, score) tuples.
        """
        with self._lock:
            if self._bm25 is None or not self._corpus:
                return []
            
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            
            scores = self._bm25.get_scores(query_tokens)
            
            # Get top-k results
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return [(idx, score) for idx, score in ranked[:top_k] if score > 0]
    
    def get_document(self, index: int) -> Optional[Dict[str, Any]]:
        """Get document by index."""
        with self._lock:
            if 0 <= index < len(self._corpus):
                return self._corpus[index]
            return None
    
    @property
    def size(self) -> int:
        """Number of documents in index."""
        with self._lock:
            return len(self._corpus)
    
    @property
    def version(self) -> int:
        """Index version for cache invalidation."""
        return self._version


class RetrieverService:
    """
    Service for document retrieval.
    Supports BM25, embeddings (when enabled), and hybrid search.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._bm25_index = BM25Index()
        self._index_hash: str = ""
    
    def _get_index_hash(self, records: List[Dict[str, Any]]) -> str:
        """Generate a hash to detect index changes."""
        return f"{len(records)}:{records[-1].get('created_at', '') if records else ''}"
    
    def ensure_index(self, records: List[Dict[str, Any]]) -> None:
        """
        Ensure BM25 index is built and up-to-date.
        Rebuilds if records have changed.
        """
        new_hash = self._get_index_hash(records)
        
        if new_hash != self._index_hash:
            logger.info("Rebuilding BM25 index...")
            self._bm25_index.build(
                records,
                k1=self.settings.bm25_k1,
                b=self.settings.bm25_b
            )
            self._index_hash = new_hash
    
    def search_bm25(
        self,
        query: str,
        records: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            records: Index records (will build/update index if needed)
            top_k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        # Ensure index is current
        self.ensure_index(records)
        
        # Search
        hits = self._bm25_index.search(query, top_k)
        
        # Convert to SearchResult
        results = []
        for idx, score in hits:
            doc = self._bm25_index.get_document(idx)
            if doc:
                results.append(SearchResult(
                    doc_id=doc.get("doc_id") or doc.get("metadata", {}).get("doc_id", ""),
                    doc_path=doc.get("doc_path", ""),
                    chunk_id=doc.get("chunk_id", 0),
                    text=doc.get("text", ""),
                    score=score,
                    highlight=highlight_matches(doc.get("text", ""), query),
                    metadata=doc.get("metadata", {}),
                ))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        records: List[Dict[str, Any]],
        top_k: int = 10,
        bm25_weight: float = 0.5,
        embedding_results: Optional[List[SearchResult]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and embedding results.
        
        Args:
            query: Search query
            records: Index records
            top_k: Number of final results
            bm25_weight: Weight for BM25 scores (1 - bm25_weight for embeddings)
            embedding_results: Pre-computed embedding search results
        
        Returns:
            List of SearchResult objects with combined scores
        """
        # Get BM25 results
        bm25_results = self.search_bm25(query, records, top_k=top_k * 2)
        
        if not embedding_results:
            return bm25_results[:top_k]
        
        # Normalize scores for each method (min-max normalization)
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            return {
                f"{r.doc_path}:{r.chunk_id}": (r.score - min_score) / score_range
                for r in results
            }
        
        bm25_normalized = normalize_scores(bm25_results)
        embed_normalized = normalize_scores(embedding_results)
        
        # Combine scores
        combined: Dict[str, Tuple[SearchResult, float]] = {}
        embedding_weight = 1.0 - bm25_weight
        
        for result in bm25_results:
            key = f"{result.doc_path}:{result.chunk_id}"
            bm25_score = bm25_normalized.get(key, 0)
            embed_score = embed_normalized.get(key, 0)
            combined_score = bm25_weight * bm25_score + embedding_weight * embed_score
            combined[key] = (result, combined_score)
        
        for result in embedding_results:
            key = f"{result.doc_path}:{result.chunk_id}"
            if key not in combined:
                bm25_score = bm25_normalized.get(key, 0)
                embed_score = embed_normalized.get(key, 0)
                combined_score = bm25_weight * bm25_score + embedding_weight * embed_score
                combined[key] = (result, combined_score)
        
        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        
        # Return top-k with updated scores
        return [
            SearchResult(
                doc_id=r.doc_id,
                doc_path=r.doc_path,
                chunk_id=r.chunk_id,
                text=r.text,
                score=score,
                highlight=r.highlight,
                metadata=r.metadata,
            )
            for r, score in sorted_results[:top_k]
        ]
    
    def results_to_citations(self, results: List[SearchResult]) -> List[Citation]:
        """Convert search results to citations."""
        return [
            Citation(
                doc_id=r.doc_id,
                doc_path=r.doc_path,
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "bm25_index_size": self._bm25_index.size,
            "bm25_index_version": self._bm25_index.version,
            "bm25_k1": self.settings.bm25_k1,
            "bm25_b": self.settings.bm25_b,
            "embeddings_enabled": self.settings.embeddings_enabled,
        }


# Singleton instance
_retriever_instance: Optional[RetrieverService] = None


def get_retriever() -> RetrieverService:
    """Get or create the retriever service singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverService(get_settings())
    return _retriever_instance

