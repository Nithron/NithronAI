"""
nAI RAG Kit - Rerankers
=======================

Reranking models for improving retrieval quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RankedResult:
    """A reranked search result."""
    text: str
    original_score: float
    rerank_score: float
    index: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseReranker(ABC):
    """Base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of results to return (None = all)
        
        Returns:
            List of RankedResult sorted by rerank_score descending
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based reranker.
    
    Uses a cross-encoder model to compute query-document relevance.
    More accurate than bi-encoders but slower.
    
    Requires: sentence-transformers
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )
        return self._model
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        if not documents:
            return []
        
        model = self._get_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = model.predict(pairs, batch_size=self.batch_size)
        
        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RankedResult(
                text=doc,
                original_score=original_scores[i] if original_scores else 0.0,
                rerank_score=float(score),
                index=i,
            ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API based reranker.
    
    Uses Cohere's hosted reranking model.
    Requires: cohere API key
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        top_n: int = 10
    ):
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self._client = None
    
    def _get_client(self):
        """Lazy initialize Cohere client."""
        if self._client is None:
            import cohere
            self._client = cohere.Client(self.api_key)
        return self._client
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        if not documents:
            return []
        
        client = self._get_client()
        
        # Call Cohere Rerank API
        response = client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or self.top_n,
        )
        
        # Create results
        results = []
        for result in response.results:
            results.append(RankedResult(
                text=documents[result.index],
                original_score=original_scores[result.index] if original_scores else 0.0,
                rerank_score=result.relevance_score,
                index=result.index,
            ))
        
        return results


class EnsembleReranker(BaseReranker):
    """
    Ensemble reranker that combines multiple reranking methods.
    
    Uses weighted scoring from multiple rerankers.
    """
    
    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]],  # (reranker, weight)
    ):
        self.rerankers = rerankers
        total_weight = sum(w for _, w in rerankers)
        self.normalized_weights = [(r, w / total_weight) for r, w in rerankers]
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        if not documents:
            return []
        
        # Collect scores from all rerankers
        all_scores = {i: 0.0 for i in range(len(documents))}
        
        for reranker, weight in self.normalized_weights:
            results = reranker.rerank(query, documents, original_scores)
            
            # Normalize scores to [0, 1] range
            if results:
                min_score = min(r.rerank_score for r in results)
                max_score = max(r.rerank_score for r in results)
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                for result in results:
                    normalized = (result.rerank_score - min_score) / score_range
                    all_scores[result.index] += normalized * weight
        
        # Create final results
        results = []
        for i, doc in enumerate(documents):
            results.append(RankedResult(
                text=doc,
                original_score=original_scores[i] if original_scores else 0.0,
                rerank_score=all_scores[i],
                index=i,
            ))
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results

