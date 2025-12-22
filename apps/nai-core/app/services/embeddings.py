"""
nAI Core Embedding Service
Vector embeddings with sentence-transformers and Qdrant integration
"""

from typing import List, Dict, Any, Optional
import os
import uuid

from ..config import Settings, get_settings
from ..models.schemas import SearchResult, ChunkRecord
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for embedding-based search.
    Uses sentence-transformers for embeddings and Qdrant for storage.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._qdrant_client = None
        self._initialized = False
    
    def _init_model(self) -> bool:
        """Initialize the embedding model."""
        if self._model is not None:
            return True
        
        if not self.settings.embeddings_enabled:
            return False
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.settings.embeddings_model}")
            self._model = SentenceTransformer(self.settings.embeddings_model)
            
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def _init_qdrant(self) -> bool:
        """Initialize Qdrant client."""
        if self._qdrant_client is not None:
            return True
        
        if not self.settings.qdrant_enabled:
            return False
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self._qdrant_client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )
            
            # Ensure collection exists
            collections = self._qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.settings.qdrant_collection not in collection_names:
                self._qdrant_client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.settings.embeddings_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.settings.qdrant_collection}")
            
            return True
        except ImportError:
            logger.warning("qdrant-client not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize embedding service (model + Qdrant)."""
        if self._initialized:
            return True
        
        model_ok = self._init_model()
        qdrant_ok = self._init_qdrant()
        
        self._initialized = model_ok and qdrant_ok
        return self._initialized
    
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self.settings.embeddings_enabled and self.initialize()
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not self._init_model():
            return None
        
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self._init_model():
            return None
        
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return None
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index chunks with embeddings in Qdrant.
        Returns number of chunks indexed.
        """
        if not self.initialize():
            return 0
        
        from qdrant_client.models import PointStruct
        
        # Generate embeddings
        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embed_texts(texts)
        
        if embeddings is None:
            return 0
        
        # Create points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "doc_id": chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id", ""),
                        "doc_path": chunk.get("doc_path", ""),
                        "chunk_id": chunk.get("chunk_id", 0),
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {}),
                    }
                )
            )
        
        # Upsert to Qdrant
        try:
            self._qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
            )
            logger.info(f"Indexed {len(points)} chunks in Qdrant")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to index in Qdrant: {e}")
            return 0
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search using embeddings.
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score
        
        Returns:
            List of SearchResult objects
        """
        if not self.initialize():
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []
        
        try:
            # Search Qdrant
            results = self._qdrant_client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )
            
            # Convert to SearchResult
            search_results = []
            for hit in results:
                payload = hit.payload or {}
                search_results.append(SearchResult(
                    doc_id=payload.get("doc_id", ""),
                    doc_path=payload.get("doc_path", ""),
                    chunk_id=payload.get("chunk_id", 0),
                    text=payload.get("text", ""),
                    score=hit.score,
                    metadata=payload.get("metadata", {}),
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all vectors for a document."""
        if not self._init_qdrant():
            return False
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            self._qdrant_client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted embeddings for doc_id: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        stats = {
            "enabled": self.settings.embeddings_enabled,
            "model": self.settings.embeddings_model,
            "qdrant_enabled": self.settings.qdrant_enabled,
        }
        
        if self._init_qdrant():
            try:
                info = self._qdrant_client.get_collection(
                    collection_name=self.settings.qdrant_collection
                )
                stats["qdrant_vectors_count"] = info.vectors_count
                stats["qdrant_points_count"] = info.points_count
            except Exception:
                pass
        
        return stats


# Singleton instance
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(get_settings())
    return _embedding_service_instance

