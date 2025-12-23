"""
nAI RAG Kit
===========

A collection of utilities for Retrieval-Augmented Generation:
- Chunkers: Text splitting strategies
- Rerankers: Result reranking models
- Evaluators: Quality metrics for RAG systems
- Prompts: Templates for LLM interactions
"""

__version__ = "0.1.0"

from .chunkers import (
    CharacterChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
)

from .rerankers import (
    CrossEncoderReranker,
    CohereReranker,
)

from .evaluators import (
    RetrievalMetrics,
    AnswerQualityMetrics,
)

__all__ = [
    # Chunkers
    "CharacterChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Rerankers
    "CrossEncoderReranker",
    "CohereReranker",
    # Evaluators
    "RetrievalMetrics",
    "AnswerQualityMetrics",
]

