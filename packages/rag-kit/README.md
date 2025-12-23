# nAI RAG Kit

Reusable components for Retrieval-Augmented Generation systems.

## Features

- **Chunkers**: Text splitting strategies (character, sentence, recursive, semantic)
- **Rerankers**: Result reranking models (cross-encoder, Cohere)
- **Evaluators**: Quality metrics (Recall, Precision, MRR, NDCG, Answer Quality)

## Installation

```bash
pip install nai-rag-kit

# With rerankers
pip install nai-rag-kit[rerankers]

# With evaluators (requires LLM access)
pip install nai-rag-kit[evaluators]

# Everything
pip install nai-rag-kit[all]
```

## Usage

### Chunkers

```python
from rag_kit import SentenceChunker, SemanticChunker

# Sentence-aware chunking
chunker = SentenceChunker(
    max_chunk_size=1000,
    min_chunk_size=100,
    overlap_sentences=1
)
chunks = chunker.chunk(document_text)

# Semantic chunking (groups by topic)
semantic = SemanticChunker(
    max_chunk_size=1000,
    similarity_threshold=0.5
)
chunks = semantic.chunk(document_text)
```

### Rerankers

```python
from rag_kit import CrossEncoderReranker

# Cross-encoder reranking
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

results = reranker.rerank(
    query="What is machine learning?",
    documents=["Doc 1...", "Doc 2...", "Doc 3..."],
    top_k=5
)

for r in results:
    print(f"{r.index}: {r.rerank_score:.3f}")
```

### Evaluators

```python
from rag_kit import RetrievalMetrics

metrics = RetrievalMetrics(k_values=[1, 3, 5, 10])

# Single query evaluation
result = metrics.evaluate_single(
    retrieved=["doc1", "doc2", "doc3"],
    relevant=["doc1", "doc5"]
)
print(f"Recall@5: {result.recall_at_k[5]:.3f}")
print(f"MRR: {result.mrr:.3f}")

# Batch evaluation
queries = [
    {"retrieved": [...], "relevant": [...]},
    {"retrieved": [...], "relevant": [...]},
]
aggregate = metrics.evaluate_batch(queries)
print(aggregate.summary())
```

## Chunking Strategies

| Chunker | Best For |
|---------|----------|
| `CharacterChunker` | Simple, fast processing |
| `SentenceChunker` | General-purpose with sentence boundaries |
| `RecursiveChunker` | Structured documents (headers, paragraphs) |
| `SemanticChunker` | Topic-coherent chunks (requires embeddings) |

## License

AGPL-3.0-only
