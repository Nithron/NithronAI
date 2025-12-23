# nAI Retrieval Evaluation

Framework for evaluating retrieval quality in nAI.

## Metrics

- **Recall@K**: Fraction of relevant documents retrieved in top K
- **Precision@K**: Fraction of top K documents that are relevant
- **MRR**: Mean Reciprocal Rank - average of 1/rank of first relevant doc
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of queries with at least one relevant in top K

## Usage

### Create Test Cases

Create a JSON file with test cases:

```json
[
  {
    "query": "What is machine learning?",
    "relevant_docs": ["ml_intro.pdf", "ai_basics.md"],
    "expected_answer": "Machine learning is...",
    "metadata": {
      "category": "technical",
      "difficulty": "easy"
    }
  }
]
```

### Run Evaluation

```bash
python eval_retrieval.py \
  --test-file test_cases.json \
  --api-url http://localhost:8000 \
  --output results.json \
  --k-values 1,3,5,10
```

### Output

```
==================================================
AGGREGATE METRICS
==================================================
Queries: 10/10 successful
MRR: 0.7500
Recall@1: 0.4000
Precision@1: 0.4000
NDCG@1: 0.4000
Recall@5: 0.8500
Precision@5: 0.3400
NDCG@5: 0.6234
```

### Programmatic Usage

```python
from eval_retrieval import RetrievalEvaluator, TestCase

evaluator = RetrievalEvaluator(api_url="http://localhost:8000")

# Single query
result = evaluator.evaluate_single(TestCase(
    query="What is AI?",
    relevant_docs=["ai.pdf", "intro.md"]
))

print(f"Recall@5: {result['recall@5']:.3f}")
print(f"MRR: {result['mrr']:.3f}")

# Batch evaluation
test_cases = [...]
results = evaluator.evaluate_batch(test_cases)
print(results["aggregate"])
```

## Test Case Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Search query or question |
| `relevant_docs` | string[] | ✅ | List of relevant document names |
| `expected_answer` | string | - | Expected answer (for future use) |
| `metadata` | object | - | Additional metadata for filtering |

## CI Integration

The evaluation runs automatically on main branch pushes:

```yaml
- name: Run evaluation
  run: |
    python evals/retrieval/eval_retrieval.py \
      --test-file evals/retrieval/test_cases.json \
      --api-url http://localhost:8000 \
      --output eval_results.json
```

Results are uploaded as artifacts for review.

## Benchmarking Tips

1. **Create diverse test cases**: Cover different query types, document formats, and difficulty levels
2. **Use realistic queries**: Match expected user behavior
3. **Track over time**: Compare metrics across versions
4. **Set thresholds**: Define minimum acceptable metrics for CI gates

## Metrics Interpretation

| Metric | Good | Excellent | Description |
|--------|------|-----------|-------------|
| MRR | > 0.5 | > 0.8 | First relevant result ranking |
| Recall@5 | > 0.6 | > 0.9 | Coverage of relevant docs |
| Precision@5 | > 0.3 | > 0.5 | Relevance of results |
| NDCG@5 | > 0.5 | > 0.8 | Ranking quality |
