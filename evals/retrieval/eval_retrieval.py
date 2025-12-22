"""
nAI Retrieval Evaluation Framework
===================================

Evaluate retrieval quality using standard IR metrics:
- Recall@K: Fraction of relevant docs in top-K
- Precision@K: Fraction of top-K docs that are relevant
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant doc
- NDCG@K: Normalized Discounted Cumulative Gain

Usage:
    python eval_retrieval.py --test-file test_cases.json --api-url http://localhost:8000

Test file format (JSON):
    [
        {
            "query": "What is X?",
            "relevant_docs": ["doc1.pdf", "doc2.md"],
            "expected_answer": "X is..." (optional)
        },
        ...
    ]
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import httpx
from datetime import datetime


@dataclass
class TestCase:
    """A single test case for evaluation."""
    query: str
    relevant_docs: List[str]
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from the search API."""
    query: str
    retrieved_docs: List[str]
    scores: List[float]
    answer: Optional[str] = None


@dataclass
class EvalMetrics:
    """Evaluation metrics for a test run."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall": self.recall_at_k,
            "precision": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg_at_k,
        }


class RetrievalEvaluator:
    """Evaluator for retrieval quality."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
    
    def search(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Execute a search query against the API."""
        response = self.client.post(
            f"{self.api_url}/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        data = response.json()
        
        return RetrievalResult(
            query=query,
            retrieved_docs=[r["doc_path"] for r in data.get("results", [])],
            scores=[r["score"] for r in data.get("results", [])],
        )
    
    def ask(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Execute a question against the ask API."""
        response = self.client.post(
            f"{self.api_url}/ask",
            json={"question": query, "top_k": top_k, "use_llm": False}
        )
        response.raise_for_status()
        data = response.json()
        
        return RetrievalResult(
            query=query,
            retrieved_docs=[c["doc_path"] for c in data.get("citations", [])],
            scores=[c["score"] for c in data.get("citations", [])],
            answer=data.get("answer"),
        )
    
    def calculate_recall_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(self._normalize_paths(retrieved[:k]))
        relevant_set = set(self._normalize_paths(relevant))
        
        hits = len(retrieved_at_k & relevant_set)
        return hits / len(relevant_set)
    
    def calculate_precision_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K."""
        retrieved_at_k = self._normalize_paths(retrieved[:k])
        if not retrieved_at_k:
            return 0.0
        
        relevant_set = set(self._normalize_paths(relevant))
        hits = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        return hits / len(retrieved_at_k)
    
    def calculate_mrr(
        self, 
        retrieved: List[str], 
        relevant: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank for a single query."""
        relevant_set = set(self._normalize_paths(relevant))
        retrieved_norm = self._normalize_paths(retrieved)
        
        for i, doc in enumerate(retrieved_norm):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K."""
        relevant_set = set(self._normalize_paths(relevant))
        retrieved_norm = self._normalize_paths(retrieved[:k])
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_norm):
            if doc in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Ideal DCG
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def _normalize_paths(self, paths: List[str]) -> List[str]:
        """Normalize document paths for comparison."""
        return [Path(p).name.lower() for p in paths]
    
    def evaluate_single(
        self, 
        test_case: TestCase, 
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate a single test case."""
        result = self.search(test_case.query, top_k=max(k_values))
        
        metrics = {
            "query": test_case.query,
            "retrieved": result.retrieved_docs[:5],
            "relevant": test_case.relevant_docs,
        }
        
        for k in k_values:
            metrics[f"recall@{k}"] = self.calculate_recall_at_k(
                result.retrieved_docs, test_case.relevant_docs, k
            )
            metrics[f"precision@{k}"] = self.calculate_precision_at_k(
                result.retrieved_docs, test_case.relevant_docs, k
            )
            metrics[f"ndcg@{k}"] = self.calculate_ndcg_at_k(
                result.retrieved_docs, test_case.relevant_docs, k
            )
        
        metrics["mrr"] = self.calculate_mrr(result.retrieved_docs, test_case.relevant_docs)
        
        return metrics
    
    def evaluate_batch(
        self, 
        test_cases: List[TestCase],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate a batch of test cases."""
        results = []
        
        for test_case in test_cases:
            try:
                result = self.evaluate_single(test_case, k_values)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "query": test_case.query,
                    "status": "error",
                    "error": str(e),
                })
        
        # Aggregate metrics
        successful = [r for r in results if r.get("status") == "success"]
        
        if not successful:
            return {"results": results, "aggregate": None}
        
        aggregate = {}
        for k in k_values:
            aggregate[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in successful) / len(successful)
            aggregate[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in successful) / len(successful)
            aggregate[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in successful) / len(successful)
        
        aggregate["mrr"] = sum(r["mrr"] for r in successful) / len(successful)
        aggregate["total_queries"] = len(test_cases)
        aggregate["successful_queries"] = len(successful)
        
        return {
            "results": results,
            "aggregate": aggregate,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def load_test_cases(file_path: str) -> List[TestCase]:
    """Load test cases from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return [
        TestCase(
            query=item["query"],
            relevant_docs=item["relevant_docs"],
            expected_answer=item.get("expected_answer"),
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--test-file", required=True, help="Path to test cases JSON")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--k-values", default="1,3,5,10", help="K values for metrics")
    
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Load test cases
    print(f"Loading test cases from {args.test_file}...")
    test_cases = load_test_cases(args.test_file)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Run evaluation
    print(f"Evaluating against {args.api_url}...")
    evaluator = RetrievalEvaluator(args.api_url)
    results = evaluator.evaluate_batch(test_cases, k_values)
    
    # Print aggregate metrics
    if results["aggregate"]:
        print("\n" + "=" * 50)
        print("AGGREGATE METRICS")
        print("=" * 50)
        agg = results["aggregate"]
        print(f"Queries: {agg['successful_queries']}/{agg['total_queries']} successful")
        print(f"MRR: {agg['mrr']:.4f}")
        for k in k_values:
            print(f"Recall@{k}: {agg[f'recall@{k}']:.4f}")
            print(f"Precision@{k}: {agg[f'precision@{k}']:.4f}")
            print(f"NDCG@{k}: {agg[f'ndcg@{k}']:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

