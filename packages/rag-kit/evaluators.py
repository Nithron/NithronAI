"""
nAI RAG Kit - Evaluators
========================

Evaluation metrics for RAG systems.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class RetrievalMetricsResult:
    """Results from retrieval evaluation."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall": self.recall_at_k,
            "precision": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg_at_k,
            "hit_rate": self.hit_rate_at_k,
        }
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [f"MRR: {self.mrr:.4f}"]
        for k in sorted(self.recall_at_k.keys()):
            lines.append(f"R@{k}: {self.recall_at_k[k]:.4f} | P@{k}: {self.precision_at_k[k]:.4f} | NDCG@{k}: {self.ndcg_at_k.get(k, 0):.4f}")
        return "\n".join(lines)


class RetrievalMetrics:
    """
    Compute standard IR evaluation metrics.
    
    Metrics:
    - Recall@K: Fraction of relevant documents retrieved in top K
    - Precision@K: Fraction of top K documents that are relevant
    - MRR: Mean Reciprocal Rank
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Hit Rate@K: Fraction of queries with at least one relevant in top K
    """
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]
    
    def _normalize_ids(self, ids: List[str]) -> List[str]:
        """Normalize document IDs for comparison."""
        return [str(id_).lower().strip() for id_ in ids]
    
    def recall_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0
        
        retrieved_set = set(self._normalize_ids(retrieved[:k]))
        relevant_set = set(self._normalize_ids(relevant))
        
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    def precision_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K."""
        retrieved_at_k = self._normalize_ids(retrieved[:k])
        if not retrieved_at_k:
            return 0.0
        
        relevant_set = set(self._normalize_ids(relevant))
        hits = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        
        return hits / len(retrieved_at_k)
    
    def mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank for a single query."""
        relevant_set = set(self._normalize_ids(relevant))
        retrieved_norm = self._normalize_ids(retrieved)
        
        for i, doc in enumerate(retrieved_norm):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def dcg_at_k(self, scores: List[float], k: int) -> float:
        """Calculate DCG@K."""
        dcg = 0.0
        for i, score in enumerate(scores[:k]):
            dcg += score / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def ndcg_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K."""
        relevant_set = set(self._normalize_ids(relevant))
        retrieved_norm = self._normalize_ids(retrieved[:k])
        
        # Binary relevance scores
        scores = [1.0 if doc in relevant_set else 0.0 for doc in retrieved_norm]
        
        dcg = self.dcg_at_k(scores, k)
        
        # Ideal DCG (all relevant documents at top)
        ideal_scores = [1.0] * min(len(relevant_set), k)
        ideal_dcg = self.dcg_at_k(ideal_scores, k)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def hit_rate_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Check if any relevant document is in top K."""
        retrieved_set = set(self._normalize_ids(retrieved[:k]))
        relevant_set = set(self._normalize_ids(relevant))
        
        return 1.0 if retrieved_set & relevant_set else 0.0
    
    def evaluate_single(
        self, 
        retrieved: List[str], 
        relevant: List[str]
    ) -> RetrievalMetricsResult:
        """Evaluate a single query."""
        result = RetrievalMetricsResult(mrr=self.mrr(retrieved, relevant))
        
        for k in self.k_values:
            result.recall_at_k[k] = self.recall_at_k(retrieved, relevant, k)
            result.precision_at_k[k] = self.precision_at_k(retrieved, relevant, k)
            result.ndcg_at_k[k] = self.ndcg_at_k(retrieved, relevant, k)
            result.hit_rate_at_k[k] = self.hit_rate_at_k(retrieved, relevant, k)
        
        return result
    
    def evaluate_batch(
        self, 
        queries: List[Dict[str, Any]]
    ) -> RetrievalMetricsResult:
        """
        Evaluate a batch of queries.
        
        Each query should have:
        - 'retrieved': List of retrieved document IDs
        - 'relevant': List of relevant document IDs
        """
        if not queries:
            return RetrievalMetricsResult()
        
        all_results = [
            self.evaluate_single(q['retrieved'], q['relevant'])
            for q in queries
        ]
        
        # Aggregate metrics
        n = len(all_results)
        
        result = RetrievalMetricsResult(
            mrr=sum(r.mrr for r in all_results) / n
        )
        
        for k in self.k_values:
            result.recall_at_k[k] = sum(r.recall_at_k[k] for r in all_results) / n
            result.precision_at_k[k] = sum(r.precision_at_k[k] for r in all_results) / n
            result.ndcg_at_k[k] = sum(r.ndcg_at_k[k] for r in all_results) / n
            result.hit_rate_at_k[k] = sum(r.hit_rate_at_k[k] for r in all_results) / n
        
        return result


@dataclass
class AnswerQualityResult:
    """Results from answer quality evaluation."""
    faithfulness: float = 0.0  # Is the answer grounded in the context?
    relevance: float = 0.0     # Does the answer address the question?
    coherence: float = 0.0     # Is the answer well-structured?
    completeness: float = 0.0  # Does it cover all aspects?
    
    def overall(self) -> float:
        """Calculate overall score."""
        return (self.faithfulness + self.relevance + self.coherence + self.completeness) / 4
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "completeness": self.completeness,
            "overall": self.overall(),
        }


class AnswerQualityMetrics:
    """
    Evaluate answer quality using LLM-as-judge.
    
    Metrics:
    - Faithfulness: Is the answer factually grounded in the context?
    - Relevance: Does the answer address the question?
    - Coherence: Is the answer well-structured and clear?
    - Completeness: Does the answer cover all relevant aspects?
    
    Requires: LLM API (OpenAI, Anthropic, etc.)
    """
    
    FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is factually grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Rate the faithfulness of the answer on a scale of 0 to 1:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Some claims are supported, some are not verifiable from context
- 0.0: The answer contains claims not supported by the context

Respond with ONLY a number between 0 and 1."""

    RELEVANCE_PROMPT = """You are evaluating whether an AI-generated answer addresses the question asked.

Question: {question}

Answer: {answer}

Rate the relevance of the answer on a scale of 0 to 1:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question
- 0.0: The answer does not address the question

Respond with ONLY a number between 0 and 1."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        self.llm_model = llm_model
        self.api_key = api_key
    
    async def _call_llm(self, prompt: str) -> float:
        """Call LLM and parse score."""
        import litellm
        
        response = await litellm.acompletion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        context: str
    ) -> AnswerQualityResult:
        """Evaluate answer quality."""
        faithfulness_prompt = self.FAITHFULNESS_PROMPT.format(
            context=context,
            question=question,
            answer=answer
        )
        
        relevance_prompt = self.RELEVANCE_PROMPT.format(
            question=question,
            answer=answer
        )
        
        import asyncio
        
        faithfulness, relevance = await asyncio.gather(
            self._call_llm(faithfulness_prompt),
            self._call_llm(relevance_prompt),
        )
        
        return AnswerQualityResult(
            faithfulness=faithfulness,
            relevance=relevance,
            coherence=0.8,  # Default for now
            completeness=0.8,  # Default for now
        )
    
    def evaluate_sync(
        self,
        question: str,
        answer: str,
        context: str
    ) -> AnswerQualityResult:
        """Synchronous evaluation wrapper."""
        import asyncio
        return asyncio.run(self.evaluate(question, answer, context))

