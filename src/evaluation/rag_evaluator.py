"""
RAG Pipeline Evaluation Framework

Comprehensive evaluation for retrieval quality, generation accuracy,
and end-to-end system performance.

This module implements Task 6 of the AI Developer Roadmap:
- Accuracy evaluation methodology
- Current vs expected comparison
- Multi-dimensional metrics
- LLM-as-judge evaluation
"""

import json
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import Counter
import re
import math

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    COST = "cost"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Retrieval metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain

    # Generation metrics
    bleu: float = 0.0
    rouge_l: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0

    # Performance metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0

    # Cost metrics
    tokens_per_query: float = 0.0
    cost_per_query: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "retrieval": 0.25,
            "generation": 0.35,
            "end_to_end": 0.25,
            "performance": 0.15,
        }

        retrieval_score = (self.mrr + self.ndcg) / 2
        generation_score = (self.faithfulness + self.answer_relevancy) / 2
        e2e_score = self.context_precision
        perf_score = 1.0 - min(1.0, self.latency_p50 / 5.0)  # Normalize latency

        return (
            retrieval_score * weights["retrieval"]
            + generation_score * weights["generation"]
            + e2e_score * weights["end_to_end"]
            + perf_score * weights["performance"]
        )


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    query: str
    expected_answer: Optional[str] = None
    actual_answer: Optional[str] = None
    retrieved_contexts: List[str] = field(default_factory=list)
    ground_truth_contexts: List[str] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    evaluation_id: str
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str = ""
    num_samples: int = 0

    metrics: Optional[EvaluationMetrics] = None
    sample_results: List[Dict[str, Any]] = field(default_factory=list)

    # Summary
    overall_score: float = 0.0
    grade: str = "F"

    # Issues and insights
    issues: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "evaluated_at": self.evaluated_at,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "issues": self.issues,
            "insights": self.insights,
        }


@dataclass
class BaselineComparison:
    """Comparison between baseline and enhanced system."""

    baseline_metrics: EvaluationMetrics
    enhanced_metrics: EvaluationMetrics

    improvement_pct: Dict[str, float] = field(default_factory=dict)
    significant_improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_metrics.to_dict(),
            "enhanced": self.enhanced_metrics.to_dict(),
            "improvement_pct": self.improvement_pct,
            "significant_improvements": self.significant_improvements,
            "regressions": self.regressions,
        }


class RAGEvaluator:
    """
    Multi-dimensional RAG evaluation framework.

    This class implements the core functionality for Task 6:
    Accuracy Evaluation of the AI Developer Roadmap.

    Metrics:
    - Retrieval: Precision@K, Recall@K, MRR, NDCG
    - Generation: ROUGE, BLEU, Faithfulness
    - End-to-end: Answer relevancy, context precision
    - Performance: Latency, throughput, cost

    Example:
        evaluator = RAGEvaluator()

        # Prepare test set
        test_samples = [
            {"query": "What is ML?", "expected_answer": "Machine Learning is..."},
            ...
        ]

        # Evaluate
        result = evaluator.evaluate(test_samples)
        print(f"Overall Score: {result.overall_score}")

        # Compare baseline vs enhanced
        comparison = evaluator.compare_systems(baseline_result, enhanced_result)
    """

    def __init__(
        self,
        llm_judge: Optional[Callable] = None,
        cost_per_1k_tokens: float = 0.002,
    ):
        """
        Initialize the RAG evaluator.

        Args:
            llm_judge: Optional LLM function for judge-based evaluation
            cost_per_1k_tokens: Cost per 1000 tokens for cost estimation
        """
        self.llm_judge = llm_judge
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self._evaluation_id_counter = 0

        logger.info("RAGEvaluator initialized")

    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation ID."""
        self._evaluation_id_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"eval_{timestamp}_{self._evaluation_id_counter:04d}"

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        dataset_name: str = "test_set",
    ) -> EvaluationResult:
        """
        Evaluate RAG system on test samples.

        Args:
            test_samples: List of test samples with queries and expected answers
            dataset_name: Name of the test dataset

        Returns:
            EvaluationResult with comprehensive metrics
        """
        eval_id = self._generate_evaluation_id()

        # Convert to EvaluationSample objects
        samples = [self._parse_sample(s) for s in test_samples]

        # Calculate metrics
        metrics = EvaluationMetrics()

        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(samples)
        metrics.precision_at_k = retrieval_metrics.get("precision_at_k", {})
        metrics.recall_at_k = retrieval_metrics.get("recall_at_k", {})
        metrics.mrr = retrieval_metrics.get("mrr", 0.0)
        metrics.ndcg = retrieval_metrics.get("ndcg", 0.0)

        # Generation metrics
        generation_metrics = self.evaluate_generation(samples)
        metrics.bleu = generation_metrics.get("bleu", 0.0)
        metrics.rouge_l = generation_metrics.get("rouge_l", 0.0)
        metrics.faithfulness = generation_metrics.get("faithfulness", 0.0)
        metrics.answer_relevancy = generation_metrics.get("answer_relevancy", 0.0)

        # End-to-end metrics
        e2e_metrics = self.evaluate_end_to_end(samples)
        metrics.context_precision = e2e_metrics.get("context_precision", 0.0)

        # Performance metrics
        perf_metrics = self.evaluate_performance(samples)
        metrics.latency_p50 = perf_metrics.get("latency_p50", 0.0)
        metrics.latency_p95 = perf_metrics.get("latency_p95", 0.0)
        metrics.latency_p99 = perf_metrics.get("latency_p99", 0.0)
        metrics.throughput = perf_metrics.get("throughput", 0.0)

        # Cost metrics
        cost_metrics = self.evaluate_cost(samples)
        metrics.tokens_per_query = cost_metrics.get("tokens_per_query", 0.0)
        metrics.cost_per_query = cost_metrics.get("cost_per_query", 0.0)

        # Create result
        result = EvaluationResult(
            evaluation_id=eval_id,
            dataset_name=dataset_name,
            num_samples=len(samples),
            metrics=metrics,
            overall_score=metrics.overall_score,
            grade=self._score_to_grade(metrics.overall_score),
        )

        # Generate insights
        result.issues = self._identify_issues(metrics)
        result.insights = self._generate_insights(metrics, samples)

        logger.info(f"Evaluation complete: {eval_id}, score={result.overall_score:.3f}")
        return result

    def _parse_sample(self, sample: Dict[str, Any]) -> EvaluationSample:
        """Parse a raw sample dict into EvaluationSample."""
        return EvaluationSample(
            query=sample.get("query", ""),
            expected_answer=sample.get("expected_answer"),
            actual_answer=sample.get("actual_answer"),
            retrieved_contexts=sample.get("retrieved_contexts", []),
            ground_truth_contexts=sample.get("ground_truth_contexts", []),
            relevance_scores=sample.get("relevance_scores", []),
            latency=sample.get("latency", 0.0),
            metadata=sample.get("metadata", {}),
        )

    def evaluate_retrieval(
        self,
        samples: List[EvaluationSample],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.

        Args:
            samples: Evaluation samples
            k_values: K values for precision@k and recall@k

        Returns:
            Dictionary of retrieval metrics
        """
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        reciprocal_ranks = []
        ndcg_scores = []

        for sample in samples:
            relevant_set = set(sample.ground_truth_contexts)
            retrieved = sample.retrieved_contexts

            if not relevant_set:
                continue

            # Calculate metrics
            for k in k_values:
                top_k = retrieved[:k]
                relevant_in_k = len(set(top_k) & relevant_set)

                precision_at_k[k].append(relevant_in_k / k if k > 0 else 0)
                recall_at_k[k].append(
                    relevant_in_k / len(relevant_set) if relevant_set else 0
                )

            # MRR
            rr = 0.0
            for i, doc in enumerate(retrieved):
                if doc in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)

            # NDCG
            ndcg = self._calculate_ndcg(
                retrieved, relevant_set, sample.relevance_scores
            )
            ndcg_scores.append(ndcg)

        return {
            "precision_at_k": {
                k: statistics.mean(v) if v else 0.0 for k, v in precision_at_k.items()
            },
            "recall_at_k": {
                k: statistics.mean(v) if v else 0.0 for k, v in recall_at_k.items()
            },
            "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "ndcg": statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
        }

    def _calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: set,
        scores: List[float],
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not retrieved or not relevant:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved):
            rel = 1.0 if doc in relevant else 0.0
            if scores and i < len(scores):
                rel = scores[i]
            dcg += rel / math.log2(i + 2)  # i+2 because log_2(1) = 0

        # Ideal DCG
        ideal_rels = sorted([1.0] * len(relevant), reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_generation(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality.

        Args:
            samples: Evaluation samples with expected and actual answers

        Returns:
            Dictionary of generation metrics
        """
        bleu_scores = []
        rouge_scores = []
        faithfulness_scores = []
        relevancy_scores = []

        for sample in samples:
            if not sample.expected_answer or not sample.actual_answer:
                continue

            # BLEU score (simplified)
            bleu = self._calculate_bleu(sample.expected_answer, sample.actual_answer)
            bleu_scores.append(bleu)

            # ROUGE-L score (simplified)
            rouge = self._calculate_rouge_l(
                sample.expected_answer, sample.actual_answer
            )
            rouge_scores.append(rouge)

            # Faithfulness (context-based)
            faithfulness = self._calculate_faithfulness(
                sample.actual_answer,
                sample.retrieved_contexts,
            )
            faithfulness_scores.append(faithfulness)

            # Answer relevancy
            relevancy = self._calculate_answer_relevancy(
                sample.query,
                sample.actual_answer,
            )
            relevancy_scores.append(relevancy)

        return {
            "bleu": statistics.mean(bleu_scores) if bleu_scores else 0.0,
            "rouge_l": statistics.mean(rouge_scores) if rouge_scores else 0.0,
            "faithfulness": statistics.mean(faithfulness_scores)
            if faithfulness_scores
            else 0.0,
            "answer_relevancy": statistics.mean(relevancy_scores)
            if relevancy_scores
            else 0.0,
        }

    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate simplified BLEU score."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens:
            return 0.0

        # Unigram precision
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)

        overlap = sum((hyp_counts & ref_counts).values())
        precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0

        # Brevity penalty
        bp = (
            1.0
            if len(hyp_tokens) >= len(ref_tokens)
            else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        )

        return bp * precision

    def _calculate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L score (longest common subsequence)."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return 0.0

        # LCS calculation
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]

        precision = lcs_length / n if n > 0 else 0.0
        recall = lcs_length / m if m > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)  # F1

    def _calculate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Calculate faithfulness (how well answer is grounded in context)."""
        if not contexts or not answer:
            return 0.0

        combined_context = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(combined_context.split())

        # Calculate overlap
        overlap = len(answer_words & context_words)

        # Normalize by answer length
        return overlap / len(answer_words) if answer_words else 0.0

    def _calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Calculate answer relevancy to query."""
        if not query or not answer:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
        }
        query_words -= stop_words

        if not query_words:
            return 0.5

        overlap = len(query_words & answer_words)
        return min(1.0, overlap / len(query_words))

    def evaluate_end_to_end(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Evaluate end-to-end quality."""
        context_precision_scores = []

        for sample in samples:
            if not sample.ground_truth_contexts:
                continue

            relevant = set(sample.ground_truth_contexts)
            retrieved = sample.retrieved_contexts

            # Context precision
            relevant_in_retrieved = len(set(retrieved) & relevant)
            precision = relevant_in_retrieved / len(retrieved) if retrieved else 0.0
            context_precision_scores.append(precision)

        return {
            "context_precision": statistics.mean(context_precision_scores)
            if context_precision_scores
            else 0.0,
        }

    def evaluate_performance(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Evaluate performance metrics."""
        latencies = [s.latency for s in samples if s.latency > 0]

        if not latencies:
            return {
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
                "throughput": 0.0,
            }

        latencies.sort()
        n = len(latencies)

        return {
            "latency_p50": latencies[int(n * 0.50)],
            "latency_p95": latencies[min(int(n * 0.95), n - 1)],
            "latency_p99": latencies[min(int(n * 0.99), n - 1)],
            "throughput": 1.0 / statistics.mean(latencies) if latencies else 0.0,
        }

    def evaluate_cost(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Evaluate cost metrics."""
        total_tokens = 0

        for sample in samples:
            # Estimate tokens (rough approximation)
            query_tokens = len(sample.query.split()) * 1.3
            answer_tokens = len((sample.actual_answer or "").split()) * 1.3
            context_tokens = sum(
                len(c.split()) * 1.3 for c in sample.retrieved_contexts
            )
            total_tokens += query_tokens + answer_tokens + context_tokens

        avg_tokens = total_tokens / len(samples) if samples else 0
        cost = (avg_tokens / 1000) * self.cost_per_1k_tokens

        return {
            "tokens_per_query": avg_tokens,
            "cost_per_query": cost,
        }

    def evaluate_with_llm_judge(
        self,
        samples: List[EvaluationSample],
        criteria: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using LLM as judge.

        Args:
            samples: Samples to evaluate
            criteria: Evaluation criteria (defaults to standard RAG criteria)

        Returns:
            Dictionary of scores by criteria
        """
        criteria = criteria or [
            "relevance",
            "accuracy",
            "completeness",
            "coherence",
        ]

        scores = {c: [] for c in criteria}

        if self.llm_judge:
            for sample in samples:
                for criterion in criteria:
                    prompt = self._build_judge_prompt(sample, criterion)
                    try:
                        score = self.llm_judge(prompt)
                        scores[criterion].append(float(score))
                    except Exception as e:
                        logger.warning(f"LLM judge failed: {e}")
        else:
            # Fallback to heuristic scoring
            for sample in samples:
                for criterion in criteria:
                    score = self._heuristic_score(sample, criterion)
                    scores[criterion].append(score)

        return {c: statistics.mean(v) if v else 0.0 for c, v in scores.items()}

    def _build_judge_prompt(self, sample: EvaluationSample, criterion: str) -> str:
        """Build prompt for LLM judge."""
        return f"""
Evaluate the following answer based on {criterion}.

Query: {sample.query}
Context: {" ".join(sample.retrieved_contexts[:2])}
Answer: {sample.actual_answer}
Expected: {sample.expected_answer or "N/A"}

Rate the {criterion} on a scale of 0.0 to 1.0.
Return only the numeric score.
"""

    def _heuristic_score(self, sample: EvaluationSample, criterion: str) -> float:
        """Calculate heuristic score for a criterion."""
        if criterion == "relevance":
            return self._calculate_answer_relevancy(
                sample.query, sample.actual_answer or ""
            )
        elif criterion == "accuracy":
            if sample.expected_answer and sample.actual_answer:
                return self._calculate_rouge_l(
                    sample.expected_answer, sample.actual_answer
                )
            return 0.5
        elif criterion == "completeness":
            return min(1.0, len((sample.actual_answer or "").split()) / 50)
        elif criterion == "coherence":
            return 0.8  # Assume reasonable coherence
        return 0.5

    def compare_systems(
        self,
        baseline: EvaluationResult,
        enhanced: EvaluationResult,
    ) -> BaselineComparison:
        """
        Compare baseline and enhanced system results.

        Args:
            baseline: Baseline evaluation result
            enhanced: Enhanced system evaluation result

        Returns:
            BaselineComparison with improvements and regressions
        """
        comparison = BaselineComparison(
            baseline_metrics=baseline.metrics,
            enhanced_metrics=enhanced.metrics,
        )

        # Calculate improvements
        metrics_to_compare = [
            ("mrr", "MRR"),
            ("ndcg", "NDCG"),
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_precision", "Context Precision"),
        ]

        for metric_attr, metric_name in metrics_to_compare:
            baseline_val = getattr(baseline.metrics, metric_attr, 0)
            enhanced_val = getattr(enhanced.metrics, metric_attr, 0)

            if baseline_val > 0:
                improvement = ((enhanced_val - baseline_val) / baseline_val) * 100
                comparison.improvement_pct[metric_name] = improvement

                if improvement > 5:
                    comparison.significant_improvements.append(
                        f"{metric_name}: +{improvement:.1f}%"
                    )
                elif improvement < -5:
                    comparison.regressions.append(f"{metric_name}: {improvement:.1f}%")

        # Compare latency (lower is better)
        if baseline.metrics.latency_p50 > 0:
            latency_change = (
                (enhanced.metrics.latency_p50 - baseline.metrics.latency_p50)
                / baseline.metrics.latency_p50
            ) * 100
            comparison.improvement_pct[
                "Latency P50"
            ] = -latency_change  # Negative change is improvement

            if latency_change < -10:
                comparison.significant_improvements.append(
                    f"Latency P50: {-latency_change:.1f}% faster"
                )
            elif latency_change > 10:
                comparison.regressions.append(
                    f"Latency P50: {latency_change:.1f}% slower"
                )

        logger.info(
            f"Comparison: {len(comparison.significant_improvements)} improvements, "
            f"{len(comparison.regressions)} regressions"
        )
        return comparison

    def _identify_issues(self, metrics: EvaluationMetrics) -> List[str]:
        """Identify potential issues based on metrics."""
        issues = []

        if metrics.mrr < 0.5:
            issues.append("Low MRR indicates relevant documents not ranked highly")

        if metrics.faithfulness < 0.6:
            issues.append("Low faithfulness - answers may not be grounded in context")

        if metrics.latency_p95 > 3.0:
            issues.append("High P95 latency may impact user experience")

        if metrics.answer_relevancy < 0.5:
            issues.append("Answers may not be relevant to user queries")

        return issues

    def _generate_insights(
        self,
        metrics: EvaluationMetrics,
        samples: List[EvaluationSample],
    ) -> List[str]:
        """Generate insights from evaluation results."""
        insights = []

        if metrics.overall_score > 0.8:
            insights.append("Overall system performance is excellent")
        elif metrics.overall_score > 0.6:
            insights.append("System performs adequately with room for improvement")
        else:
            insights.append("System needs significant improvements")

        if metrics.precision_at_k.get(1, 0) > 0.8:
            insights.append("Top-1 retrieval is highly accurate")

        if metrics.faithfulness > metrics.answer_relevancy:
            insights.append(
                "Consider improving query understanding for better relevancy"
            )

        return insights

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def export_report(
        self,
        result: EvaluationResult,
        output_path: str,
    ) -> str:
        """Export evaluation report to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Exported evaluation report to {output_path}")
        return output_path

    def generate_comparison_table(
        self,
        comparison: BaselineComparison,
    ) -> str:
        """Generate Markdown comparison table."""
        lines = [
            "| Metric | Baseline | Enhanced | Change |",
            "|--------|----------|----------|--------|",
        ]

        metrics = [
            ("MRR", "mrr"),
            ("NDCG", "ndcg"),
            ("Faithfulness", "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Latency P50 (s)", "latency_p50"),
        ]

        for name, attr in metrics:
            baseline = getattr(comparison.baseline_metrics, attr, 0)
            enhanced = getattr(comparison.enhanced_metrics, attr, 0)
            change = comparison.improvement_pct.get(name, 0)

            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
            lines.append(f"| {name} | {baseline:.3f} | {enhanced:.3f} | {change_str} |")

        return "\n".join(lines)
