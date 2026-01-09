"""
Tests for Evaluation Module

Unit tests for Task 6: Accuracy Evaluation
"""

import pytest
import json
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    BaselineComparison,
    MetricsDashboard,
    ChartType,
)
from src.evaluation.rag_evaluator import EvaluationSample


class TestRAGEvaluator:
    """Tests for RAGEvaluator class."""

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = RAGEvaluator()

        assert evaluator.cost_per_1k_tokens == 0.002
        assert evaluator.llm_judge is None

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        evaluator = RAGEvaluator()

        samples = [
            {
                "query": "What is machine learning?",
                "expected_answer": "Machine learning is a subset of AI.",
                "actual_answer": "Machine learning is a type of AI technology.",
                "retrieved_contexts": ["ML is AI technology.", "AI includes ML."],
                "ground_truth_contexts": ["ML is AI technology."],
                "relevance_scores": [0.9, 0.7],
                "latency": 1.2,
            },
            {
                "query": "What is deep learning?",
                "expected_answer": "Deep learning uses neural networks.",
                "actual_answer": "Deep learning is based on neural nets.",
                "retrieved_contexts": ["DL uses neural networks."],
                "ground_truth_contexts": ["DL uses neural networks."],
                "relevance_scores": [0.95],
                "latency": 1.5,
            },
        ]

        result = evaluator.evaluate(samples, dataset_name="Test")

        assert result.num_samples == 2
        assert 0 <= result.overall_score <= 1
        assert result.grade in ["A", "B", "C", "D", "F"]

    def test_evaluate_retrieval(self):
        """Test retrieval metrics calculation."""
        evaluator = RAGEvaluator()

        samples = [
            EvaluationSample(
                query="test query",
                retrieved_contexts=["relevant", "other", "more"],
                ground_truth_contexts=["relevant"],
                relevance_scores=[1.0, 0.5, 0.3],
            ),
        ]

        metrics = evaluator.evaluate_retrieval(samples)

        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "mrr" in metrics
        assert "ndcg" in metrics

    def test_evaluate_generation(self):
        """Test generation metrics calculation."""
        evaluator = RAGEvaluator()

        samples = [
            EvaluationSample(
                query="What is AI?",
                expected_answer="Artificial Intelligence is the simulation of human intelligence.",
                actual_answer="AI is the simulation of human intelligence by machines.",
                retrieved_contexts=["AI simulates human intelligence."],
            ),
        ]

        metrics = evaluator.evaluate_generation(samples)

        assert "bleu" in metrics
        assert "rouge_l" in metrics
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics

    def test_calculate_bleu(self):
        """Test BLEU score calculation."""
        evaluator = RAGEvaluator()

        reference = "the cat sat on the mat"
        hypothesis = "the cat is on the mat"

        bleu = evaluator._calculate_bleu(reference, hypothesis)

        assert 0 <= bleu <= 1

    def test_calculate_rouge_l(self):
        """Test ROUGE-L score calculation."""
        evaluator = RAGEvaluator()

        reference = "the quick brown fox jumps over the lazy dog"
        hypothesis = "the quick brown fox leaps over the lazy dog"

        rouge = evaluator._calculate_rouge_l(reference, hypothesis)

        assert 0 <= rouge <= 1
        assert rouge > 0.5  # High overlap expected

    def test_calculate_faithfulness(self):
        """Test faithfulness calculation."""
        evaluator = RAGEvaluator()

        answer = "Machine learning enables systems to learn from data."
        contexts = ["ML systems learn from data automatically."]

        faithfulness = evaluator._calculate_faithfulness(answer, contexts)

        assert 0 <= faithfulness <= 1

    def test_evaluate_performance(self):
        """Test performance metrics."""
        evaluator = RAGEvaluator()

        samples = [
            EvaluationSample(query="q1", latency=1.0),
            EvaluationSample(query="q2", latency=1.5),
            EvaluationSample(query="q3", latency=2.0),
            EvaluationSample(query="q4", latency=2.5),
            EvaluationSample(query="q5", latency=3.0),
        ]

        metrics = evaluator.evaluate_performance(samples)

        assert metrics["latency_p50"] <= metrics["latency_p95"]
        assert metrics["latency_p95"] <= metrics["latency_p99"]

    def test_evaluate_cost(self):
        """Test cost metrics."""
        evaluator = RAGEvaluator(cost_per_1k_tokens=0.002)

        samples = [
            EvaluationSample(
                query="Short query",
                actual_answer="Short answer",
                retrieved_contexts=["Some context"],
            ),
        ]

        metrics = evaluator.evaluate_cost(samples)

        assert metrics["tokens_per_query"] > 0
        assert metrics["cost_per_query"] > 0

    def test_compare_systems(self):
        """Test system comparison."""
        evaluator = RAGEvaluator()

        baseline_result = EvaluationResult(
            evaluation_id="baseline",
            metrics=EvaluationMetrics(mrr=0.65, ndcg=0.72, faithfulness=0.68),
        )

        enhanced_result = EvaluationResult(
            evaluation_id="enhanced",
            metrics=EvaluationMetrics(mrr=0.82, ndcg=0.85, faithfulness=0.89),
        )

        comparison = evaluator.compare_systems(baseline_result, enhanced_result)

        assert len(comparison.significant_improvements) > 0
        assert comparison.improvement_pct["MRR"] > 0

    def test_evaluate_with_llm_judge_heuristic(self):
        """Test LLM judge with heuristic fallback."""
        evaluator = RAGEvaluator(llm_judge=None)  # No LLM, use heuristics

        samples = [
            EvaluationSample(
                query="What is ML?",
                expected_answer="Machine learning is a subset of AI.",
                actual_answer="ML is an AI technology for learning.",
                retrieved_contexts=["ML is AI."],
            ),
        ]

        scores = evaluator.evaluate_with_llm_judge(samples)

        assert "relevance" in scores
        assert "accuracy" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_export_report(self, tmp_path):
        """Test exporting evaluation report."""
        evaluator = RAGEvaluator()

        result = EvaluationResult(
            evaluation_id="test_001",
            dataset_name="Test",
            num_samples=10,
            metrics=EvaluationMetrics(mrr=0.8, ndcg=0.85),
            overall_score=0.82,
            grade="B",
        )

        output_path = str(tmp_path / "report.json")
        evaluator.export_report(result, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["grade"] == "B"

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        evaluator = RAGEvaluator()

        comparison = BaselineComparison(
            baseline_metrics=EvaluationMetrics(mrr=0.65, faithfulness=0.68),
            enhanced_metrics=EvaluationMetrics(mrr=0.82, faithfulness=0.89),
            improvement_pct={"MRR": 26.2, "Faithfulness": 30.9},
        )

        table = evaluator.generate_comparison_table(comparison)

        assert "MRR" in table
        assert "Baseline" in table
        assert "Enhanced" in table

    def test_score_to_grade(self):
        """Test score to grade conversion."""
        evaluator = RAGEvaluator()

        assert evaluator._score_to_grade(0.95) == "A"
        assert evaluator._score_to_grade(0.85) == "B"
        assert evaluator._score_to_grade(0.75) == "C"
        assert evaluator._score_to_grade(0.65) == "D"
        assert evaluator._score_to_grade(0.55) == "F"


class TestMetricsDashboard:
    """Tests for MetricsDashboard class."""

    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        dashboard = MetricsDashboard("Test Dashboard")

        assert dashboard.title == "Test Dashboard"
        assert len(dashboard.charts) == 0
        assert len(dashboard.tables) == 0

    def test_add_comparison_chart(self):
        """Test adding comparison chart."""
        dashboard = MetricsDashboard("Test")

        dashboard.add_comparison_chart(
            "Metrics Comparison",
            baseline={"MRR": 0.65, "NDCG": 0.72},
            enhanced={"MRR": 0.82, "NDCG": 0.85},
        )

        assert len(dashboard.charts) == 1
        assert dashboard.charts[0].title == "Metrics Comparison"

    def test_add_metrics_table(self):
        """Test adding metrics table."""
        dashboard = MetricsDashboard("Test")

        dashboard.add_metrics_table(
            "Performance Metrics",
            {"Latency P50": 1.2, "Latency P95": 2.8},
        )

        assert len(dashboard.tables) == 1

    def test_add_distribution_chart(self):
        """Test adding distribution chart."""
        dashboard = MetricsDashboard("Test")

        dashboard.add_distribution_chart(
            "Grade Distribution",
            values=[45, 35, 15, 5],
            labels=["A", "B", "C", "D/F"],
            chart_type=ChartType.PIE,
        )

        assert len(dashboard.charts) == 1
        assert dashboard.charts[0].chart_type == ChartType.PIE

    def test_add_trend_chart(self):
        """Test adding trend chart."""
        dashboard = MetricsDashboard("Test")

        dashboard.add_trend_chart(
            "Performance Over Time",
            data_series={"MRR": [0.6, 0.7, 0.8], "NDCG": [0.65, 0.75, 0.85]},
            x_labels=["Week 1", "Week 2", "Week 3"],
        )

        assert len(dashboard.charts) == 1
        assert dashboard.charts[0].chart_type == ChartType.LINE

    def test_export_to_html(self, tmp_path):
        """Test exporting dashboard to HTML."""
        dashboard = MetricsDashboard("Evaluation Dashboard")

        dashboard.add_comparison_chart(
            "Baseline vs Enhanced",
            baseline={"MRR": 0.65},
            enhanced={"MRR": 0.82},
        )

        dashboard.add_metrics_table(
            "Summary",
            {"Overall Score": 0.82, "Grade": "B"},
        )

        output_path = str(tmp_path / "dashboard.html")
        dashboard.export_to_html(output_path)

        with open(output_path) as f:
            html = f.read()

        assert "<!DOCTYPE html>" in html
        assert "Evaluation Dashboard" in html
        assert "Chart.js" in html or "chart-container" in html

    def test_generate_markdown_report(self):
        """Test generating Markdown report."""
        dashboard = MetricsDashboard("Test Dashboard")

        dashboard.add_metrics_table(
            "Metrics",
            {"MRR": 0.82, "NDCG": 0.85},
        )

        dashboard.add_comparison_chart(
            "Comparison",
            baseline={"MRR": 0.65},
            enhanced={"MRR": 0.82},
        )

        markdown = dashboard.generate_markdown_report()

        assert "# Test Dashboard" in markdown
        assert "MRR" in markdown

    def test_value_to_grade(self):
        """Test value to grade conversion."""
        dashboard = MetricsDashboard("Test")

        assert dashboard._value_to_grade(0.95) == "A"
        assert dashboard._value_to_grade(0.85) == "B"
        assert dashboard._value_to_grade(0.75) == "C"
        assert dashboard._value_to_grade(0.65) == "D"
        assert dashboard._value_to_grade(0.55) == "F"


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = EvaluationMetrics(
            mrr=0.82,
            ndcg=0.85,
            faithfulness=0.89,
            answer_relevancy=0.84,
            latency_p50=1.2,
        )

        assert metrics.mrr == 0.82
        assert metrics.latency_p50 == 1.2

    def test_overall_score(self):
        """Test overall score calculation."""
        metrics = EvaluationMetrics(
            mrr=0.8,
            ndcg=0.8,
            faithfulness=0.8,
            answer_relevancy=0.8,
            context_precision=0.8,
            latency_p50=1.0,
        )

        score = metrics.overall_score

        assert 0 <= score <= 1

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = EvaluationMetrics(mrr=0.82, ndcg=0.85)

        d = metrics.to_dict()

        assert d["mrr"] == 0.82
        assert d["ndcg"] == 0.85


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self):
        """Test creating result."""
        result = EvaluationResult(
            evaluation_id="eval_001",
            dataset_name="Test Set",
            num_samples=100,
            overall_score=0.82,
            grade="B",
        )

        assert result.evaluation_id == "eval_001"
        assert result.grade == "B"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EvaluationResult(
            evaluation_id="eval_001",
            dataset_name="Test",
            num_samples=50,
            overall_score=0.85,
            grade="B",
        )

        d = result.to_dict()

        assert d["evaluation_id"] == "eval_001"
        assert d["overall_score"] == 0.85
