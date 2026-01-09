"""
Evaluation Package

Provides comprehensive RAG evaluation, metrics calculation,
and performance comparison for AI systems.
"""

from .rag_evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    BaselineComparison,
)
from .metrics_dashboard import (
    MetricsDashboard,
    ChartType,
)

__all__ = [
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "BaselineComparison",
    "MetricsDashboard",
    "ChartType",
]
