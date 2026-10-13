"""
Evaluation Package

Provides comprehensive RAG evaluation, metrics calculation,
and performance comparison for AI systems.
"""

from .metrics_dashboard import (
    ChartType,
    MetricsDashboard,
)
from .rag_evaluator import (
    BaselineComparison,
    EvaluationMetrics,
    EvaluationResult,
    RAGEvaluator,
)

__all__ = [
    "BaselineComparison",
    "ChartType",
    "EvaluationMetrics",
    "EvaluationResult",
    "MetricsDashboard",
    "RAGEvaluator",
]
