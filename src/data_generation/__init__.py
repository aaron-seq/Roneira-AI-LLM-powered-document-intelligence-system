"""
Data Generation Package

Provides synthetic data generation for training, testing,
and validation of AI/ML systems.
"""

from .synthetic_data_generator import (
    SyntheticDataGenerator,
    QAPair,
    GenerationConfig,
)
from .dataset_validator import (
    DatasetValidator,
    DatasetSplit,
    ValidationReport,
)

__all__ = [
    "SyntheticDataGenerator",
    "QAPair",
    "GenerationConfig",
    "DatasetValidator",
    "DatasetSplit",
    "ValidationReport",
]
