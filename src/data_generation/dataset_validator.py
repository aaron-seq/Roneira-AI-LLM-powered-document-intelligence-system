"""
Dataset Validation and Split Management

Handles train/test/validation splits with stratification,
quality validation, and comprehensive statistics reporting.

This module completes Task 2 of the AI Developer Roadmap by providing:
- Stratified train/test/validation splitting
- Data leakage detection
- Quality scoring and validation
- Comprehensive statistics reporting
"""

import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import hashlib

from .synthetic_data_generator import QAPair, Difficulty, QuestionType

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Container for train/validation/test splits."""

    train: List[QAPair] = field(default_factory=list)
    validation: List[QAPair] = field(default_factory=list)
    test: List[QAPair] = field(default_factory=list)

    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    stratify_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_size(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def get_split(self, split_name: str) -> List[QAPair]:
        """Get a specific split by name."""
        return getattr(self, split_name, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_size": len(self.train),
            "validation_size": len(self.validation),
            "test_size": len(self.test),
            "total_size": self.total_size,
            "split_ratios": self.split_ratios,
            "stratify_by": self.stratify_by,
            "created_at": self.created_at,
        }


@dataclass
class LeakageReport:
    """Report on data leakage between splits."""

    has_leakage: bool = False
    duplicate_questions: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_contexts: List[Dict[str, Any]] = field(default_factory=list)
    similar_pairs: List[Dict[str, Any]] = field(default_factory=list)
    leakage_score: float = 0.0  # 0 = no leakage, 1 = severe leakage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset split."""

    completeness: float = 0.0  # % of pairs with all required fields
    diversity_score: float = 0.0  # Vocabulary/topic diversity
    balance_score: float = 0.0  # Distribution balance across categories
    avg_question_length: float = 0.0
    avg_answer_length: float = 0.0
    unique_question_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        return (
            self.completeness * 0.3
            + self.diversity_score * 0.25
            + self.balance_score * 0.25
            + self.unique_question_ratio * 0.2
        )


@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""

    dataset_name: str = "unnamed"
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Split information
    split_info: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics per split
    train_quality: Optional[QualityMetrics] = None
    validation_quality: Optional[QualityMetrics] = None
    test_quality: Optional[QualityMetrics] = None

    # Leakage analysis
    leakage_report: Optional[LeakageReport] = None

    # Distribution statistics
    difficulty_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    type_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    category_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Overall assessment
    is_valid: bool = True
    quality_grade: str = "A"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "validated_at": self.validated_at,
            "split_info": self.split_info,
            "quality_metrics": {
                "train": self.train_quality.to_dict() if self.train_quality else None,
                "validation": self.validation_quality.to_dict()
                if self.validation_quality
                else None,
                "test": self.test_quality.to_dict() if self.test_quality else None,
            },
            "leakage_report": self.leakage_report.to_dict()
            if self.leakage_report
            else None,
            "distributions": {
                "difficulty": self.difficulty_distribution,
                "type": self.type_distribution,
                "category": self.category_distribution,
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
            "quality_grade": self.quality_grade,
        }


class DatasetValidator:
    """
    Dataset splitting and validation framework.

    This class implements the validation component of Task 2:
    Training & Test Data Generation of the AI Developer Roadmap.

    Features:
    - Stratified splitting by topic/complexity
    - Data leakage detection and prevention
    - Quality scoring with multiple metrics
    - Statistical distribution analysis
    - Comprehensive validation reporting

    Example:
        validator = DatasetValidator()
        split = validator.create_splits(qa_pairs, ratios=(0.8, 0.1, 0.1))
        report = validator.validate(split)
        validator.export_report(report, "validation_report.json")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the dataset validator.

        Args:
            seed: Random seed for reproducible splits
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        logger.info("DatasetValidator initialized")

    def create_splits(
        self,
        data: List[QAPair],
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        stratify_by: Optional[str] = "difficulty",
        shuffle: bool = True,
    ) -> DatasetSplit:
        """
        Create train/validation/test splits from data.

        Args:
            data: List of QAPair objects to split
            ratios: (train, validation, test) ratios
            stratify_by: Field to stratify by ('difficulty', 'category', 'question_type')
            shuffle: Whether to shuffle data before splitting

        Returns:
            DatasetSplit containing train, validation, and test sets
        """
        if abs(sum(ratios) - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")

        if not data:
            return DatasetSplit(split_ratios=ratios, stratify_by=stratify_by)

        if shuffle:
            data = data.copy()
            random.shuffle(data)

        if stratify_by:
            split = self._stratified_split(data, ratios, stratify_by)
        else:
            split = self._simple_split(data, ratios)

        split.split_ratios = ratios
        split.stratify_by = stratify_by

        logger.info(
            f"Created splits: train={len(split.train)}, "
            f"val={len(split.validation)}, test={len(split.test)}"
        )
        return split

    def _simple_split(
        self,
        data: List[QAPair],
        ratios: Tuple[float, float, float],
    ) -> DatasetSplit:
        """Simple random split without stratification."""
        n = len(data)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        return DatasetSplit(
            train=data[:train_end],
            validation=data[train_end:val_end],
            test=data[val_end:],
        )

    def _stratified_split(
        self,
        data: List[QAPair],
        ratios: Tuple[float, float, float],
        stratify_by: str,
    ) -> DatasetSplit:
        """Stratified split maintaining distribution of stratify_by field."""
        # Group by stratification key
        groups: Dict[str, List[QAPair]] = defaultdict(list)

        for pair in data:
            if stratify_by == "difficulty":
                key = pair.difficulty.value
            elif stratify_by == "category":
                key = pair.category
            elif stratify_by == "question_type":
                key = pair.question_type.value
            else:
                key = "default"
            groups[key].append(pair)

        # Split each group proportionally
        split = DatasetSplit()

        for key, group_data in groups.items():
            random.shuffle(group_data)
            n = len(group_data)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])

            split.train.extend(group_data[:train_end])
            split.validation.extend(group_data[train_end:val_end])
            split.test.extend(group_data[val_end:])

        # Shuffle final splits
        random.shuffle(split.train)
        random.shuffle(split.validation)
        random.shuffle(split.test)

        return split

    def check_leakage(self, dataset: DatasetSplit) -> LeakageReport:
        """
        Check for data leakage between splits.

        Args:
            dataset: DatasetSplit to check

        Returns:
            LeakageReport with leakage analysis
        """
        report = LeakageReport()

        # Collect hashes for each split
        train_questions = {self._hash_text(p.question): p.id for p in dataset.train}
        train_contexts = {self._hash_text(p.context): p.id for p in dataset.train}

        # Check validation split
        for pair in dataset.validation:
            q_hash = self._hash_text(pair.question)
            c_hash = self._hash_text(pair.context)

            if q_hash in train_questions:
                report.duplicate_questions.append(
                    {
                        "validation_id": pair.id,
                        "train_id": train_questions[q_hash],
                        "question": pair.question[:100],
                    }
                )

            if c_hash in train_contexts:
                report.duplicate_contexts.append(
                    {
                        "validation_id": pair.id,
                        "train_id": train_contexts[c_hash],
                    }
                )

        # Check test split
        for pair in dataset.test:
            q_hash = self._hash_text(pair.question)
            c_hash = self._hash_text(pair.context)

            if q_hash in train_questions:
                report.duplicate_questions.append(
                    {
                        "test_id": pair.id,
                        "train_id": train_questions[q_hash],
                        "question": pair.question[:100],
                    }
                )

            if c_hash in train_contexts:
                report.duplicate_contexts.append(
                    {
                        "test_id": pair.id,
                        "train_id": train_contexts[c_hash],
                    }
                )

        # Calculate leakage score
        total_val_test = len(dataset.validation) + len(dataset.test)
        if total_val_test > 0:
            leak_count = len(report.duplicate_questions) + len(
                report.duplicate_contexts
            )
            report.leakage_score = min(1.0, leak_count / total_val_test)

        report.has_leakage = report.leakage_score > 0

        logger.info(f"Leakage check: score={report.leakage_score:.3f}")
        return report

    def _hash_text(self, text: str) -> str:
        """Create hash of normalized text."""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def calculate_quality_metrics(self, pairs: List[QAPair]) -> QualityMetrics:
        """
        Calculate quality metrics for a set of pairs.

        Args:
            pairs: List of QAPair objects

        Returns:
            QualityMetrics with calculated scores
        """
        if not pairs:
            return QualityMetrics()

        metrics = QualityMetrics()

        # Completeness - check required fields
        complete_count = sum(1 for p in pairs if p.question and p.answer and p.context)
        metrics.completeness = complete_count / len(pairs)

        # Diversity - unique words ratio
        all_words = []
        for p in pairs:
            all_words.extend(p.question.lower().split())
            all_words.extend(p.answer.lower().split())

        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            metrics.diversity_score = min(1.0, unique_ratio * 2)  # Scale up

        # Balance - distribution evenness
        categories = Counter(p.category for p in pairs)
        if len(categories) > 1:
            counts = list(categories.values())
            avg_count = sum(counts) / len(counts)
            variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
            cv = (variance**0.5) / avg_count if avg_count > 0 else 0
            metrics.balance_score = max(0, 1 - cv)
        else:
            metrics.balance_score = 1.0

        # Average lengths
        metrics.avg_question_length = sum(len(p.question) for p in pairs) / len(pairs)
        metrics.avg_answer_length = sum(len(p.answer) for p in pairs) / len(pairs)

        # Unique question ratio
        unique_questions = len(set(p.question for p in pairs))
        metrics.unique_question_ratio = unique_questions / len(pairs)

        return metrics

    def validate(
        self,
        dataset: DatasetSplit,
        dataset_name: str = "unnamed",
    ) -> ValidationReport:
        """
        Perform comprehensive validation of a dataset.

        Args:
            dataset: DatasetSplit to validate
            dataset_name: Name for the report

        Returns:
            ValidationReport with complete analysis
        """
        report = ValidationReport(dataset_name=dataset_name)
        report.split_info = dataset.to_dict()

        # Calculate quality metrics for each split
        report.train_quality = self.calculate_quality_metrics(dataset.train)
        report.validation_quality = self.calculate_quality_metrics(dataset.validation)
        report.test_quality = self.calculate_quality_metrics(dataset.test)

        # Check for leakage
        report.leakage_report = self.check_leakage(dataset)

        # Calculate distributions
        report.difficulty_distribution = self._get_distribution(dataset, "difficulty")
        report.type_distribution = self._get_distribution(dataset, "question_type")
        report.category_distribution = self._get_distribution(dataset, "category")

        # Identify issues
        report.issues = self._identify_issues(dataset, report)
        report.warnings = self._identify_warnings(dataset, report)

        # Determine validity and grade
        report.is_valid = len(report.issues) == 0
        report.quality_grade = self._calculate_grade(report)

        logger.info(
            f"Validation complete: grade={report.quality_grade}, valid={report.is_valid}"
        )
        return report

    def _get_distribution(
        self,
        dataset: DatasetSplit,
        field: str,
    ) -> Dict[str, Dict[str, int]]:
        """Get distribution of a field across splits."""
        distribution = {"train": {}, "validation": {}, "test": {}}

        for split_name in ["train", "validation", "test"]:
            pairs = getattr(dataset, split_name)
            counter: Counter = Counter()

            for pair in pairs:
                if field == "difficulty":
                    counter[pair.difficulty.value] += 1
                elif field == "question_type":
                    counter[pair.question_type.value] += 1
                elif field == "category":
                    counter[pair.category] += 1

            distribution[split_name] = dict(counter)

        return distribution

    def _identify_issues(
        self,
        dataset: DatasetSplit,
        report: ValidationReport,
    ) -> List[str]:
        """Identify critical issues with the dataset."""
        issues = []

        # Check for empty splits
        if len(dataset.train) == 0:
            issues.append("Training set is empty")
        if len(dataset.validation) == 0:
            issues.append("Validation set is empty")
        if len(dataset.test) == 0:
            issues.append("Test set is empty")

        # Check for severe leakage
        if report.leakage_report and report.leakage_report.leakage_score > 0.3:
            issues.append(
                f"Severe data leakage detected: {report.leakage_report.leakage_score:.1%}"
            )

        # Check for low completeness
        for split_name, quality in [
            ("train", report.train_quality),
            ("validation", report.validation_quality),
            ("test", report.test_quality),
        ]:
            if quality and quality.completeness < 0.8:
                issues.append(
                    f"{split_name} has low completeness: {quality.completeness:.1%}"
                )

        return issues

    def _identify_warnings(
        self,
        dataset: DatasetSplit,
        report: ValidationReport,
    ) -> List[str]:
        """Identify warnings (non-critical issues)."""
        warnings = []

        # Check for minor leakage
        if report.leakage_report:
            if 0 < report.leakage_report.leakage_score <= 0.3:
                warnings.append(
                    f"Minor data leakage detected: {report.leakage_report.leakage_score:.1%}"
                )

        # Check for imbalanced distributions
        for dist_name, distribution in [
            ("difficulty", report.difficulty_distribution),
            ("type", report.type_distribution),
        ]:
            train_dist = distribution.get("train", {})
            if train_dist:
                counts = list(train_dist.values())
                if max(counts) > min(counts) * 3:
                    warnings.append(
                        f"Imbalanced {dist_name} distribution in training set"
                    )

        # Check for low diversity
        if report.train_quality and report.train_quality.diversity_score < 0.5:
            warnings.append("Low vocabulary diversity in training set")

        return warnings

    def _calculate_grade(self, report: ValidationReport) -> str:
        """Calculate overall quality grade."""
        if report.issues:
            return "F"

        # Average quality scores
        scores = []
        for quality in [
            report.train_quality,
            report.validation_quality,
            report.test_quality,
        ]:
            if quality:
                scores.append(quality.overall_score)

        avg_score = sum(scores) / len(scores) if scores else 0

        # Penalize for warnings
        penalty = len(report.warnings) * 0.05
        final_score = max(0, avg_score - penalty)

        if final_score >= 0.9:
            return "A"
        elif final_score >= 0.8:
            return "B"
        elif final_score >= 0.7:
            return "C"
        elif final_score >= 0.6:
            return "D"
        else:
            return "F"

    def export_report(
        self,
        report: ValidationReport,
        output_path: str,
    ) -> str:
        """
        Export validation report to file.

        Args:
            report: ValidationReport to export
            output_path: Path for output file

        Returns:
            Path to exported file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Exported validation report to {output_path}")
        return output_path

    def export_splits(
        self,
        dataset: DatasetSplit,
        output_dir: str,
        format: str = "json",
    ) -> Dict[str, str]:
        """
        Export dataset splits to files.

        Args:
            dataset: DatasetSplit to export
            output_dir: Directory for output files
            format: Output format (json or jsonl)

        Returns:
            Dictionary mapping split names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        for split_name in ["train", "validation", "test"]:
            pairs = getattr(dataset, split_name)
            filename = f"{split_name}.{format}"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                if format == "json":
                    json.dump([p.to_dict() for p in pairs], f, indent=2)
                elif format == "jsonl":
                    for pair in pairs:
                        f.write(json.dumps(pair.to_dict()) + "\n")

            paths[split_name] = str(filepath)

        logger.info(f"Exported splits to {output_dir}")
        return paths

    def generate_statistics_report(
        self,
        dataset: DatasetSplit,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics report.

        Args:
            dataset: DatasetSplit to analyze

        Returns:
            Dictionary with statistics
        """
        all_pairs = dataset.train + dataset.validation + dataset.test

        return {
            "summary": {
                "total_pairs": len(all_pairs),
                "train_size": len(dataset.train),
                "validation_size": len(dataset.validation),
                "test_size": len(dataset.test),
                "actual_ratios": [
                    len(dataset.train) / len(all_pairs) if all_pairs else 0,
                    len(dataset.validation) / len(all_pairs) if all_pairs else 0,
                    len(dataset.test) / len(all_pairs) if all_pairs else 0,
                ],
            },
            "difficulty_counts": dict(Counter(p.difficulty.value for p in all_pairs)),
            "type_counts": dict(Counter(p.question_type.value for p in all_pairs)),
            "category_counts": dict(Counter(p.category for p in all_pairs)),
            "length_stats": {
                "avg_question_length": sum(len(p.question) for p in all_pairs)
                / len(all_pairs)
                if all_pairs
                else 0,
                "avg_answer_length": sum(len(p.answer) for p in all_pairs)
                / len(all_pairs)
                if all_pairs
                else 0,
                "avg_context_length": sum(len(p.context) for p in all_pairs)
                / len(all_pairs)
                if all_pairs
                else 0,
            },
            "special_counts": {
                "negative_examples": sum(1 for p in all_pairs if p.is_negative),
                "edge_cases": sum(1 for p in all_pairs if p.is_edge_case),
            },
        }
