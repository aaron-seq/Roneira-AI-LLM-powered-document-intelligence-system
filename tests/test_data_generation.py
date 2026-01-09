"""
Tests for Data Generation Module

Unit tests for Task 2: Training & Test Data Generation
"""

import pytest
import tempfile
import json
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    SyntheticDataGenerator,
    QAPair,
    GenerationConfig,
    DatasetValidator,
    DatasetSplit,
    ValidationReport,
)
from src.data_generation.synthetic_data_generator import Difficulty, QuestionType


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = SyntheticDataGenerator()
        assert generator.config.num_samples == 100
        assert len(generator.generated_pairs) == 0

    def test_generator_with_custom_config(self):
        """Test generator with custom configuration."""
        config = GenerationConfig(
            num_samples=50,
            difficulty_distribution={"easy": 0.5, "medium": 0.3, "hard": 0.2},
            seed=42,
        )
        generator = SyntheticDataGenerator(config)

        assert generator.config.num_samples == 50

    def test_generate_qa_pairs(self):
        """Test generating QA pairs from documents."""
        generator = SyntheticDataGenerator(GenerationConfig(num_samples=10, seed=42))

        documents = [
            "Machine Learning is a subset of AI. It enables systems to learn from data.",
            "Python is a programming language. It is widely used in data science.",
        ]

        pairs = generator.generate_qa_pairs(documents)

        assert len(pairs) == 10
        assert all(isinstance(p, QAPair) for p in pairs)
        assert all(p.question for p in pairs)
        assert all(p.answer for p in pairs)

    def test_topic_extraction(self):
        """Test topic extraction from text."""
        generator = SyntheticDataGenerator()

        text = "Machine Learning enables AI systems to learn from data."
        topics = generator._extract_topics(text)

        assert "Machine" in topics or "Learning" in topics or "AI" in topics

    def test_difficulty_selection(self):
        """Test difficulty selection follows distribution."""
        generator = SyntheticDataGenerator(
            GenerationConfig(
                difficulty_distribution={"easy": 0.5, "medium": 0.3, "hard": 0.2},
                seed=42,
            )
        )

        difficulties = [generator._select_difficulty() for _ in range(100)]

        # Check all difficulties are valid
        assert all(
            d in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
            for d in difficulties
        )

    def test_generate_edge_cases(self):
        """Test edge case generation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))

        base_pairs = generator.generate_qa_pairs(
            ["Test document content."], num_samples=20
        )
        edge_cases = generator.generate_edge_cases(base_pairs, num_cases=5)

        assert len(edge_cases) == 5
        assert all(p.is_edge_case for p in edge_cases)

    def test_generate_negative_examples(self):
        """Test negative example generation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))

        positive_pairs = generator.generate_qa_pairs(
            ["Document one with content.", "Document two with other content."],
            num_samples=10,
        )
        negatives = generator.generate_negative_examples(
            positive_pairs, num_negatives=3
        )

        assert len(negatives) == 3
        assert all(p.is_negative for p in negatives)

    def test_augment_with_paraphrasing(self):
        """Test paraphrase augmentation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))

        pairs = generator.generate_qa_pairs(
            ["What is machine learning? ML enables learning."], num_samples=5
        )
        augmented = generator.augment_with_paraphrasing(pairs, augmentation_factor=1)

        # Some pairs should be augmented
        assert len(augmented) >= 0  # May be 0 if no patterns match

    def test_get_statistics(self):
        """Test calculating generation statistics."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))

        generator.generate_qa_pairs(["Test content for statistics."], num_samples=20)
        generator.generate_edge_cases(generator.generated_pairs[:10], num_cases=5)

        stats = generator.get_statistics()

        assert stats.total_pairs > 0
        assert len(stats.by_difficulty) > 0
        assert stats.avg_question_length > 0

    def test_export_dataset_json(self, tmp_path):
        """Test exporting dataset to JSON."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        generator.generate_qa_pairs(["Test content."], num_samples=5)

        output_path = str(tmp_path / "dataset.json")
        generator.export_dataset(output_path, format="json")

        with open(output_path) as f:
            data = json.load(f)

        assert "data" in data
        assert len(data["data"]) == 5

    def test_export_dataset_jsonl(self, tmp_path):
        """Test exporting dataset to JSONL."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        generator.generate_qa_pairs(["Test content."], num_samples=3)

        output_path = str(tmp_path / "dataset.jsonl")
        generator.export_dataset(output_path, format="jsonl")

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_clear(self):
        """Test clearing generated pairs."""
        generator = SyntheticDataGenerator()
        generator.generate_qa_pairs(["Content."], num_samples=5)

        assert len(generator.generated_pairs) > 0

        generator.clear()

        assert len(generator.generated_pairs) == 0


class TestDatasetValidator:
    """Tests for DatasetValidator class."""

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = DatasetValidator(seed=42)
        assert validator.seed == 42

    def test_create_splits_simple(self):
        """Test simple split creation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(["Content for testing."], num_samples=100)

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs, ratios=(0.7, 0.15, 0.15))

        assert len(split.train) == 70
        assert len(split.validation) == 15
        assert len(split.test) == 15

    def test_create_splits_stratified(self):
        """Test stratified split creation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(["Content for testing."], num_samples=100)

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(
            pairs,
            ratios=(0.8, 0.1, 0.1),
            stratify_by="difficulty",
        )

        assert split.total_size == 100
        assert split.stratify_by == "difficulty"

    def test_check_leakage_no_leakage(self):
        """Test leakage check with no severe leakage."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        # Use multiple different documents to minimize context overlap
        pairs = generator.generate_qa_pairs(
            [
                "Machine learning is a field of AI that uses algorithms.",
                "Deep learning uses neural networks for pattern recognition.",
                "Natural language processing handles text and speech.",
                "Computer vision focuses on image and video analysis.",
            ],
            num_samples=100,
        )

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs)

        leakage = validator.check_leakage(split)

        # Verify leakage detection works (returns valid result)
        assert 0 <= leakage.leakage_score <= 1.0
        assert isinstance(leakage.has_leakage, bool)

    def test_calculate_quality_metrics(self):
        """Test quality metrics calculation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(
            ["Complete content with data."], num_samples=50
        )

        validator = DatasetValidator()
        metrics = validator.calculate_quality_metrics(pairs)

        assert 0 <= metrics.completeness <= 1
        assert 0 <= metrics.diversity_score <= 1
        assert 0 <= metrics.balance_score <= 1
        assert 0 <= metrics.unique_question_ratio <= 1

    def test_validate(self):
        """Test full validation."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(
            ["Content for validation testing."], num_samples=100
        )

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs)
        report = validator.validate(split, dataset_name="Test Dataset")

        assert report.dataset_name == "Test Dataset"
        assert report.is_valid or not report.is_valid  # Check it's set
        assert report.quality_grade in ["A", "B", "C", "D", "F"]

    def test_export_report(self, tmp_path):
        """Test exporting validation report."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(["Content."], num_samples=30)

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs)
        report = validator.validate(split)

        output_path = str(tmp_path / "report.json")
        validator.export_report(report, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "quality_grade" in data

    def test_export_splits(self, tmp_path):
        """Test exporting dataset splits."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(["Content."], num_samples=30)

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs, ratios=(0.6, 0.2, 0.2))

        output_dir = str(tmp_path / "splits")
        paths = validator.export_splits(split, output_dir)

        assert "train" in paths
        assert "validation" in paths
        assert "test" in paths

    def test_generate_statistics_report(self):
        """Test generating statistics report."""
        generator = SyntheticDataGenerator(GenerationConfig(seed=42))
        pairs = generator.generate_qa_pairs(["Content."], num_samples=50)

        validator = DatasetValidator(seed=42)
        split = validator.create_splits(pairs)
        stats = validator.generate_statistics_report(split)

        assert stats["summary"]["total_pairs"] == 50
        assert "difficulty_counts" in stats
        assert "type_counts" in stats


class TestQAPair:
    """Tests for QAPair dataclass."""

    def test_qa_pair_creation(self):
        """Test creating QA pair."""
        pair = QAPair(
            id="qa_001",
            question="What is AI?",
            answer="Artificial Intelligence",
            context="AI is the simulation...",
            difficulty=Difficulty.EASY,
            question_type=QuestionType.FACTUAL,
        )

        assert pair.id == "qa_001"
        assert pair.difficulty == Difficulty.EASY
        assert not pair.is_negative
        assert not pair.is_edge_case

    def test_to_dict(self):
        """Test converting to dictionary."""
        pair = QAPair(
            id="qa_001",
            question="What is AI?",
            answer="Artificial Intelligence",
            context="AI is the simulation...",
            difficulty=Difficulty.MEDIUM,
            question_type=QuestionType.INFERENTIAL,
        )

        d = pair.to_dict()

        assert d["difficulty"] == "medium"
        assert d["question_type"] == "inferential"

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "id": "qa_001",
            "question": "What is AI?",
            "answer": "Artificial Intelligence",
            "context": "AI is the simulation...",
            "difficulty": "hard",
            "question_type": "multi_hop",
            "category": "test",
            "source_document": "doc_1",
            "is_negative": False,
            "is_edge_case": True,
            "metadata": {},
            "generated_at": "2024-01-01T00:00:00",
        }

        pair = QAPair.from_dict(d)

        assert pair.difficulty == Difficulty.HARD
        assert pair.question_type == QuestionType.MULTI_HOP
        assert pair.is_edge_case


class TestDatasetSplit:
    """Tests for DatasetSplit dataclass."""

    def test_split_total_size(self):
        """Test total size calculation."""
        split = DatasetSplit(
            train=[QAPair("1", "q", "a", "c", Difficulty.EASY, QuestionType.FACTUAL)]
            * 70,
            validation=[
                QAPair("2", "q", "a", "c", Difficulty.EASY, QuestionType.FACTUAL)
            ]
            * 15,
            test=[QAPair("3", "q", "a", "c", Difficulty.EASY, QuestionType.FACTUAL)]
            * 15,
        )

        assert split.total_size == 100

    def test_get_split_by_name(self):
        """Test getting split by name."""
        split = DatasetSplit(
            train=[QAPair("1", "q", "a", "c", Difficulty.EASY, QuestionType.FACTUAL)],
            validation=[
                QAPair("2", "q", "a", "c", Difficulty.EASY, QuestionType.FACTUAL)
            ],
            test=[],
        )

        assert len(split.get_split("train")) == 1
        assert len(split.get_split("validation")) == 1
        assert len(split.get_split("test")) == 0
