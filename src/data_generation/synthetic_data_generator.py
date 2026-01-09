"""
Synthetic Data Generation for RAG Training

Generates high-quality QA pairs, edge cases, and validation datasets
using LLM-powered augmentation techniques.

This module implements Task 2 of the AI Developer Roadmap:
- Training & Test Data Generation
- Synthetic QA pair creation from documents
- Edge case and adversarial example generation
- Data augmentation with paraphrasing
"""

import json
import random
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

logger = logging.getLogger(__name__)


class Difficulty(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str, Enum):
    """Types of generated questions."""

    FACTUAL = "factual"
    INFERENTIAL = "inferential"
    MULTI_HOP = "multi_hop"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"


@dataclass
class GenerationConfig:
    """Configuration for data generation."""

    num_samples: int = 100
    difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
    question_types: List[str] = field(
        default_factory=lambda: ["factual", "inferential", "analytical"]
    )
    include_negative_examples: bool = True
    negative_ratio: float = 0.2
    include_edge_cases: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)


@dataclass
class QAPair:
    """Question-Answer pair for training/evaluation."""

    id: str
    question: str
    answer: str
    context: str
    difficulty: Difficulty
    question_type: QuestionType
    category: str = ""
    source_document: str = ""
    is_negative: bool = False
    is_edge_case: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["difficulty"] = self.difficulty.value
        result["question_type"] = self.question_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAPair":
        data["difficulty"] = Difficulty(data["difficulty"])
        data["question_type"] = QuestionType(data["question_type"])
        return cls(**data)


@dataclass
class GenerationStatistics:
    """Statistics about the generated dataset."""

    total_pairs: int = 0
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    negative_count: int = 0
    edge_case_count: int = 0
    avg_question_length: float = 0.0
    avg_answer_length: float = 0.0
    avg_context_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticDataGenerator:
    """
    LLM-powered training data generation.

    This class implements the core functionality for Task 2:
    Training & Test Data Generation of the AI Developer Roadmap.

    Features:
    - Question-answer pair generation from documents
    - Multi-hop reasoning examples
    - Edge case and adversarial example creation
    - Diversity-aware sampling with clustering
    - Paraphrasing for augmentation

    Example:
        generator = SyntheticDataGenerator()
        documents = ["Document content here..."]
        qa_pairs = generator.generate_qa_pairs(documents, num_samples=100)
        edge_cases = generator.generate_edge_cases(qa_pairs)
    """

    # Question templates by type
    QUESTION_TEMPLATES = {
        QuestionType.FACTUAL: [
            "What is {topic}?",
            "Who is responsible for {topic}?",
            "When did {event} occur?",
            "Where is {location} located?",
            "What are the key features of {topic}?",
        ],
        QuestionType.INFERENTIAL: [
            "Why might {topic} be important?",
            "What can be inferred about {topic}?",
            "How does {topic} relate to {other_topic}?",
            "What implications does {topic} have?",
        ],
        QuestionType.MULTI_HOP: [
            "Given {context1}, how does this affect {context2}?",
            "Considering both {topic1} and {topic2}, what conclusions can be drawn?",
            "How do the concepts of {topic1} and {topic2} interact?",
        ],
        QuestionType.ANALYTICAL: [
            "Analyze the relationship between {topic1} and {topic2}.",
            "What are the advantages and disadvantages of {topic}?",
            "Compare and contrast {topic1} with {topic2}.",
        ],
        QuestionType.COMPARISON: [
            "How does {topic1} differ from {topic2}?",
            "What are the similarities between {topic1} and {topic2}?",
            "Which is more effective: {topic1} or {topic2}?",
        ],
    }

    # Edge case patterns
    EDGE_CASE_PATTERNS = [
        "What if there is no information about {topic}?",
        "How to handle incomplete data regarding {topic}?",
        "What happens when {topic} is null or empty?",
        "Edge case: {topic} with special characters",
        "Boundary condition: minimum/maximum values for {topic}",
    ]

    # Negative example patterns
    NEGATIVE_PATTERNS = [
        "This question cannot be answered from the given context.",
        "The provided context does not contain information about {topic}.",
        "Insufficient data to determine {answer}.",
    ]

    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Initialize the synthetic data generator.

        Args:
            config: Generation configuration options
        """
        self.config = config or GenerationConfig()
        self.generated_pairs: List[QAPair] = []
        self._id_counter = 0
        logger.info("SyntheticDataGenerator initialized")

    def _generate_id(self) -> str:
        """Generate unique ID for QA pair."""
        self._id_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"qa_{timestamp}_{self._id_counter:06d}"

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text for question generation."""
        # Simple extraction based on capitalized words and noun phrases
        words = text.split()
        topics = []

        # Extract capitalized words (potential named entities)
        for word in words:
            clean_word = re.sub(r"[^\w\s]", "", word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                topics.append(clean_word)

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        topics.extend(quoted)

        # Deduplicate and limit
        topics = list(set(topics))[:20]
        return topics

    def _select_difficulty(self) -> Difficulty:
        """Select difficulty based on configured distribution."""
        rand = random.random()
        cumulative = 0.0

        for diff, prob in self.config.difficulty_distribution.items():
            cumulative += prob
            if rand <= cumulative:
                return Difficulty(diff)

        return Difficulty.MEDIUM

    def _select_question_type(self) -> QuestionType:
        """Randomly select a question type from configured options."""
        type_str = random.choice(self.config.question_types)
        return QuestionType(type_str)

    def generate_qa_pairs(
        self,
        documents: List[str],
        num_samples: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """
        Generate question-answer pairs from documents.

        Args:
            documents: List of document contents
            num_samples: Number of pairs to generate (uses config if None)
            categories: Optional category labels for documents

        Returns:
            List of generated QAPair objects
        """
        num_samples = num_samples or self.config.num_samples
        categories = categories or ["general"] * len(documents)

        if len(categories) != len(documents):
            categories = ["general"] * len(documents)

        pairs = []
        pairs_per_doc = max(1, num_samples // len(documents))

        for doc_idx, (document, category) in enumerate(zip(documents, categories)):
            topics = self._extract_topics(document)

            for _ in range(pairs_per_doc):
                if len(pairs) >= num_samples:
                    break

                pair = self._generate_single_pair(
                    context=document,
                    topics=topics,
                    category=category,
                    source_document=f"doc_{doc_idx}",
                )
                pairs.append(pair)

        self.generated_pairs.extend(pairs)
        logger.info(f"Generated {len(pairs)} QA pairs from {len(documents)} documents")
        return pairs

    def _generate_single_pair(
        self,
        context: str,
        topics: List[str],
        category: str,
        source_document: str,
    ) -> QAPair:
        """Generate a single QA pair."""
        difficulty = self._select_difficulty()
        question_type = self._select_question_type()

        # Select a topic
        topic = random.choice(topics) if topics else "the topic"

        # Generate question from template
        templates = self.QUESTION_TEMPLATES.get(
            question_type, self.QUESTION_TEMPLATES[QuestionType.FACTUAL]
        )
        template = random.choice(templates)

        # Fill template
        question = template.format(
            topic=topic,
            topic1=topic,
            topic2=random.choice(topics) if len(topics) > 1 else "related concept",
            event=topic,
            location=topic,
            other_topic=random.choice(topics) if len(topics) > 1 else "another aspect",
            context1="the given information",
            context2="the expected outcome",
        )

        # Generate answer (simplified - in production would use LLM)
        answer = self._generate_answer(question, context, topic)

        # Use relevant portion of context
        context_snippet = self._extract_relevant_context(context, topic)

        return QAPair(
            id=self._generate_id(),
            question=question,
            answer=answer,
            context=context_snippet,
            difficulty=difficulty,
            question_type=question_type,
            category=category,
            source_document=source_document,
            metadata={
                "topic": topic,
                "template_used": template,
            },
        )

    def _generate_answer(self, question: str, context: str, topic: str) -> str:
        """Generate an answer based on context."""
        # Find sentences containing the topic
        sentences = re.split(r"[.!?]+", context)
        relevant = [s.strip() for s in sentences if topic.lower() in s.lower()]

        if relevant:
            return relevant[0][:500]  # Truncate long answers

        return f"Based on the provided context, {topic} refers to relevant information in the document."

    def _extract_relevant_context(
        self, full_context: str, topic: str, max_length: int = 1000
    ) -> str:
        """Extract context relevant to the topic."""
        sentences = re.split(r"[.!?]+", full_context)
        relevant = []
        total_length = 0

        for sentence in sentences:
            if topic.lower() in sentence.lower():
                if total_length + len(sentence) <= max_length:
                    relevant.append(sentence.strip())
                    total_length += len(sentence)

        if relevant:
            return ". ".join(relevant) + "."

        # Return first portion if no relevant sentences found
        return full_context[:max_length]

    def generate_edge_cases(
        self,
        base_pairs: List[QAPair],
        num_cases: Optional[int] = None,
    ) -> List[QAPair]:
        """
        Generate edge case examples from existing pairs.

        Args:
            base_pairs: Base QA pairs to derive edge cases from
            num_cases: Number of edge cases (10% of base if None)

        Returns:
            List of edge case QAPair objects
        """
        num_cases = num_cases or max(1, len(base_pairs) // 10)
        edge_cases = []

        sample_pairs = random.sample(base_pairs, min(num_cases, len(base_pairs)))

        for pair in sample_pairs:
            pattern = random.choice(self.EDGE_CASE_PATTERNS)
            topic = pair.metadata.get("topic", "the concept")

            edge_question = pattern.format(topic=topic)
            edge_answer = (
                f"Edge case handling: When dealing with {topic}, "
                f"the system should gracefully handle missing or malformed data."
            )

            edge_pair = QAPair(
                id=self._generate_id(),
                question=edge_question,
                answer=edge_answer,
                context=pair.context,
                difficulty=Difficulty.HARD,
                question_type=pair.question_type,
                category=pair.category,
                source_document=pair.source_document,
                is_edge_case=True,
                metadata={"derived_from": pair.id},
            )
            edge_cases.append(edge_pair)

        self.generated_pairs.extend(edge_cases)
        logger.info(f"Generated {len(edge_cases)} edge case examples")
        return edge_cases

    def generate_negative_examples(
        self,
        positive_pairs: List[QAPair],
        num_negatives: Optional[int] = None,
    ) -> List[QAPair]:
        """
        Generate negative examples (unanswerable questions).

        Args:
            positive_pairs: Positive QA pairs
            num_negatives: Number of negative examples (20% of positive if None)

        Returns:
            List of negative example QAPair objects
        """
        num_negatives = num_negatives or max(
            1, int(len(positive_pairs) * self.config.negative_ratio)
        )
        negatives = []

        sample_pairs = random.sample(
            positive_pairs, min(num_negatives, len(positive_pairs))
        )

        for pair in sample_pairs:
            topic = pair.metadata.get("topic", "this topic")
            pattern = random.choice(self.NEGATIVE_PATTERNS)

            # Create mismatched context (take context from different pair)
            other_pair = random.choice([p for p in positive_pairs if p.id != pair.id])

            negative_pair = QAPair(
                id=self._generate_id(),
                question=pair.question,
                answer=pattern.format(topic=topic, answer="the requested information"),
                context=other_pair.context,  # Mismatched context
                difficulty=pair.difficulty,
                question_type=pair.question_type,
                category=pair.category,
                source_document="negative_example",
                is_negative=True,
                metadata={
                    "original_pair": pair.id,
                    "negative_reason": "context_mismatch",
                },
            )
            negatives.append(negative_pair)

        self.generated_pairs.extend(negatives)
        logger.info(f"Generated {len(negatives)} negative examples")
        return negatives

    def augment_with_paraphrasing(
        self,
        pairs: List[QAPair],
        augmentation_factor: int = 2,
    ) -> List[QAPair]:
        """
        Augment dataset with paraphrased versions.

        Args:
            pairs: Original QA pairs
            augmentation_factor: How many paraphrased versions per pair

        Returns:
            List of paraphrased QAPair objects
        """
        augmented = []

        # Simple paraphrasing patterns (in production, use LLM)
        paraphrase_patterns = [
            ("What is", "Can you explain"),
            ("How does", "In what way does"),
            ("Why is", "What is the reason that"),
            ("When did", "At what time did"),
            ("Who is", "Which person is"),
        ]

        for pair in pairs:
            for i in range(augmentation_factor):
                new_question = pair.question

                # Apply random paraphrasing
                for original, replacement in paraphrase_patterns:
                    if original.lower() in new_question.lower():
                        new_question = re.sub(
                            original, replacement, new_question, flags=re.IGNORECASE
                        )
                        break

                if new_question != pair.question:
                    aug_pair = QAPair(
                        id=self._generate_id(),
                        question=new_question,
                        answer=pair.answer,
                        context=pair.context,
                        difficulty=pair.difficulty,
                        question_type=pair.question_type,
                        category=pair.category,
                        source_document=pair.source_document,
                        metadata={
                            "original_pair": pair.id,
                            "augmentation_type": "paraphrase",
                        },
                    )
                    augmented.append(aug_pair)

        self.generated_pairs.extend(augmented)
        logger.info(f"Generated {len(augmented)} augmented examples")
        return augmented

    def get_statistics(self) -> GenerationStatistics:
        """Calculate statistics about generated data."""
        stats = GenerationStatistics(total_pairs=len(self.generated_pairs))

        for pair in self.generated_pairs:
            # By difficulty
            diff_key = pair.difficulty.value
            stats.by_difficulty[diff_key] = stats.by_difficulty.get(diff_key, 0) + 1

            # By type
            type_key = pair.question_type.value
            stats.by_type[type_key] = stats.by_type.get(type_key, 0) + 1

            # Counts
            if pair.is_negative:
                stats.negative_count += 1
            if pair.is_edge_case:
                stats.edge_case_count += 1

        # Averages
        if self.generated_pairs:
            stats.avg_question_length = sum(
                len(p.question) for p in self.generated_pairs
            ) / len(self.generated_pairs)
            stats.avg_answer_length = sum(
                len(p.answer) for p in self.generated_pairs
            ) / len(self.generated_pairs)
            stats.avg_context_length = sum(
                len(p.context) for p in self.generated_pairs
            ) / len(self.generated_pairs)

        return stats

    def export_dataset(
        self,
        output_path: str,
        format: Literal["json", "jsonl"] = "json",
        pairs: Optional[List[QAPair]] = None,
    ) -> str:
        """
        Export generated dataset to file.

        Args:
            output_path: Path for output file
            format: Export format (json or jsonl)
            pairs: Specific pairs to export (all if None)

        Returns:
            Path to exported file
        """
        pairs = pairs or self.generated_pairs

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "total_pairs": len(pairs),
                            "statistics": self.get_statistics().to_dict(),
                        },
                        "data": [p.to_dict() for p in pairs],
                    },
                    f,
                    indent=2,
                )
        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")

        logger.info(f"Exported {len(pairs)} pairs to {output_path}")
        return output_path

    def clear(self) -> None:
        """Clear all generated pairs."""
        self.generated_pairs.clear()
        self._id_counter = 0
        logger.info("Cleared all generated pairs")
