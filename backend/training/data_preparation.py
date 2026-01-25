"""
Data Preparation Service for LLM Training.

Provides utilities for preparing document data for fine-tuning,
including text formatting, dataset creation, and quality filtering.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example.
    
    Attributes:
        input_text: Input/prompt text.
        output_text: Expected output/completion.
        metadata: Optional metadata about the example.
    """
    input_text: str
    output_text: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_chat_format(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Convert to chat message format.
        
        Args:
            system_prompt: Optional system message.
            
        Returns:
            List of chat messages.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.input_text})
        messages.append({"role": "assistant", "content": self.output_text})
        return messages
    
    def to_instruction_format(self) -> str:
        """Convert to instruction format string.
        
        Returns:
            Formatted instruction string.
        """
        return f"### Instruction:\n{self.input_text}\n\n### Response:\n{self.output_text}"


class DataPreparationService:
    """Service for preparing training data from documents.
    
    Handles document-to-training data conversion with support for
    multiple output formats and quality filtering.
    """
    
    def __init__(
        self,
        min_input_length: int = 10,
        max_input_length: int = 4096,
        min_output_length: int = 5,
        max_output_length: int = 2048,
    ):
        """Initialize data preparation service.
        
        Args:
            min_input_length: Minimum input text length.
            max_input_length: Maximum input text length.
            min_output_length: Minimum output text length.
            max_output_length: Maximum output text length.
        """
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self._processed_hashes: set = set()
    
    def create_document_qa_examples(
        self,
        document_text: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        num_questions: int = 5
    ) -> List[TrainingExample]:
        """Create Q&A training examples from document.
        
        Generates synthetic question-answer pairs based on
        document content for fine-tuning.
        
        Args:
            document_text: Full document text.
            document_metadata: Optional document metadata.
            num_questions: Target number of Q&A pairs.
            
        Returns:
            List of TrainingExample objects.
        """
        examples = []
        
        # Split document into sections
        sections = self._split_into_sections(document_text)
        
        for section in sections[:num_questions]:
            if len(section.strip()) < self.min_output_length:
                continue
            
            # Generate question from section
            question = self._generate_question_prompt(section, document_metadata)
            
            example = TrainingExample(
                input_text=question,
                output_text=section.strip(),
                metadata={
                    "source": "document_qa",
                    **(document_metadata or {})
                }
            )
            
            if self._is_valid_example(example):
                examples.append(example)
        
        return examples
    
    def create_summarization_examples(
        self,
        document_text: str,
        summary: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> TrainingExample:
        """Create summarization training example.
        
        Args:
            document_text: Full document text.
            summary: Document summary.
            document_metadata: Optional metadata.
            
        Returns:
            TrainingExample for summarization.
        """
        prompt = f"Summarize the following document:\n\n{document_text[:self.max_input_length]}"
        
        return TrainingExample(
            input_text=prompt,
            output_text=summary,
            metadata={
                "source": "summarization",
                **(document_metadata or {})
            }
        )
    
    def create_extraction_examples(
        self,
        document_text: str,
        extracted_entities: Dict[str, List[str]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[TrainingExample]:
        """Create entity extraction training examples.
        
        Args:
            document_text: Full document text.
            extracted_entities: Dict of entity types to values.
            document_metadata: Optional metadata.
            
        Returns:
            List of TrainingExample objects.
        """
        examples = []
        
        for entity_type, entities in extracted_entities.items():
            if not entities:
                continue
            
            prompt = (
                f"Extract all {entity_type} from the following document:\n\n"
                f"{document_text[:self.max_input_length]}"
            )
            
            output = json.dumps({entity_type: entities}, indent=2)
            
            example = TrainingExample(
                input_text=prompt,
                output_text=output,
                metadata={
                    "source": "entity_extraction",
                    "entity_type": entity_type,
                    **(document_metadata or {})
                }
            )
            
            if self._is_valid_example(example):
                examples.append(example)
        
        return examples
    
    def prepare_dataset(
        self,
        examples: List[TrainingExample],
        output_format: str = "chat",
        system_prompt: Optional[str] = None,
        deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepare examples for training dataset.
        
        Args:
            examples: List of training examples.
            output_format: 'chat', 'instruction', or 'completion'.
            system_prompt: System prompt for chat format.
            deduplicate: Remove duplicate examples.
            
        Returns:
            List of formatted training records.
        """
        dataset = []
        
        for example in examples:
            if deduplicate:
                example_hash = self._compute_hash(example)
                if example_hash in self._processed_hashes:
                    continue
                self._processed_hashes.add(example_hash)
            
            if output_format == "chat":
                record = {
                    "messages": example.to_chat_format(system_prompt),
                    "metadata": example.metadata
                }
            elif output_format == "instruction":
                record = {
                    "text": example.to_instruction_format(),
                    "metadata": example.metadata
                }
            else:  # completion format
                record = {
                    "prompt": example.input_text,
                    "completion": example.output_text,
                    "metadata": example.metadata
                }
            
            dataset.append(record)
        
        logger.info(f"Prepared {len(dataset)} training examples in {output_format} format")
        return dataset
    
    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl"
    ) -> None:
        """Save dataset to file.
        
        Args:
            dataset: Prepared dataset records.
            output_path: Output file path.
            format: 'jsonl' or 'json'.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if format == "jsonl":
                for record in dataset:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(dataset)} examples to {output_path}")
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections."""
        # Split by common section delimiters
        patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=[A-Z][a-z]+:)',  # Headers
            r'\n(?=\d+\.)',  # Numbered lists
        ]
        
        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(pattern, section))
            sections = new_sections
        
        return [s.strip() for s in sections if s.strip()]
    
    def _generate_question_prompt(
        self,
        section: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a question prompt for a section."""
        doc_type = metadata.get("document_type", "document") if metadata else "document"
        
        templates = [
            f"Based on the {doc_type}, explain: ",
            f"What information is provided about: ",
            f"Summarize the following from the {doc_type}: ",
            f"Extract key details regarding: ",
        ]
        
        # Use first few words as topic
        words = section.split()[:10]
        topic = " ".join(words) + "..."
        
        import random
        template = random.choice(templates)
        return f"{template}\n\n{topic}"
    
    def _is_valid_example(self, example: TrainingExample) -> bool:
        """Check if example meets quality criteria."""
        input_len = len(example.input_text)
        output_len = len(example.output_text)
        
        if input_len < self.min_input_length or input_len > self.max_input_length:
            return False
        
        if output_len < self.min_output_length or output_len > self.max_output_length:
            return False
        
        # Basic quality checks
        if not example.input_text.strip() or not example.output_text.strip():
            return False
        
        return True
    
    def _compute_hash(self, example: TrainingExample) -> str:
        """Compute hash for deduplication."""
        content = f"{example.input_text}|{example.output_text}"
        return hashlib.md5(content.encode()).hexdigest()
