"""
Advanced Text Splitter Service for Document Chunking

Provides multiple text splitting strategies for preparing documents for
embedding and vector storage in a RAG pipeline.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    @abstractmethod
    def split_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text into chunks."""
        pass

    def _create_chunks(
        self, texts: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Create TextChunk objects from list of text strings."""
        chunks = []
        current_pos = 0

        for i, text in enumerate(texts):
            if text.strip():  # Skip empty chunks
                chunk = TextChunk(
                    content=text.strip(),
                    chunk_index=len(chunks),
                    start_char=current_pos,
                    end_char=current_pos + len(text),
                    metadata=metadata or {},
                )
                chunks.append(chunk)
            current_pos += len(text)

        return chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursively splits text by different separators.
    Tries to keep semantically related content together.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.keep_separator = keep_separator

    def split_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text recursively using separators."""
        final_chunks = self._split_text_recursive(text, self.separators)
        return self._create_chunks(final_chunks, metadata)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        # Find the appropriate separator
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by the separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Process splits
        good_splits = []
        for split in splits:
            if self.keep_separator and separator:
                if good_splits:
                    good_splits[-1] += separator

            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            elif new_separators:
                # Recursively split large chunks
                recursive_chunks = self._split_text_recursive(split, new_separators)
                final_chunks.extend(recursive_chunks)
            else:
                good_splits.append(split)

        # Merge small splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits while respecting chunk_size."""
        merged_chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    merged_chunks.append("".join(current_chunk))
                    # Handle overlap
                    overlap_text = "".join(current_chunk)[-self.chunk_overlap :]
                    current_chunk = [overlap_text] if self.chunk_overlap > 0 else []
                    current_length = len(overlap_text) if self.chunk_overlap > 0 else 0

            current_chunk.append(split)
            current_length += split_length

        if current_chunk:
            merged_chunks.append("".join(current_chunk))

        return merged_chunks


class SentenceTextSplitter(BaseTextSplitter):
    """
    Splits text by sentences while respecting chunk size.
    Better for maintaining semantic coherence.
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size

    def split_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text by sentences."""
        sentences = self._split_into_sentences(text)
        merged = self._merge_sentences(sentences)
        return self._create_chunks(merged, metadata)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks."""
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = self.length_function(sentence)

            # If single sentence exceeds chunk_size, add it as its own chunk
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                chunks.append(sentence)
                current_chunk = []
                current_length = 0
                continue

            # Check if adding this sentence exceeds chunk_size
            if current_length + sentence_length + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Handle overlap by keeping last few sentences
                    overlap_sentences = self._get_overlap_sentences(current_chunk)
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap from the end of chunk."""
        if not self.chunk_overlap or not sentences:
            return []

        overlap_sentences = []
        overlap_length = 0

        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break

        return overlap_sentences


class TextSplitterService:
    """
    Main service for text splitting operations.
    Provides a unified interface for different splitting strategies.
    """

    def __init__(
        self, default_chunk_size: int = 1000, default_chunk_overlap: int = 200
    ):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.is_initialized = True
        logger.info("TextSplitterService initialized")

    def split_document(
        self,
        text: str,
        strategy: str = "recursive",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Split document text using specified strategy.

        Args:
            text: The text to split
            strategy: Splitting strategy ('recursive', 'sentence')
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            metadata: Additional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap

        splitter = self._get_splitter(strategy, chunk_size, chunk_overlap)
        chunks = splitter.split_text(text, metadata)

        logger.info(
            f"Split document into {len(chunks)} chunks using {strategy} strategy "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

        return chunks

    def _get_splitter(
        self, strategy: str, chunk_size: int, chunk_overlap: int
    ) -> BaseTextSplitter:
        """Get the appropriate splitter for the strategy."""
        splitters = {
            "recursive": RecursiveCharacterTextSplitter,
            "sentence": SentenceTextSplitter,
        }

        splitter_class = splitters.get(strategy, RecursiveCharacterTextSplitter)
        return splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {"total_chunks": 0}

        char_counts = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(char_counts) / len(chunks),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(chunks),
        }
