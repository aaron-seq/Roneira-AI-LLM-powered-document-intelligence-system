"""
Document Summarization Service.

Provides configurable document summarization with
extractive and abstractive methods, and length control.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SummaryLength(str, Enum):
    """Summary length options."""
    BRIEF = "brief"         # 1-2 sentences
    SHORT = "short"         # 3-5 sentences
    MEDIUM = "medium"       # 1-2 paragraphs
    LONG = "long"           # 3-4 paragraphs
    COMPREHENSIVE = "comprehensive"  # Full summary


class SummaryMethod(str, Enum):
    """Summarization method."""
    EXTRACTIVE = "extractive"     # Select important sentences
    ABSTRACTIVE = "abstractive"   # Generate new summary text
    HYBRID = "hybrid"             # Combine both approaches


@dataclass
class SummaryConfig:
    """Configuration for summarization.
    
    Attributes:
        length: Desired summary length.
        method: Summarization method.
        max_sentences: Maximum sentences (for extractive).
        max_words: Maximum words in output.
        preserve_key_phrases: Phrases that must appear.
        focus_topics: Topics to emphasize.
    """
    length: SummaryLength = SummaryLength.MEDIUM
    method: SummaryMethod = SummaryMethod.HYBRID
    max_sentences: int = 10
    max_words: int = 300
    preserve_key_phrases: List[str] = field(default_factory=list)
    focus_topics: List[str] = field(default_factory=list)
    
    @property
    def sentence_targets(self) -> Dict[SummaryLength, int]:
        """Get sentence count targets per length."""
        return {
            SummaryLength.BRIEF: 2,
            SummaryLength.SHORT: 5,
            SummaryLength.MEDIUM: 10,
            SummaryLength.LONG: 20,
            SummaryLength.COMPREHENSIVE: 30,
        }


@dataclass
class SummaryResult:
    """Result of document summarization.
    
    Attributes:
        summary: The generated summary.
        method_used: Method that was used.
        original_length: Original document word count.
        summary_length: Summary word count.
        compression_ratio: Compression achieved.
        key_phrases: Important phrases extracted.
    """
    summary: str
    method_used: SummaryMethod
    original_length: int
    summary_length: int
    compression_ratio: float
    key_phrases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "method": self.method_used.value,
            "original_words": self.original_length,
            "summary_words": self.summary_length,
            "compression_ratio": round(self.compression_ratio, 2),
            "key_phrases": self.key_phrases,
        }


class ExtractiveSummarizer:
    """Extractive summarization using sentence scoring."""
    
    def __init__(self):
        """Initialize extractive summarizer."""
        self._stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "and",
            "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "this", "that", "these", "those", "it"
        }
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        focus_keywords: Optional[List[str]] = None
    ) -> str:
        """Extract most important sentences.
        
        Args:
            text: Document text to summarize.
            num_sentences: Number of sentences to extract.
            focus_keywords: Keywords to boost sentence scores.
            
        Returns:
            Extractive summary.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences
        word_freq = self._calculate_word_frequencies(text)
        scores = []
        
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, word_freq, focus_keywords)
            # Boost early sentences slightly (position bias)
            position_boost = 1.0 + (0.1 * (1 - i / len(sentences)))
            scores.append((i, sent, score * position_boost))
        
        # Select top sentences
        scores.sort(key=lambda x: x[2], reverse=True)
        selected = sorted(scores[:num_sentences], key=lambda x: x[0])
        
        return " ".join(sent for _, sent, _ in selected)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    def _calculate_word_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequencies."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self._stopwords]
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        if freq:
            max_freq = max(freq.values())
            freq = {k: v / max_freq for k, v in freq.items()}
        
        return freq
    
    def _score_sentence(
        self,
        sentence: str,
        word_freq: Dict[str, float],
        focus_keywords: Optional[List[str]] = None
    ) -> float:
        """Score a sentence based on word importance."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        
        if not words:
            return 0.0
        
        score = sum(word_freq.get(w, 0) for w in words) / len(words)
        
        # Boost for focus keywords
        if focus_keywords:
            for keyword in focus_keywords:
                if keyword.lower() in sentence.lower():
                    score *= 1.5
        
        return score


class AbstractiveSummarizer:
    """Abstractive summarization using LLM."""
    
    def __init__(self, llm_provider=None):
        """Initialize abstractive summarizer.
        
        Args:
            llm_provider: LLM for generation.
        """
        self._llm = llm_provider
    
    async def summarize(
        self,
        text: str,
        length: SummaryLength = SummaryLength.MEDIUM,
        focus_topics: Optional[List[str]] = None
    ) -> str:
        """Generate abstractive summary.
        
        Args:
            text: Document text.
            length: Desired summary length.
            focus_topics: Topics to emphasize.
            
        Returns:
            Abstractive summary.
        """
        if not self._llm:
            raise ValueError("LLM provider required for abstractive summarization")
        
        length_instructions = {
            SummaryLength.BRIEF: "Write a 1-2 sentence summary.",
            SummaryLength.SHORT: "Write a brief summary of 3-5 sentences.",
            SummaryLength.MEDIUM: "Write a summary of 1-2 paragraphs.",
            SummaryLength.LONG: "Write a comprehensive summary of 3-4 paragraphs.",
            SummaryLength.COMPREHENSIVE: "Write a detailed summary covering all main points.",
        }
        
        focus_instruction = ""
        if focus_topics:
            focus_instruction = f"\nFocus especially on: {', '.join(focus_topics)}"
        
        prompt = (
            f"{length_instructions[length]}{focus_instruction}\n\n"
            f"Document:\n{text[:8000]}\n\n"  # Limit input
            f"Summary:"
        )
        
        response = await self._llm.generate(prompt)
        return response.content.strip()


class DocumentSummarizer:
    """Main document summarization service.
    
    Provides configurable summarization with multiple methods
    and automatic fallback.
    """
    
    def __init__(self, llm_provider=None):
        """Initialize summarizer.
        
        Args:
            llm_provider: Optional LLM for abstractive summarization.
        """
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(llm_provider)
        self._llm = llm_provider
    
    async def summarize(
        self,
        text: str,
        config: Optional[SummaryConfig] = None
    ) -> SummaryResult:
        """Summarize a document.
        
        Args:
            text: Document text.
            config: Summarization configuration.
            
        Returns:
            SummaryResult with summary and metadata.
        """
        config = config or SummaryConfig()
        original_words = len(text.split())
        
        try:
            if config.method == SummaryMethod.EXTRACTIVE:
                summary = self._extractive_summarize(text, config)
                method_used = SummaryMethod.EXTRACTIVE
                
            elif config.method == SummaryMethod.ABSTRACTIVE:
                if self._llm:
                    summary = await self.abstractive.summarize(
                        text, config.length, config.focus_topics
                    )
                    method_used = SummaryMethod.ABSTRACTIVE
                else:
                    summary = self._extractive_summarize(text, config)
                    method_used = SummaryMethod.EXTRACTIVE
                    
            else:  # HYBRID
                summary = await self._hybrid_summarize(text, config)
                method_used = SummaryMethod.HYBRID
            
            summary_words = len(summary.split())
            compression = summary_words / original_words if original_words else 0
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(text)
            
            return SummaryResult(
                summary=summary,
                method_used=method_used,
                original_length=original_words,
                summary_length=summary_words,
                compression_ratio=compression,
                key_phrases=key_phrases
            )
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple truncation
            truncated = " ".join(text.split()[:config.max_words])
            return SummaryResult(
                summary=truncated,
                method_used=SummaryMethod.EXTRACTIVE,
                original_length=original_words,
                summary_length=len(truncated.split()),
                compression_ratio=len(truncated.split()) / original_words if original_words else 0
            )
    
    def _extractive_summarize(
        self,
        text: str,
        config: SummaryConfig
    ) -> str:
        """Perform extractive summarization."""
        targets = config.sentence_targets
        num_sentences = targets.get(config.length, 10)
        
        return self.extractive.summarize(
            text,
            num_sentences=min(num_sentences, config.max_sentences),
            focus_keywords=config.focus_topics
        )
    
    async def _hybrid_summarize(
        self,
        text: str,
        config: SummaryConfig
    ) -> str:
        """Perform hybrid summarization."""
        # First extract key sentences
        extracted = self._extractive_summarize(text, config)
        
        # Then refine with LLM if available
        if self._llm:
            prompt = (
                f"Rewrite and improve this summary to be more coherent "
                f"and readable:\n\n{extracted}\n\nImproved summary:"
            )
            response = await self._llm.generate(prompt)
            return response.content.strip()
        
        return extracted
    
    def _extract_key_phrases(
        self,
        text: str,
        top_k: int = 5
    ) -> List[str]:
        """Extract key phrases from text."""
        # Simple n-gram extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Get bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        from collections import Counter
        bigram_counts = Counter(bigrams)
        
        # Filter out common patterns
        common_starts = {"the", "of", "in", "to", "and", "for"}
        filtered = [
            phrase for phrase, count in bigram_counts.most_common(20)
            if not phrase.split()[0] in common_starts and count > 1
        ]
        
        return filtered[:top_k]
    
    async def summarize_multiple(
        self,
        documents: List[Dict[str, str]],
        config: Optional[SummaryConfig] = None
    ) -> List[SummaryResult]:
        """Summarize multiple documents.
        
        Args:
            documents: List of {"id": ..., "text": ...} dicts.
            config: Shared summarization config.
            
        Returns:
            List of SummaryResults.
        """
        tasks = [
            self.summarize(doc["text"], config)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)


# Global instance
_summarizer: Optional[DocumentSummarizer] = None


def get_summarizer(llm_provider=None) -> DocumentSummarizer:
    """Get the global summarizer."""
    global _summarizer
    if _summarizer is None:
        _summarizer = DocumentSummarizer(llm_provider)
    return _summarizer
