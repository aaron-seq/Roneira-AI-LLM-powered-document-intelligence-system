"""
Guardrail Service for Input/Output Validation

Provides content filtering, PII detection, and output
validation for safe AI interactions.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Categories of potentially harmful content."""

    SAFE = "safe"
    PII = "pii"
    PROFANITY = "profanity"
    HARMFUL = "harmful"
    INJECTION = "injection"
    SPAM = "spam"


@dataclass
class ValidationResult:
    """Result of content validation."""

    is_valid: bool
    category: ContentCategory
    confidence: float
    details: str
    filtered_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "category": self.category.value,
            "confidence": self.confidence,
            "details": self.details,
            "filtered_content": self.filtered_content,
        }


class PIIDetector:
    """Detector for Personally Identifiable Information."""

    # Common PII patterns
    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(
            r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b"
        ),
        "ssn": re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text."""
        findings = []

        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                findings.append(
                    {
                        "type": pii_type,
                        "value": match,
                        "masked": self._mask_value(match, pii_type),
                    }
                )

        return findings

    def mask_pii(self, text: str) -> str:
        """Mask PII in text."""
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            masked = pattern.sub(
                lambda m: self._mask_value(m.group(), pii_type), masked
            )
        return masked

    def _mask_value(self, value: str, pii_type: str) -> str:
        """Create masked version of PII value."""
        if pii_type == "email":
            parts = value.split("@")
            return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_type == "phone":
            return "***-***-" + value[-4:]
        elif pii_type == "ssn":
            return "***-**-" + value[-4:]
        elif pii_type == "credit_card":
            return "****-****-****-" + value[-4:]
        else:
            return "*" * len(value)


class ContentFilter:
    """Filter for harmful or inappropriate content."""

    # Basic profanity list (would be more comprehensive in production)
    PROFANITY_PATTERNS = [
        # Add patterns as needed
    ]

    # Injection attack patterns
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(previous|all|above)\s+instructions?", re.I),
        re.compile(r"disregard\s+(previous|all|above)", re.I),
        re.compile(r"you\s+are\s+now\s+", re.I),
        re.compile(r"pretend\s+you\s+are", re.I),
        re.compile(r"act\s+as\s+if\s+you", re.I),
        re.compile(r"jailbreak", re.I),
        re.compile(r"DAN\s*mode", re.I),
    ]

    def check_injection(self, text: str) -> Tuple[bool, str]:
        """Check for prompt injection attempts."""
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return True, f"Potential injection detected: {pattern.pattern}"
        return False, ""

    def check_profanity(self, text: str) -> Tuple[bool, str]:
        """Check for profanity."""
        for pattern in self.PROFANITY_PATTERNS:
            if re.search(pattern, text, re.I):
                return True, "Profanity detected"
        return False, ""

    def check_spam(self, text: str) -> Tuple[bool, str]:
        """Check for spam-like content."""
        # Check for excessive repetition
        words = text.lower().split()
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            max_ratio = max(word_counts.values()) / len(words)
            if max_ratio > 0.5 and len(words) > 10:
                return True, "Excessive repetition detected"

        # Check for excessive caps
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.7:
                return True, "Excessive capitalization detected"

        return False, ""


class GuardrailService:
    """
    Main service for input/output validation and content safety.
    """

    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_content_filter: bool = True,
        auto_mask_pii: bool = True,
        max_input_length: int = 10000,
        max_output_length: int = 50000,
    ):
        self.enable_pii_detection = enable_pii_detection
        self.enable_content_filter = enable_content_filter
        self.auto_mask_pii = auto_mask_pii
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.pii_detector = PIIDetector()
        self.content_filter = ContentFilter()

        self.is_initialized = True
        logger.info("GuardrailService initialized")

    async def validate_input(self, text: str, strict: bool = False) -> ValidationResult:
        """
        Validate user input before processing.

        Args:
            text: Input text to validate
            strict: If True, block on any warning

        Returns:
            ValidationResult with validation status
        """
        # Length check
        if len(text) > self.max_input_length:
            return ValidationResult(
                is_valid=False,
                category=ContentCategory.SPAM,
                confidence=1.0,
                details=f"Input exceeds maximum length of {self.max_input_length}",
            )

        # Empty check
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                category=ContentCategory.SAFE,
                confidence=1.0,
                details="Input is empty",
            )

        # Content filter checks
        if self.enable_content_filter:
            # Check for injection
            is_injection, injection_detail = self.content_filter.check_injection(text)
            if is_injection:
                return ValidationResult(
                    is_valid=False,
                    category=ContentCategory.INJECTION,
                    confidence=0.8,
                    details=injection_detail,
                )

            # Check for spam
            is_spam, spam_detail = self.content_filter.check_spam(text)
            if is_spam:
                return ValidationResult(
                    is_valid=False if strict else True,
                    category=ContentCategory.SPAM,
                    confidence=0.7,
                    details=spam_detail,
                )

        # PII check
        filtered_text = text
        if self.enable_pii_detection:
            pii_findings = self.pii_detector.detect(text)
            if pii_findings:
                if self.auto_mask_pii:
                    filtered_text = self.pii_detector.mask_pii(text)

                return ValidationResult(
                    is_valid=True,
                    category=ContentCategory.PII,
                    confidence=0.9,
                    details=f"PII detected: {[f['type'] for f in pii_findings]}",
                    filtered_content=filtered_text,
                )

        return ValidationResult(
            is_valid=True,
            category=ContentCategory.SAFE,
            confidence=1.0,
            details="Input passed all checks",
            filtered_content=filtered_text,
        )

    async def validate_output(self, text: str) -> ValidationResult:
        """
        Validate LLM output before returning to user.

        Args:
            text: Output text to validate

        Returns:
            ValidationResult with validation status
        """
        # Length check
        if len(text) > self.max_output_length:
            truncated = text[: self.max_output_length] + "... [truncated]"
            return ValidationResult(
                is_valid=True,
                category=ContentCategory.SAFE,
                confidence=1.0,
                details="Output truncated due to length",
                filtered_content=truncated,
            )

        # PII masking in output
        filtered_text = text
        if self.enable_pii_detection:
            pii_findings = self.pii_detector.detect(text)
            if pii_findings and self.auto_mask_pii:
                filtered_text = self.pii_detector.mask_pii(text)
                return ValidationResult(
                    is_valid=True,
                    category=ContentCategory.PII,
                    confidence=0.9,
                    details="PII masked in output",
                    filtered_content=filtered_text,
                )

        return ValidationResult(
            is_valid=True,
            category=ContentCategory.SAFE,
            confidence=1.0,
            details="Output passed all checks",
            filtered_content=filtered_text,
        )

    async def filter_context(self, chunks: List[str]) -> List[str]:
        """Filter retrieved context chunks for PII."""
        if not self.enable_pii_detection:
            return chunks

        filtered = []
        for chunk in chunks:
            if self.auto_mask_pii:
                filtered.append(self.pii_detector.mask_pii(chunk))
            else:
                filtered.append(chunk)

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get guardrail service statistics."""
        return {
            "pii_detection_enabled": self.enable_pii_detection,
            "content_filter_enabled": self.enable_content_filter,
            "auto_mask_pii": self.auto_mask_pii,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
        }
