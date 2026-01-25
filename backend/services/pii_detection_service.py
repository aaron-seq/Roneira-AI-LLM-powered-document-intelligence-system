"""
Enhanced PII Detection and Redaction Service.

Provides comprehensive detection and redaction of personally
identifiable information (PII) across multiple categories.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    SSN = "SSN"                     # Social Security Number
    CREDIT_CARD = "CREDIT_CARD"     # Credit card numbers
    EMAIL = "EMAIL"                  # Email addresses
    PHONE = "PHONE"                  # Phone numbers
    NAME = "NAME"                    # Person names
    ADDRESS = "ADDRESS"              # Physical addresses
    DATE_OF_BIRTH = "DOB"           # Date of birth
    PASSPORT = "PASSPORT"            # Passport numbers
    DRIVERS_LICENSE = "DL"          # Driver's license
    BANK_ACCOUNT = "BANK_ACCOUNT"   # Bank account numbers
    IP_ADDRESS = "IP_ADDRESS"       # IP addresses
    MEDICAL_ID = "MEDICAL_ID"       # Medical record numbers
    NATIONAL_ID = "NATIONAL_ID"     # National ID numbers
    CUSTOM = "CUSTOM"               # Custom patterns


class RedactionMethod(str, Enum):
    """Methods for redacting PII."""
    MASK = "mask"           # Replace with [REDACTED]
    HASH = "hash"           # Replace with hash
    PARTIAL = "partial"     # Partially mask (show last 4)
    CATEGORY = "category"   # Replace with category label
    ENCRYPT = "encrypt"     # Encrypt the value
    REMOVE = "remove"       # Remove entirely


@dataclass
class PIIMatch:
    """A detected PII match.
    
    Attributes:
        pii_type: Type of PII detected.
        original_text: Original text that was matched.
        start_pos: Start position in text.
        end_pos: End position in text.
        confidence: Detection confidence (0-1).
        redacted_text: Redacted version of the text.
    """
    pii_type: PIIType
    original_text: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    redacted_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.pii_type.value,
            "original": self.original_text,
            "position": {"start": self.start_pos, "end": self.end_pos},
            "confidence": self.confidence,
            "redacted": self.redacted_text,
        }


@dataclass
class RedactionConfig:
    """Configuration for PII redaction.
    
    Attributes:
        method: Default redaction method.
        per_type_methods: Override methods per PII type.
        mask_char: Character to use for masking.
        show_last_n: Characters to show for partial masking.
        enabled_types: PII types to detect (None = all).
        custom_patterns: Custom regex patterns to detect.
    """
    method: RedactionMethod = RedactionMethod.MASK
    per_type_methods: Dict[PIIType, RedactionMethod] = field(default_factory=dict)
    mask_char: str = "*"
    show_last_n: int = 4
    enabled_types: Optional[Set[PIIType]] = None
    custom_patterns: Dict[str, str] = field(default_factory=dict)


class PIIDetector:
    """Pattern-based PII detection."""
    
    def __init__(self):
        """Initialize PII detector with patterns."""
        self._patterns = {
            PIIType.SSN: [
                (r'\b\d{3}-\d{2}-\d{4}\b', 0.95),
                (r'\b\d{9}\b', 0.7),  # Lower confidence for 9 digits
            ],
            PIIType.CREDIT_CARD: [
                (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b', 0.95),
                (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 0.85),
            ],
            PIIType.EMAIL: [
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.98),
            ],
            PIIType.PHONE: [
                (r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.90),
                (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.85),
            ],
            PIIType.IP_ADDRESS: [
                (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 0.90),
                (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', 0.90),  # IPv6
            ],
            PIIType.DATE_OF_BIRTH: [
                (r'\b(?:dob|date\s+of\s+birth|birthdate)[:\s]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', 0.95),
                (r'\b(?:born|birthday)[:\s]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', 0.90),
            ],
            PIIType.PASSPORT: [
                (r'\b[A-Z]{1,2}\d{6,9}\b', 0.75),  # Various formats
            ],
            PIIType.DRIVERS_LICENSE: [
                (r'\b(?:DL|driver\'?s?\s*license)[:\s#]*[A-Z0-9-]{5,15}\b', 0.85),
            ],
            PIIType.BANK_ACCOUNT: [
                (r'\b(?:account|acct)[:\s#]*\d{8,17}\b', 0.85),
                (r'\b(?:routing)[:\s#]*\d{9}\b', 0.90),
            ],
            PIIType.ADDRESS: [
                (r'\b\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\.?\b', 0.80),
            ],
        }
        
        # Common name patterns (lower confidence)
        self._name_patterns = [
            (r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', 0.85),
        ]
    
    def detect(
        self,
        text: str,
        enabled_types: Optional[Set[PIIType]] = None,
        custom_patterns: Optional[Dict[str, str]] = None
    ) -> List[PIIMatch]:
        """Detect PII in text.
        
        Args:
            text: Text to scan for PII.
            enabled_types: PII types to detect (None = all).
            custom_patterns: Additional patterns to check.
            
        Returns:
            List of PII matches found.
        """
        matches = []
        
        # Check standard patterns
        for pii_type, patterns in self._patterns.items():
            if enabled_types and pii_type not in enabled_types:
                continue
            
            for pattern, confidence in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Validate match (e.g., Luhn for credit cards)
                    if self._validate_match(pii_type, match.group()):
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            original_text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence
                        ))
        
        # Check name patterns
        if not enabled_types or PIIType.NAME in enabled_types:
            for pattern, confidence in self._name_patterns:
                for match in re.finditer(pattern, text):
                    matches.append(PIIMatch(
                        pii_type=PIIType.NAME,
                        original_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence
                    ))
        
        # Check custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append(PIIMatch(
                        pii_type=PIIType.CUSTOM,
                        original_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.80
                    ))
        
        return self._deduplicate_matches(matches)
    
    def _validate_match(self, pii_type: PIIType, text: str) -> bool:
        """Validate a match with additional checks."""
        if pii_type == PIIType.CREDIT_CARD:
            return self._luhn_check(text)
        if pii_type == PIIType.IP_ADDRESS:
            return self._valid_ip(text)
        return True
    
    def _luhn_check(self, num_str: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        digits = re.sub(r'\D', '', num_str)
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        total = 0
        for i, digit in enumerate(reversed(digits)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        
        return total % 10 == 0
    
    def _valid_ip(self, ip_str: str) -> bool:
        """Validate IP address."""
        parts = ip_str.split('.')
        if len(parts) != 4:
            return False
        
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False
    
    def _deduplicate_matches(
        self,
        matches: List[PIIMatch]
    ) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Sort by start position, then by confidence (descending)
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.start_pos, -m.confidence)
        )
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            if match.start_pos >= last_end:
                result.append(match)
                last_end = match.end_pos
        
        return result


class PIIRedactor:
    """Redact detected PII from text."""
    
    def __init__(self, config: Optional[RedactionConfig] = None):
        """Initialize redactor.
        
        Args:
            config: Redaction configuration.
        """
        self.config = config or RedactionConfig()
    
    def redact(
        self,
        text: str,
        matches: List[PIIMatch]
    ) -> Tuple[str, List[PIIMatch]]:
        """Redact PII from text.
        
        Args:
            text: Original text.
            matches: Detected PII matches.
            
        Returns:
            Tuple of (redacted text, updated matches with redacted text).
        """
        if not matches:
            return text, []
        
        # Sort matches by position (reverse order for replacement)
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)
        
        result = text
        for match in sorted_matches:
            redacted = self._create_redaction(match)
            match.redacted_text = redacted
            result = result[:match.start_pos] + redacted + result[match.end_pos:]
        
        return result, matches
    
    def _create_redaction(self, match: PIIMatch) -> str:
        """Create redacted version of PII."""
        method = self.config.per_type_methods.get(
            match.pii_type,
            self.config.method
        )
        
        if method == RedactionMethod.MASK:
            return f"[{match.pii_type.value}]"
        
        elif method == RedactionMethod.PARTIAL:
            text = match.original_text
            n = self.config.show_last_n
            if len(text) > n:
                masked = self.config.mask_char * (len(text) - n)
                return masked + text[-n:]
            return self.config.mask_char * len(text)
        
        elif method == RedactionMethod.HASH:
            import hashlib
            h = hashlib.sha256(match.original_text.encode()).hexdigest()[:8]
            return f"[{match.pii_type.value}:{h}]"
        
        elif method == RedactionMethod.CATEGORY:
            return f"<{match.pii_type.value}>"
        
        elif method == RedactionMethod.REMOVE:
            return ""
        
        else:  # Default to mask
            return f"[{match.pii_type.value}]"


class PIIDetectionService:
    """Main PII detection and redaction service.
    
    Provides comprehensive PII detection with configurable
    redaction methods and reporting.
    """
    
    def __init__(self, config: Optional[RedactionConfig] = None):
        """Initialize PII service.
        
        Args:
            config: Redaction configuration.
        """
        self.detector = PIIDetector()
        self.redactor = PIIRedactor(config)
        self.config = config or RedactionConfig()
    
    def detect(
        self,
        text: str,
        enabled_types: Optional[Set[PIIType]] = None
    ) -> List[PIIMatch]:
        """Detect PII in text.
        
        Args:
            text: Text to scan.
            enabled_types: PII types to detect.
            
        Returns:
            List of PII matches.
        """
        return self.detector.detect(
            text,
            enabled_types or self.config.enabled_types,
            self.config.custom_patterns
        )
    
    def redact(
        self,
        text: str,
        enabled_types: Optional[Set[PIIType]] = None
    ) -> Tuple[str, List[PIIMatch]]:
        """Detect and redact PII from text.
        
        Args:
            text: Text to process.
            enabled_types: PII types to detect.
            
        Returns:
            Tuple of (redacted text, matches found).
        """
        matches = self.detect(text, enabled_types)
        return self.redactor.redact(text, matches)
    
    def generate_report(
        self,
        matches: List[PIIMatch]
    ) -> Dict[str, Any]:
        """Generate PII detection report.
        
        Args:
            matches: Detected PII matches.
            
        Returns:
            Report dictionary.
        """
        type_counts: Dict[str, int] = {}
        high_confidence = 0
        low_confidence = 0
        
        for match in matches:
            type_name = match.pii_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            if match.confidence >= 0.9:
                high_confidence += 1
            elif match.confidence < 0.8:
                low_confidence += 1
        
        return {
            "total_pii_found": len(matches),
            "pii_by_type": type_counts,
            "high_confidence_matches": high_confidence,
            "low_confidence_matches": low_confidence,
            "matches": [m.to_dict() for m in matches],
        }
    
    def batch_process(
        self,
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Process multiple documents.
        
        Args:
            documents: List of {"id": ..., "text": ...} dicts.
            
        Returns:
            List of results per document.
        """
        results = []
        
        for doc in documents:
            redacted, matches = self.redact(doc["text"])
            results.append({
                "document_id": doc["id"],
                "redacted_text": redacted,
                "report": self.generate_report(matches)
            })
        
        return results


# Global instance
_pii_service: Optional[PIIDetectionService] = None


def get_pii_service(config: Optional[RedactionConfig] = None) -> PIIDetectionService:
    """Get the global PII detection service."""
    global _pii_service
    if _pii_service is None:
        _pii_service = PIIDetectionService(config)
    return _pii_service
