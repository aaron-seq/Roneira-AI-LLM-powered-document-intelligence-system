"""Custom exceptions for the Document Intelligence System.

Provides specific exception types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class DocumentIntelligenceBaseException(Exception):
    """Base exception for document intelligence system."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DocumentProcessingError(DocumentIntelligenceBaseException):
    """Raised when document processing fails."""
    
    def __init__(
        self, 
        message: str, 
        document_id: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.document_id = document_id
        self.stage = stage


class ValidationError(DocumentIntelligenceBaseException):
    """Raised when input validation fails."""
    pass


class AuthenticationError(DocumentIntelligenceBaseException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(DocumentIntelligenceBaseException):
    """Raised when authorization fails."""
    pass


class ServiceUnavailableError(DocumentIntelligenceBaseException):
    """Raised when a required service is unavailable."""
    pass


class ConfigurationError(DocumentIntelligenceBaseException):
    """Raised when configuration is invalid."""
    pass


class ExternalServiceError(DocumentIntelligenceBaseException):
    """Raised when external service calls fail."""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.status_code = status_code


class DatabaseError(DocumentIntelligenceBaseException):
    """Raised when database operations fail."""
    pass


class CacheError(DocumentIntelligenceBaseException):
    """Raised when cache operations fail."""
    pass
