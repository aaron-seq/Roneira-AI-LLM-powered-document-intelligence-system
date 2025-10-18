"""Core utilities and services for the Document Intelligence System.

Provides authentication, database management, WebSocket connections,
and custom exceptions for the application.
"""

# Authentication utilities
from .authentication import (
    create_access_token,
    verify_access_token,
    get_authenticated_user,
    hash_password,
    verify_password,
    RequireRole,
    require_admin,
    require_user
)

# Database management
from .database_manager import (
    get_database_session,
    initialize_database_connection
)

# WebSocket management
from .websocket_manager import WebSocketConnectionManager

# Custom exceptions
from .exceptions import (
    DocumentIntelligenceBaseException,
    DocumentProcessingError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
    ConfigurationError,
    ExternalServiceError,
    DatabaseError,
    CacheError
)

__all__ = [
    # Authentication
    "create_access_token",
    "verify_access_token",
    "get_authenticated_user",
    "hash_password",
    "verify_password",
    "RequireRole",
    "require_admin",
    "require_user",
    
    # Database
    "get_database_session",
    "initialize_database_connection",
    
    # WebSocket
    "WebSocketConnectionManager",
    
    # Exceptions
    "DocumentIntelligenceBaseException",
    "DocumentProcessingError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",
    "ConfigurationError",
    "ExternalServiceError",
    "DatabaseError",
    "CacheError"
]
