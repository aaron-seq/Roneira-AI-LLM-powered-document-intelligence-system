"""Pydantic response models for the Document Intelligence API.

Defines structured response models for all API endpoints ensuring
consistent response formats and proper documentation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(
        description="Overall system status: healthy, unhealthy, or degraded"
    )
    version: str = Field(description="Application version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    database_status: str = Field(
        default="unknown", description="Database connection status"
    )
    services_status: str = Field(
        default="unknown", description="Overall services status"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "database_status": "connected",
                "services_status": "operational",
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    document_id: str = Field(description="Unique document identifier")
    status: str = Field(
        default="queued",
        description="Processing status: queued, processing, completed, failed",
    )
    message: str = Field(
        default="Document uploaded successfully",
        description="Human-readable status message",
    )
    filename: str = Field(description="Original filename")
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Upload timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Document uploaded successfully and queued for processing",
                "filename": "invoice.pdf",
                "upload_timestamp": "2024-01-15T10:30:00Z",
            }
        }


class DocumentStatusResponse(BaseModel):
    """Response model for document status endpoint."""

    document_id: str = Field(description="Unique document identifier")
    status: str = Field(
        description="Processing status: queued, processing, completed, failed"
    )
    progress: int = Field(
        default=0, ge=0, le=100, description="Processing progress percentage"
    )
    message: Optional[str] = Field(
        default=None, description="Status message or error description"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Document creation timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Processing completion timestamp"
    )
    extracted_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Extracted document data when processing is complete"
    )
    ai_insights: Optional[Dict[str, Any]] = Field(
        default=None, description="AI-generated insights and analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "progress": 100,
                "message": "Document processed successfully",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:45Z",
                "extracted_data": {
                    "text": "Invoice content...",
                    "tables": [],
                    "entities": [],
                },
                "ai_insights": {
                    "summary": "This is an invoice document...",
                    "classification": "financial",
                },
            }
        }


class AuthTokenResponse(BaseModel):
    """Response model for authentication endpoint."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(
        default="bearer", description="Token type for Authorization header"
    )
    expires_in: int = Field(
        default=1800, description="Token expiration time in seconds"
    )
    user_id: Optional[str] = Field(
        default=None, description="Authenticated user identifier"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user_id": "user123",
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(description="Error message")
    error_code: Optional[str] = Field(
        default=None, description="Application-specific error code"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Document not found",
                "error_code": "DOC_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
