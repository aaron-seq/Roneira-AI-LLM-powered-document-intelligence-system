"""
Application configuration management for the Document Intelligence System.

Provides secure, validated configuration loading from environment variables
with proper validation and environment-specific settings.
"""

import os
from typing import List, Optional
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import field_validator, Field, ConfigDict


class ApplicationConfiguration(BaseSettings):
    """Main application configuration with environment-based settings."""

    # Application metadata
    application_name: str = Field(
        default="Document Intelligence System", env="APP_NAME"
    )
    application_version: str = Field(default="2.0.0", env="VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug_mode: bool = Field(default=False, env="DEBUG")

    # Server configuration
    server_host: str = Field(default="0.0.0.0", env="HOST")
    server_port: int = Field(default=8000, env="PORT")
    worker_count: int = Field(default=4, env="WORKERS")

    # Security settings
    secret_key: str = Field(default="", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256")
    token_expiry_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # CORS configuration
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "https://*.vercel.app",
            "https://*.railway.app",
            "https://*.render.com",
        ]
    )

    # Database configuration
    database_url: str = Field(
        default="sqlite:///./document_intelligence.db", env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_connection_pool_size: int = Field(default=10)

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434", env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3", env="OLLAMA_MODEL")

    # Azure OpenAI configuration
    azure_openai_api_key: str = Field(default="", env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(
        default="2024-02-01", env="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4-turbo-preview", env="AZURE_OPENAI_DEPLOYMENT_NAME"
    )

    # Azure Document Intelligence configuration
    azure_doc_intelligence_key: str = Field(
        default="", env="AZURE_DOCUMENT_INTELLIGENCE_KEY"
    )
    azure_doc_intelligence_endpoint: str = Field(
        default="", env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
    )

    # File upload settings
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    allowed_file_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]
    )
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")

    # API rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)

    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Ensure environment is one of the valid options."""
        valid_environments = ["development", "staging", "production"]
        if value not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return value

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        """Ensure secret key is properly configured for production."""
        if not value:
            # For development, provide a default key
            return "dev-secret-key-change-in-production"

        if len(value) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")

        return value

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def max_file_size_bytes(self) -> int:
        """Convert file size from MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def get_database_configuration(self) -> dict:
        """Get database connection configuration."""
        return {
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug_mode,
        }

    def get_azure_openai_configuration(self) -> dict:
        """Get Azure OpenAI configuration."""
        return {
            "api_key": self.azure_openai_api_key,
            "azure_endpoint": self.azure_openai_endpoint,  # Fixed: was api_base
            "api_version": self.azure_openai_api_version,
            "deployment_name": self.azure_openai_deployment_name,
        }

    def get_azure_document_intelligence_config(self) -> dict:
        """Get Azure Document Intelligence configuration.

        Returns configuration dictionary for Azure Document Intelligence service.
        This method was missing and caused import errors in services.
        """
        return {
            "key": self.azure_doc_intelligence_key,
            "endpoint": self.azure_doc_intelligence_endpoint,
        }

    def get_redis_configuration(self) -> dict:
        """Get Redis configuration."""
        config = {
            "url": self.redis_url,
            "max_connections": self.redis_connection_pool_size,
        }
        if self.redis_password:
            config["password"] = self.redis_password
        return config

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class DevelopmentConfiguration(ApplicationConfiguration):
    """Development-specific configuration."""

    debug_mode: bool = True
    log_level: str = "DEBUG"


class ProductionConfiguration(ApplicationConfiguration):
    """Production-specific configuration."""

    debug_mode: bool = False
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_application_settings() -> ApplicationConfiguration:
    """Get application settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionConfiguration()
    else:
        return DevelopmentConfiguration()


# Global settings instance
application_settings = get_application_settings()

# Legacy alias for backward compatibility
settings = application_settings
