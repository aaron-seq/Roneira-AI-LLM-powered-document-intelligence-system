"""
Configuration management with Ollama support
"""

import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application configuration with Ollama LLM support"""

    # Application
    app_name: str = Field(default="AI Document Intelligence System", env="APP_NAME")
    debug: bool = Field(default=True, env="DEBUG")
    version: str = Field(default="2.0.0", env="APP_VERSION")

    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production", env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./document_intelligence.db", env="DATABASE_URL"
    )

    # Local LLM Configuration (Ollama)
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2:3b", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")  # 2 minutes

    # Document Processing
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    tesseract_path: str = Field(
        default="", env="TESSERACT_PATH"
    )  # Auto-detect if empty

    # File handling
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")
    processed_files_directory: str = Field(
        default="./processed", env="PROCESSED_FILES_DIRECTORY"
    )
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB

    # CORS
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
        ],
        env="ALLOWED_ORIGINS",
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*"], env="ALLOWED_HOSTS"
    )

    @validator("allowed_origins", "allowed_hosts", pre=True)
    def parse_cors_lists(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @property
    def has_ollama_config(self) -> bool:
        """Check if Ollama is configured"""
        return bool(self.ollama_base_url and self.ollama_model)

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
