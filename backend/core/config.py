"""
Enhanced configuration management with environment-based settings
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class ApplicationSettings(BaseSettings):
    """Application configuration with validation and environment support"""
    
    # Application
    app_name: str = Field(default="AI Document Intelligence System", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    version: str = Field(default="2.0.0", env="APP_VERSION")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./document_intelligence.db",
        env="DATABASE_URL"
    )
    
    # Redis (for production, fallback to in-memory for development)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Azure AI Services
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(default="gpt-4", env="AZURE_OPENAI_DEPLOYMENT")
    azure_document_intelligence_key: Optional[str] = Field(default=None, env="AZURE_DOCUMENT_INTELLIGENCE_KEY")
    azure_document_intelligence_endpoint: Optional[str] = Field(default=None, env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    
    # OpenAI (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # File handling
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")
    processed_files_directory: str = Field(default="./processed", env="PROCESSED_FILES_DIRECTORY")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="ALLOWED_ORIGINS"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )
    
    @validator('allowed_origins', 'allowed_hosts', pre=True)
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_application_settings() -> ApplicationSettings:
    """Get cached application settings"""
    return ApplicationSettings()
