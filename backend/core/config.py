"""Application configuration.

Settings are read from environment variables (and a local ``.env`` file).
Every field below is a knob a real deployment needs; anything documented in
``.env.example`` is defined here, so documented variables actually take effect.

Note on naming: pydantic-settings v2 resolves environment variables from the
*field name* (case-insensitively). Where the public environment variable name
differs from the field name we set an explicit ``validation_alias`` rather than
the ``env=`` keyword, which pydantic v2 silently ignores.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The placeholder shipped in .env.example. Refused outside development.
INSECURE_SECRET_KEYS = {
    "your-secret-key-change-in-production",
    "your-super-secret-key-change-this-in-production",
    "a-dev-secret-key",
    "changeme",
    "secret",
}


class Settings(BaseSettings):
    """Application configuration."""

    # ---------------------------------------------------------------- app
    app_name: str = Field(
        default="Roneira Document Intelligence",
        validation_alias=AliasChoices("APP_NAME", "app_name"),
    )
    version: str = Field(
        default="2.1.0",
        validation_alias=AliasChoices("APP_VERSION", "VERSION", "version"),
    )
    environment: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("ENVIRONMENT", "environment"),
    )
    debug: bool = Field(default=True)

    # ----------------------------------------------------------- security
    secret_key: str = Field(default="your-secret-key-change-in-production")
    algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM", "ALGORITHM", "algorithm"),
    )
    access_token_expire_minutes: int = Field(default=30, ge=1)

    #: When false, every document/chat endpoint is reachable without a token.
    #: Useful for a local kick-the-tyres run; refused in production.
    require_authentication: bool = Field(default=True)

    #: Requests per minute per client IP, applied to the whole API.
    rate_limit_per_minute: int = Field(default=60, ge=1)
    #: Tighter budget for the expensive endpoints (upload, chat, search).
    rate_limit_expensive_per_minute: int = Field(default=20, ge=1)

    # ----------------------------------------------------------- database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./document_intelligence.db",
    )

    # ---------------------------------------------------------------- llm
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    ollama_timeout: int = Field(default=120, ge=1)

    # ---------------------------------------------------- document intake
    upload_directory: str = Field(default="./uploads")
    processed_files_directory: str = Field(default="./processed")
    max_file_size: int = Field(default=50 * 1024 * 1024, ge=1024)

    #: Keep the uploaded source file after processing. Required for the
    #: "open the original at the cited page" workflow; disable it when a
    #: deployment must not retain customer documents at rest.
    retain_source_files: bool = Field(default=True)
    #: Delete retained source files older than this. 0 disables expiry.
    source_retention_days: int = Field(default=30, ge=0)

    enable_ocr: bool = Field(default=True)
    tesseract_path: str = Field(default="")

    # ---------------------------------------------------------------- rag
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    vector_store_path: str = Field(default="./chroma_db")
    vector_collection_name: str = Field(default="documents")
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    max_conversation_history: int = Field(default=10, ge=1)
    #: Minimum similarity a chunk must reach before it is shown as a citation.
    retrieval_min_score: float = Field(default=0.15, ge=0.0, le=1.0)

    #: Refuse to start unless a real embedding model loads. Otherwise the
    #: service falls back to keyword-only lexical matching, which works but
    #: cannot match paraphrases — set this true where that is unacceptable.
    require_real_embeddings: bool = Field(default=False)

    # --------------------------------------------------------------- cors
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
        ]
    )
    #: Host header allow-list. Permissive by default so local and containerised
    #: runs work without configuration; the production validator below refuses
    #: to start with "*", which forces a real list exactly where it matters.
    allowed_hosts: List[str] = Field(default=["*"])

    # ------------------------------------------------------ observability
    metrics_enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------- validators
    @field_validator("allowed_origins", "allowed_hosts", mode="before")
    @classmethod
    def _parse_csv_list(cls, value):
        """Accept a JSON array or a comma-separated string.

        Both forms appear in the wild — docker-compose files tend to use CSV,
        Kubernetes manifests JSON — and accepting only one silently produces a
        single-element list containing the whole string.
        """
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON list: {value!r}") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
            return [str(item).strip() for item in parsed if str(item).strip()]

        return [item.strip() for item in stripped.split(",") if item.strip()]

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported log level: {value}")
        return level

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking never advances."
            )
        return self

    @model_validator(mode="after")
    def _enforce_production_hardening(self) -> "Settings":
        """Fail fast on configurations that are unsafe outside development.

        A demo secret key or an unauthenticated API is fine on a laptop and
        catastrophic on the internet. Catching it at startup is far cheaper
        than discovering it from an incident.
        """
        if self.environment != "production":
            return self

        problems: List[str] = []
        if self.secret_key in INSECURE_SECRET_KEYS:
            problems.append(
                "SECRET_KEY is still the example placeholder. Generate one with "
                '`python -c "import secrets; print(secrets.token_urlsafe(48))"`.'
            )
        if len(self.secret_key) < 32:
            problems.append("SECRET_KEY must be at least 32 characters.")
        if not self.require_authentication:
            problems.append(
                "REQUIRE_AUTHENTICATION=false exposes every document endpoint "
                "anonymously and is not permitted in production."
            )
        if "*" in self.allowed_origins:
            problems.append("ALLOWED_ORIGINS must not be '*' with credentialed CORS.")
        if "*" in self.allowed_hosts:
            problems.append("ALLOWED_HOSTS must not be '*' in production.")
        if self.debug:
            problems.append("DEBUG must be false in production.")

        if problems:
            raise ValueError(
                "Refusing to start with an unsafe production configuration:\n  - "
                + "\n  - ".join(problems)
            )
        return self

    # ---------------------------------------------------------- properties
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def has_ollama_config(self) -> bool:
        return bool(self.ollama_base_url and self.ollama_model)

    @property
    def docs_enabled(self) -> bool:
        """Interactive API docs are on everywhere except production."""
        return not self.is_production


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache and re-read the environment.

    Only used by tests and by ``scripts/`` helpers that mutate the environment.
    """
    get_settings.cache_clear()
    return get_settings()
