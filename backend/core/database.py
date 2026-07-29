"""Async database engine, session factory and schema bootstrap.

Pooling options differ by driver: SQLite (aiosqlite) uses a non-queue pool and
rejects ``pool_size``/``max_overflow``, so those are applied only to server
databases such as PostgreSQL.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class DatabaseManager:
    """Owns the async engine and hands out sessions."""

    def __init__(self, database_url: Optional[str] = None):
        self._database_url = database_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._is_initialized = False

    # ------------------------------------------------------------- lifecycle
    async def initialize(self) -> None:
        """Create the engine, session factory, and schema."""
        settings = get_settings()
        url = self._database_url or settings.database_url

        # Importing the models registers them on Base.metadata. Without this
        # create_all() produces an empty schema.
        from backend.core import models  # noqa: F401  (import for side effects)

        engine_kwargs: Dict[str, Any] = {
            "echo": settings.debug and settings.environment == "development",
            "pool_pre_ping": True,
        }
        if not self._is_sqlite(url):
            # QueuePool options are invalid for SQLite's NullPool/StaticPool.
            engine_kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)

        try:
            self._engine = create_async_engine(url, **engine_kwargs)
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            async with self._engine.begin() as conn:
                if self._is_sqlite(url):
                    # SQLite ignores FK constraints unless asked; ON DELETE
                    # CASCADE on document_chunks depends on it.
                    await conn.execute(text("PRAGMA foreign_keys=ON"))
                await conn.run_sync(Base.metadata.create_all)

            self._is_initialized = True
            logger.info("Database initialized (%s)", self._safe_url(url))

        except Exception:
            logger.exception("Database initialization failed")
            raise

    async def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._is_initialized = False
            logger.info("Database connections closed")

    # ---------------------------------------------------------------- access
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a session, committing on success and rolling back on error."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """Return True when a trivial query succeeds."""
        if not self._is_initialized or not self._engine:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("Database health check failed: %s", exc)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _is_sqlite(url: str) -> bool:
        return url.startswith("sqlite")

    @staticmethod
    def _safe_url(url: str) -> str:
        """Strip credentials so connection strings never reach the logs."""
        if "@" not in url:
            return url
        scheme, _, rest = url.partition("://")
        return f"{scheme}://***@{rest.rpartition('@')[2]}"
