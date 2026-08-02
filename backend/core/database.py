"""Async database engine, session factory and schema bootstrap.

Pooling options differ by driver: SQLite (aiosqlite) uses a non-queue pool and
rejects ``pool_size``/``max_overflow``, so those are applied only to server
databases such as PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

#: Attempts at creating the schema before giving up. Only a worker that keeps
#: losing the startup race needs more than the first one.
_SCHEMA_CREATE_ATTEMPTS = 5
#: Base back-off between attempts, multiplied by the attempt number.
_SCHEMA_RETRY_DELAY = 0.1


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

            await self._create_schema(url)

            self._is_initialized = True
            logger.info("Database initialized (%s)", self._safe_url(url))

        except Exception:
            logger.exception("Database initialization failed")
            raise

    async def _create_schema(self, url: str) -> None:
        """Create any missing tables, tolerating a concurrent creator.

        Gunicorn starts every worker at once and each one runs this. SQLAlchemy's
        ``create_all`` defaults to ``checkfirst=True``, which probes for the
        table and *then* issues CREATE TABLE — so two workers can both see it
        missing and both try to create it. The loser gets "table documents
        already exists" and the whole process dies during startup.

        That race is timing-dependent, so it does not fail every time: the same
        image passed CI on one run and exited on the next. Losing the race is
        harmless as long as the schema is actually there, which is what the
        recheck below establishes. An error for any other reason still raises.
        """
        engine = self._require_engine()

        last_error: Optional[Exception] = None
        for attempt in range(_SCHEMA_CREATE_ATTEMPTS):
            try:
                async with engine.begin() as conn:
                    if self._is_sqlite(url):
                        # SQLite ignores FK constraints unless asked; ON DELETE
                        # CASCADE on document_chunks depends on it.
                        await conn.execute(text("PRAGMA foreign_keys=ON"))
                    await conn.run_sync(Base.metadata.create_all)
                return
            except (OperationalError, ProgrammingError) as exc:
                last_error = exc
                if not await self._missing_tables():
                    logger.info(
                        "Schema was created concurrently by another worker; continuing"
                    )
                    return
                # The winner's CREATE runs inside a transaction, so for a
                # moment it has taken the name without the table being visible
                # to anyone else. Retrying is what distinguishes that from a
                # database that is genuinely broken.
                await asyncio.sleep(_SCHEMA_RETRY_DELAY * (attempt + 1))

        raise last_error or RuntimeError("Schema creation failed without an error")

    def _require_engine(self) -> AsyncEngine:
        """Return the engine, or say plainly that there is not one.

        Not an ``assert``: those are stripped under ``python -O``, which is
        exactly the configuration where a missing engine would turn into an
        unexplained AttributeError deep in a request.
        """
        if self._engine is None:
            raise RuntimeError("Database engine is not initialised")
        return self._engine

    async def _missing_tables(self) -> list[str]:
        """Return the model tables that are not present in the database."""

        def _inspect(sync_conn) -> list[str]:
            existing = set(sa_inspect(sync_conn).get_table_names())
            return [name for name in Base.metadata.tables if name not in existing]

        async with self._require_engine().connect() as conn:
            return await conn.run_sync(_inspect)

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
