"""Database management for the Document Intelligence System.

Provides async SQLAlchemy database connection management with
connection pooling and health check functionality.
"""

import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    """Manages async database connections and sessions.

    Handles connection pooling, session management, and health checks
    for the application database.
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize database connection pool.

        Creates the async engine and session factory based on
        configuration settings.
        """
        settings = get_settings()

        try:
            # Create async engine with appropriate settings
            self._engine = create_async_engine(
                settings.database_url,
                echo=settings.debug,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            # Create tables if they do not exist
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._is_initialized = True
            logger.info(f"Database initialized: {settings.database_url}")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if database connection is healthy, False otherwise.
        """
        if not self._is_initialized or not self._engine:
            return False

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session context manager.

        Yields:
            AsyncSession instance for database operations.
        """
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

    async def close(self) -> None:
        """Close all database connections."""
        if self._engine:
            await self._engine.dispose()
            self._is_initialized = False
            logger.info("Database connections closed")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._is_initialized
