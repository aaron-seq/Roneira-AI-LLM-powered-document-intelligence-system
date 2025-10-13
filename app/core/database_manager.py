"""Database connection and session management.

Provides database initialization, connection pooling, and session management
with proper async/await support and error handling.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, DateTime, Text, Integer
from datetime import datetime

from config import application_settings
from core.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# Global database engine and session factory
database_engine: Optional[AsyncEngine] = None
async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


class DatabaseBase(DeclarativeBase):
    """Base class for all database models."""
    pass


class DocumentRecord(DatabaseBase):
    """Database model for document records."""
    
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    processing_stage = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    extracted_data = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    processing_time_seconds = Column(Integer, nullable=True)


class UserSession(DatabaseBase):
    """Database model for user sessions."""
    
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    session_token = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(String, default="true", nullable=False)  # SQLite compatible boolean
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)


async def initialize_database_connection() -> None:
    """Initialize database engine and create tables."""
    global database_engine, async_session_factory
    
    try:
        # Handle SQLite URL format
        database_url = application_settings.database_url
        if database_url.startswith("sqlite:///"):
            database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # Create async engine
        database_engine = create_async_engine(
            database_url,
            **application_settings.get_database_configuration()
        )
        
        # Create session factory
        async_session_factory = async_sessionmaker(
            database_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with database_engine.begin() as connection:
            await connection.run_sync(DatabaseBase.metadata.create_all)
        
        logger.info("Database connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise DatabaseError(f"Database initialization failed: {e}")


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with proper context management."""
    if not async_session_factory:
        raise DatabaseError("Database not initialized")
    
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            await session.close()


async def health_check_database() -> bool:
    """Check database connectivity and health."""
    try:
        async with get_database_session() as session:
            # Simple query to test connection
            result = await session.execute("SELECT 1")
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def cleanup_database_connection() -> None:
    """Clean up database connections."""
    global database_engine
    
    if database_engine:
        await database_engine.dispose()
        logger.info("Database connections closed")
