"""
Database System for the LLM Document Intelligence System

This module sets up the asynchronous database connection using SQLAlchemy 2.0+
and defines the data models for the application. It provides a structured way
to interact with the database.

Key Components:
- SQLAlchemy async engine and session management.
- Declarative base for defining ORM models.
- Data models for Users, Documents, and Processing Jobs.
- Dependency for getting a database session in FastAPI routes.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from config import settings

logger = logging.getLogger(__name__)

# --- Database Engine and Session Configuration ---

# The async engine is the core of our database connection.
# It manages a pool of connections to the database.
db_engine = create_async_engine(
    settings.database_url,
    **settings.get_database_configuration(),
)

# The session maker is a factory for creating new database sessions.
AsyncSessionLocal = async_sessionmaker(
    bind=db_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects accessible after commit
)

# The declarative base is a base class for all our ORM models.
Base = declarative_base()


# --- Mixins for Common Columns ---


class TimestampMixin:
    """A mixin to add `created_at` and `updated_at` columns to a model."""

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# --- Data Models ---


class User(Base, TimestampMixin):
    """Represents a user of the application."""

    __tablename__ = "users"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)

    # Relationships
    documents = relationship("Document", back_populates="owner")


class Document(Base, TimestampMixin):
    """Represents a document uploaded by a user."""

    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)

    # Processing information
    processing_status = Column(String(20), default="pending", index=True)
    extracted_data = Column(JSON)

    # Foreign Keys
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Relationships
    owner = relationship("User", back_populates="documents")
    processing_jobs = relationship("ProcessingJob", back_populates="document")


class ProcessingJob(Base, TimestampMixin):
    """Represents a job for processing a document."""

    __tablename__ = "processing_jobs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), default="queued", index=True)
    result = Column(JSON)

    # Foreign Keys
    document_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )

    # Relationships
    document = relationship("Document", back_populates="processing_jobs")


# --- Database Session Management ---


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    A FastAPI dependency that provides a database session to a route.
    It handles the session lifecycle (creation, commit, rollback, close).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def managed_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    A context manager for database sessions, useful for background tasks.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


# --- Database Initialization ---


async def initialize_database():
    """Creates all database tables defined in the models."""
    try:
        async with db_engine.begin() as conn:
            logger.info("Creating database tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise
