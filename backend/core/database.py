"""Async database engine, session factory and schema bootstrap.

Pooling options differ by driver: SQLite (aiosqlite) uses a non-queue pool and
rejects ``pool_size``/``max_overflow``, so those are applied only to server
databases such as PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional
from weakref import WeakKeyDictionary

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alembic.config import Config

logger = logging.getLogger(__name__)

#: Attempts at creating the schema before giving up. Only a worker that keeps
#: losing the startup race needs more than the first one.
_SCHEMA_CREATE_ATTEMPTS = 5
#: Base back-off between attempts, multiplied by the attempt number.
_SCHEMA_RETRY_DELAY = 0.1

#: The first revision, which creates the schema as it stood when migrations
#: were introduced. Databases built by the previous ``create_all`` bootstrap
#: are stamped here rather than migrated, since their tables already exist.
_BASELINE_REVISION = "0001_baseline"

#: Repository root, which is where alembic.ini and migrations/ live.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Serialises migrations, one lock per event loop. See ``_run_migrations`` for
#: why Alembic cannot be driven concurrently in one interpreter.
#:
#: Deliberately not a single module-level ``asyncio.Lock``: a Lock binds to the
#: loop that first acquires it and then raises when used from another, so one
#: shared lock breaks any process that runs a second event loop — a worker that
#: re-enters ``asyncio.run``, a management script, or a test session with both
#: a session-scoped loop and per-test loops. Weak keys so a finished loop does
#: not keep its lock alive.
_MIGRATION_LOCKS: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    WeakKeyDictionary()
)


def _migration_lock() -> asyncio.Lock:
    """The migration lock for the running loop, created on first use."""
    loop = asyncio.get_running_loop()
    lock = _MIGRATION_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _MIGRATION_LOCKS[loop] = lock
    return lock


def _alembic_config() -> "Config":
    """Load alembic.ini with absolute paths.

    Alembic resolves ``script_location`` relative to the process's working
    directory. The application is started from wherever the operator happens
    to be — and from ``/app`` in the container — so the path is pinned to the
    repository instead of inherited from the shell.
    """
    from alembic.config import Config

    config = Config(str(_PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    return config


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
        """Bring the schema up to date, tolerating a concurrent migrator.

        Gunicorn starts every worker at once and each one runs this, so
        whatever happens here has to survive losing a race. The previous
        implementation used ``create_all``, whose ``checkfirst=True`` probes
        for the table and *then* issues CREATE TABLE — two workers could both
        see it missing, both try, and the loser's process died during startup
        with "table documents already exists". That race is timing-dependent,
        so the same image passed CI on one run and exited on the next.

        Alembic has the same shape of problem (two workers both finding the
        database below head) and the same resolution: losing is harmless as
        long as the schema actually arrived, which the recheck establishes. An
        error for any other reason still raises.
        """
        settings = get_settings()
        if not settings.db_auto_migrate:
            # The deployment runs `alembic upgrade head` as its own step.
            # Verify rather than assume: coming up against a schema that is
            # missing tables fails later, in a request, with a worse message.
            missing = await self._missing_tables()
            if missing:
                raise RuntimeError(
                    "DB_AUTO_MIGRATE is off and the database is missing tables: "
                    f"{', '.join(sorted(missing))}. Run `alembic upgrade head` "
                    "before starting the application."
                )
            logger.info("DB_AUTO_MIGRATE is off; assuming migrations were applied")
            return

        last_error: Optional[Exception] = None
        for attempt in range(_SCHEMA_CREATE_ATTEMPTS):
            try:
                await self._run_migrations(url)
                return
            except (OperationalError, ProgrammingError) as exc:
                last_error = exc
                if not await self._missing_tables():
                    logger.info(
                        "Schema was migrated concurrently by another worker; continuing"
                    )
                    return
                # The winner's DDL runs inside a transaction, so for a moment
                # it has taken the name without the table being visible to
                # anyone else. Retrying is what distinguishes that from a
                # database that is genuinely broken.
                await asyncio.sleep(_SCHEMA_RETRY_DELAY * (attempt + 1))

        raise last_error or RuntimeError("Schema migration failed without an error")

    async def _run_migrations(self, url: str) -> None:
        """Apply pending Alembic revisions through our own engine.

        Alembic cannot open its own engine from inside a running event loop,
        so it is handed the connection instead — the documented pattern for
        driving it from an async application. ``migrations/env.py`` picks the
        connection up from ``config.attributes``.

        Two details that are easy to get wrong and fail only at runtime:

        ``connect()``, not ``begin()``
            Alembic opens its own transaction around the migration, and its
            batch mode opens nested ones to rewrite a table. Handing it a
            connection that is already inside ``engine.begin()`` leaves it
            trying to begin a transaction that is no longer there, which
            surfaces as ``ResourceClosedError: This Connection is closed``
            from somewhere deep inside the migration.

        The in-process lock
            Alembic's ``EnvironmentContext`` communicates through module-level
            globals in ``alembic.context``. Two migrations running
            concurrently in one process tear down each other's state, and the
            loser raises ``KeyError: 'script'``. Gunicorn workers are separate
            processes and do not collide here — the cross-process race is
            what the retry in ``_create_schema`` handles — but two
            application instances in one process (tests, embedded use) do.
        """
        engine = self._require_engine()

        async with _migration_lock():
            async with engine.connect() as conn:
                if self._is_sqlite(url):
                    # SQLite ignores FK constraints unless asked; ON DELETE
                    # CASCADE on document_chunks depends on it.
                    await conn.execute(text("PRAGMA foreign_keys=ON"))
                await conn.run_sync(self._upgrade_or_stamp)
                # Alembic reports "will assume non-transactional DDL" on
                # SQLite and so does not wrap the migration in a transaction
                # of its own. Without this commit the DDL lands but the
                # alembic_version row does not, leaving a database whose
                # tables exist and whose recorded revision is empty — which
                # the next boot then tries to migrate again.
                await conn.commit()

    @staticmethod
    def _upgrade_or_stamp(sync_conn) -> None:
        """Migrate to head, or adopt a database that predates migrations.

        A database created by the old ``create_all`` path has every table but
        no ``alembic_version`` row. Running ``upgrade head`` against it would
        replay the baseline revision and fail on tables that already exist, so
        that case is stamped at the baseline instead: the schema is recorded as
        already-at-that-revision and later migrations apply normally.

        The distinction is "has our tables but no version row", not "is
        non-empty" — a database holding somebody else's tables is not one of
        ours to adopt.
        """
        from alembic import command
        from alembic.migration import MigrationContext

        config = _alembic_config()
        config.attributes["connection"] = sync_conn

        context = MigrationContext.configure(sync_conn)
        if context.get_current_revision() is None:
            existing = set(sa_inspect(sync_conn).get_table_names())
            if any(name in existing for name in Base.metadata.tables):
                logger.info(
                    "Adopting a pre-migrations database: stamping it at %s",
                    _BASELINE_REVISION,
                )
                command.stamp(config, _BASELINE_REVISION)

        command.upgrade(config, "head")

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
