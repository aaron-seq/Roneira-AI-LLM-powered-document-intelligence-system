"""Alembic environment.

Three things make this file more than the generated default:

**The URL comes from the application's own settings.** ``alembic.ini`` carries
no ``sqlalchemy.url``, so ``alembic upgrade head`` and the running service can
never disagree about which database they are talking to.

**It accepts a connection from the caller.** The application applies pending
migrations during startup from inside a live event loop, where Alembic cannot
open its own engine. When ``config.attributes["connection"]`` is set, that
connection is used; this is the documented pattern for driving Alembic from an
async application.

**The command line path is async too.** It drives the same ``+aiosqlite`` and
``+asyncpg`` drivers the application uses, rather than requiring a second,
synchronous driver (psycopg2) to be installed for the sake of the CLI.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importing the models is what populates Base.metadata. Without it,
# autogenerate sees an empty schema and cheerfully writes a migration that
# drops every table.
import backend.core.models  # noqa: F401  (imported for side effects)
from backend.core.config import get_settings
from backend.core.database import Base

config = context.config

# Only configure logging for the command-line path. In-application, the host
# already owns the logging configuration and fileConfig would replace it.
if config.config_file_name is not None and not config.attributes.get("connection"):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _configure(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER a column in place. Batch mode rewrites the table
        # instead, which is the only way a column change reaches a SQLite
        # database — and SQLite is the default here. No-op on PostgreSQL.
        render_as_batch=True,
        # Without this, a changed column type is silently not migrated.
        compare_type=True,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    ``alembic upgrade head --sql`` shows exactly what a release will do to a
    database before it does it.
    """
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:
    """Run the migrations on a synchronous connection."""
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    """Open an async engine and run the migrations through it."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = get_settings().database_url

    engine = async_engine_from_config(section, prefix="sqlalchemy.")
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    existing = config.attributes.get("connection")
    if existing is not None:
        # Handed to us by backend.core.database, already inside an event loop.
        _run(existing)
        return

    asyncio.run(_run_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
