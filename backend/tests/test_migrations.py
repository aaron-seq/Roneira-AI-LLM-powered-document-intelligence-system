"""Schema migrations.

The schema used to be created with ``Base.metadata.create_all``, which creates
missing *tables* and nothing else. A new column therefore reached a developer's
fresh database, passed CI — which starts from an empty SQLite file every run —
and then did nothing at all on any database that already existed, where it
failed at query time instead. On the deployment that matters most.

These tests cover the three situations a release actually meets: a database
that does not exist yet, a database created by the old ``create_all``
bootstrap, and a database somebody else migrates. Plus the drift check, which
is what stops a model change from being merged without the migration that
delivers it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from backend.core.config import reload_settings
from backend.core.database import Base, DatabaseManager


def _tables(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        connection.close()


def _revision(path: Path) -> str | None:
    connection = sqlite3.connect(path)
    try:
        rows = list(connection.execute("SELECT version_num FROM alembic_version"))
        return rows[0][0] if rows else None
    except sqlite3.OperationalError:
        return None  # no alembic_version table at all
    finally:
        connection.close()


@pytest.fixture
def database(tmp_path, monkeypatch):
    """A throwaway database path, with settings pointed at it.

    The settings cache is rebuilt on the way out: these tests deliberately
    repoint DATABASE_URL, and leaving that in place would send every later
    test in the session at this temporary file.
    """
    path = tmp_path / "migrate.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    reload_settings()
    yield path
    monkeypatch.undo()
    reload_settings()


async def _build_legacy_database(url: str) -> None:
    """Create the schema the way the pre-migrations bootstrap did."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import backend.core.models  # noqa: F401  (registers the tables)

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


class TestAFreshDatabase:
    @pytest.mark.asyncio
    async def test_startup_creates_the_schema_and_records_the_revision(self, database):
        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        assert {"documents", "document_chunks", "chat_feedback"} <= _tables(database)
        assert _revision(database) is not None, "the applied revision must be recorded"

    @pytest.mark.asyncio
    async def test_the_revision_is_committed_not_just_applied(self, database):
        """Alembic reports "non-transactional DDL" on SQLite and so does not
        wrap the migration in a transaction of its own. Without an explicit
        commit the DDL landed but the ``alembic_version`` row did not, leaving
        a database whose tables exist and whose recorded revision is empty —
        which the next boot then tried to migrate again, onto tables that were
        already there.
        """
        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        assert _revision(database) == "0001_baseline"

    @pytest.mark.asyncio
    async def test_a_second_startup_is_a_no_op(self, database):
        for _ in range(2):
            manager = DatabaseManager()
            await manager.initialize()
            assert await manager.health_check()
            await manager.close()

        assert _revision(database) == "0001_baseline"


class TestAnExistingDatabaseFromBeforeMigrations:
    """Upgrading an installation that predates this change must not destroy it.

    Such a database has every table and no ``alembic_version`` row. Running
    ``upgrade head`` against it would replay the baseline revision onto tables
    that already exist and fail. Startup stamps it instead.
    """

    @pytest.mark.asyncio
    async def test_it_is_stamped_rather_than_rebuilt(self, database):
        await _build_legacy_database(f"sqlite+aiosqlite:///{database}")
        assert _revision(database) is None, "fixture should have no version row"

        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        assert _revision(database) == "0001_baseline"

    @pytest.mark.asyncio
    async def test_existing_rows_survive(self, database):
        """The point of stamping rather than re-creating: the data is still there."""
        await _build_legacy_database(f"sqlite+aiosqlite:///{database}")

        connection = sqlite3.connect(database)
        connection.execute(
            "INSERT INTO chat_feedback (id, message_id, is_positive, created_at) "
            "VALUES ('f1', 'm1', 1, '2026-01-01 00:00:00')"
        )
        connection.commit()
        connection.close()

        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        connection = sqlite3.connect(database)
        try:
            rows = list(connection.execute("SELECT id FROM chat_feedback"))
        finally:
            connection.close()

        assert rows == [("f1",)], "stamping must not touch the data"


class TestConcurrentStartup:
    """Gunicorn starts every worker at once, and each one runs this.

    Workers are separate processes, so the cross-process race is handled by the
    retry-and-recheck in ``_create_schema``. Two application instances in one
    process — tests, embedded use — hit a different problem: Alembic's
    ``EnvironmentContext`` talks through module-level globals, and two
    concurrent migrations tear down each other's state, with the loser raising
    ``KeyError: 'script'``.
    """

    @pytest.mark.asyncio
    async def test_three_simultaneous_boots_all_succeed(self, database):
        managers = [DatabaseManager() for _ in range(3)]

        await asyncio.gather(*(m.initialize() for m in managers))

        try:
            for manager in managers:
                assert await manager.health_check()
            assert _revision(database) == "0001_baseline"
        finally:
            for manager in managers:
                await manager.close()

    def test_migrating_from_two_different_event_loops_works(self, tmp_path):
        """The serialising lock must not be bound to one loop.

        An ``asyncio.Lock`` created at module scope attaches to whichever loop
        first acquires it and then raises for any other, so a single shared
        lock breaks every process that runs a second event loop: a worker
        re-entering ``asyncio.run``, a management script, or a test session
        holding both a session-scoped loop and per-test loops. Caught exactly
        that way — the suite's own second loop.
        """
        url = f"sqlite+aiosqlite:///{tmp_path / 'two-loops.db'}"

        async def boot() -> bool:
            manager = DatabaseManager(database_url=url)
            await manager.initialize()
            try:
                return await manager.health_check()
            finally:
                await manager.close()

        # Two separate loops, one after the other, as two processes would.
        assert asyncio.run(boot())
        assert asyncio.run(boot())


class TestAPartialSchemaIsRefused:
    """A half-created database is not a pre-migrations database.

    Raised in review. Adoption used to stamp whenever *any* model table was
    present, so a database left behind by an interrupted bootstrap — say one
    with `documents` but no `document_chunks` — was stamped at the baseline.
    Alembic then skipped the baseline as already applied, the missing tables
    were never created, and the application started successfully and failed at
    request time. Which is precisely the defect migrations were added to stop.
    """

    @pytest.mark.asyncio
    async def test_it_refuses_rather_than_stamping(self, database):
        await _build_legacy_database(f"sqlite+aiosqlite:///{database}")

        # Simulate the bootstrap having died partway through.
        connection = sqlite3.connect(database)
        connection.execute("DROP TABLE document_chunks")
        connection.commit()
        connection.close()
        assert "document_chunks" not in _tables(database)

        manager = DatabaseManager()
        with pytest.raises(RuntimeError, match="partially created schema"):
            await manager.initialize()
        await manager.close()

    @pytest.mark.asyncio
    async def test_the_refusal_names_both_sides(self, database):
        """What is missing and what is there — enough to decide what to do."""
        await _build_legacy_database(f"sqlite+aiosqlite:///{database}")
        connection = sqlite3.connect(database)
        connection.execute("DROP TABLE document_chunks")
        connection.commit()
        connection.close()

        manager = DatabaseManager()
        with pytest.raises(RuntimeError) as excinfo:
            await manager.initialize()
        await manager.close()

        message = str(excinfo.value)
        assert "document_chunks" in message, "name what is missing"
        assert "documents" in message, "and what is present"

    @pytest.mark.asyncio
    async def test_a_partial_schema_is_not_silently_stamped(self, database):
        """The specific failure: stamped, so the baseline never runs."""
        await _build_legacy_database(f"sqlite+aiosqlite:///{database}")
        connection = sqlite3.connect(database)
        connection.execute("DROP TABLE chat_feedback")
        connection.commit()
        connection.close()

        manager = DatabaseManager()
        with pytest.raises(RuntimeError):
            await manager.initialize()
        await manager.close()

        assert _revision(database) is None, "must not have been stamped"
        assert "chat_feedback" not in _tables(database)

    @pytest.mark.asyncio
    async def test_an_empty_database_is_still_migrated_normally(self, database):
        """The guard must not catch the ordinary fresh-install case."""
        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        assert _revision(database) == "0001_baseline"
        assert {"documents", "document_chunks", "chat_feedback"} <= _tables(database)


class TestAFailedMigrationIsNotMistakenForARace:
    """ "The tables exist" is not "the migration applied".

    Raised in review. Those were the same thing under `create_all`, where a
    present table *was* the success condition. With migrations they are not: a
    future revision adding a column can fail for a real reason while every
    table is still present, and treating that as another worker having won the
    race would start the application against a schema missing the column —
    failing at request time, which is the defect migrations exist to prevent.
    """

    @pytest.mark.asyncio
    async def test_a_failure_below_head_is_not_suppressed(self, database, monkeypatch):
        from sqlalchemy.exc import OperationalError

        # A database with every table, already at head.
        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()
        assert _revision(database) == "0001_baseline"

        # Now pretend a later revision exists that this database has not
        # reached, and that applying it fails.
        monkeypatch.setattr(
            DatabaseManager,
            "_is_at_head",
            lambda self: _false(),
        )

        async def failing_migration(self, url):
            raise OperationalError(
                "ALTER TABLE documents ADD COLUMN tags", {}, Exception()
            )

        monkeypatch.setattr(DatabaseManager, "_run_migrations", failing_migration)

        manager = DatabaseManager()
        with pytest.raises(OperationalError):
            await manager.initialize()
        await manager.close()

    @pytest.mark.asyncio
    async def test_a_failure_at_head_is_treated_as_a_race(self, database, monkeypatch):
        """The legitimate case still works: another worker got there first."""
        from sqlalchemy.exc import OperationalError

        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        async def failing_migration(self, url):
            raise OperationalError("table documents already exists", {}, Exception())

        monkeypatch.setattr(DatabaseManager, "_run_migrations", failing_migration)

        manager = DatabaseManager()
        # `_is_at_head` is real here and the database genuinely is at head.
        await manager.initialize()
        assert await manager.health_check()
        await manager.close()

    @pytest.mark.asyncio
    async def test_is_at_head_reports_false_for_an_unmigrated_database(self, database):
        manager = DatabaseManager()
        manager._database_url = f"sqlite+aiosqlite:///{database}"
        from sqlalchemy.ext.asyncio import create_async_engine

        manager._engine = create_async_engine(manager._database_url)
        try:
            assert await manager._is_at_head() is False
        finally:
            await manager._engine.dispose()

    @pytest.mark.asyncio
    async def test_is_at_head_reports_true_once_migrated(self, database):
        manager = DatabaseManager()
        await manager.initialize()
        try:
            assert await manager._is_at_head() is True
        finally:
            await manager.close()


async def _false() -> bool:
    return False


class TestAutoMigrateCanBeTurnedOff:
    """Production applies migrations as its own step, not on whichever replica
    boots first. ``DB_AUTO_MIGRATE=false`` is how a deployment says so.
    """

    @pytest.mark.asyncio
    async def test_it_starts_against_an_already_migrated_database(
        self, database, monkeypatch
    ):
        manager = DatabaseManager()
        await manager.initialize()
        await manager.close()

        monkeypatch.setenv("DB_AUTO_MIGRATE", "false")
        reload_settings()

        manager = DatabaseManager()
        await manager.initialize()
        assert await manager.health_check()
        await manager.close()

    @pytest.mark.asyncio
    async def test_it_refuses_to_start_against_an_unmigrated_database(
        self, database, monkeypatch
    ):
        """Silently coming up would fail later, inside a request, with a worse
        message than the one naming the missing tables.
        """
        monkeypatch.setenv("DB_AUTO_MIGRATE", "false")
        reload_settings()

        manager = DatabaseManager()
        with pytest.raises(RuntimeError, match="DB_AUTO_MIGRATE is off"):
            await manager.initialize()
        await manager.close()

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_missing_tables(self, database, monkeypatch):
        monkeypatch.setenv("DB_AUTO_MIGRATE", "false")
        reload_settings()

        manager = DatabaseManager()
        with pytest.raises(RuntimeError) as excinfo:
            await manager.initialize()
        await manager.close()

        message = str(excinfo.value)
        assert "documents" in message
        assert "alembic upgrade head" in message, "say what to do about it"


class TestModelsAndMigrationsDoNotDrift:
    def test_the_migrations_produce_exactly_what_the_models_describe(self, tmp_path):
        """``alembic check`` in CI is what stops a model change merging alone.

        Run here as well so the failure arrives in the test suite — where a
        contributor is already looking — rather than only in CI.
        """
        import os
        import subprocess
        import sys

        database = tmp_path / "drift.db"
        environment = {
            **os.environ,
            "DATABASE_URL": f"sqlite+aiosqlite:///{database}",
            "ENVIRONMENT": "test",
            "SECRET_KEY": "drift-check-secret-key-abcdefghijklmnop",
        }
        root = Path(__file__).resolve().parents[2]

        upgrade = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert upgrade.returncode == 0, upgrade.stderr

        check = subprocess.run(
            [sys.executable, "-m", "alembic", "check"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, (
            "models and migrations have drifted. Run "
            "`alembic revision --autogenerate -m '<what changed>'`.\n"
            f"{check.stdout}\n{check.stderr}"
        )

    def test_every_revision_can_be_reversed(self, tmp_path):
        """A migration without a working downgrade is a one-way door."""
        import os
        import subprocess
        import sys

        database = tmp_path / "reverse.db"
        environment = {
            **os.environ,
            "DATABASE_URL": f"sqlite+aiosqlite:///{database}",
            "ENVIRONMENT": "test",
            "SECRET_KEY": "reverse-check-secret-key-abcdefghijklmnop",
        }
        root = Path(__file__).resolve().parents[2]

        for arguments in (
            ["upgrade", "head"],
            ["downgrade", "base"],
            ["upgrade", "head"],
        ):
            result = subprocess.run(
                [sys.executable, "-m", "alembic", *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"alembic {' '.join(arguments)}: {result.stderr}"
            )

        assert _revision(database) is not None
