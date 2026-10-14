# Migrations

The schema used to be created with `Base.metadata.create_all`. That creates
missing *tables* and nothing else, so a new column reached a developer's fresh
database, passed CI — which starts from an empty SQLite file every run — and
then did nothing at all on any database that already existed, where it failed
at query time instead.

Alembic replaces that. `backend/core/database.py` applies pending migrations
during startup when `DB_AUTO_MIGRATE` is true (the default).

## Adding a migration

```bash
# 1. Change the models in backend/core/models.py
# 2. Generate the revision
alembic revision --autogenerate -m "add tags to documents"
# 3. Read what it produced. Autogenerate is a first draft, not an answer.
# 4. Apply it
alembic upgrade head
```

Always read the generated file. Autogenerate does not detect table or column
renames (it writes a drop plus an add, which loses the data), and it cannot
know what a backfill should contain.

## Reviewing before applying

```bash
alembic upgrade head --sql          # print the SQL instead of running it
alembic current                     # which revision this database is at
alembic history --verbose           # what exists
alembic downgrade -1                # step back one revision
```

## Why `render_as_batch` is on

SQLite cannot `ALTER` a column in place. Batch mode rewrites the table instead,
which is the only way a column change reaches a SQLite database — and SQLite is
the default here. It is a no-op on PostgreSQL.

## Existing databases

A database created by the old `create_all` path has the tables but no
`alembic_version` row. Startup detects exactly that case and stamps it at the
baseline revision rather than trying to create tables that are already there.
Nothing needs to be done by hand.

## Production

Set `DB_AUTO_MIGRATE=false` and run migrations as their own deploy step:

```bash
alembic upgrade head
```

Then a schema change is a visible, ordered action with a log and an exit code,
rather than something applied by whichever replica boots first.

## Drift

CI runs `alembic check`, which fails when the models describe a schema the
migrations do not produce. That is the check that stops a model change from
being merged without the migration that delivers it.
