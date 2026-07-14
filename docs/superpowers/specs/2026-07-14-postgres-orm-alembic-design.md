# Postgres ORM + Alembic Setup

## Problem

`app/models.py` is empty and untracked, no SQLAlchemy declarative `Base` exists
anywhere in the codebase, and there is no Alembic migration setup. The async
connection pool itself (`app/infra/db/session.py`) is already implemented and
already wired into the bot lifecycle (`app/main.py: on_startup`/`on_shutdown`),
sourcing all tuning from `Settings` (`postgres_pool_size`, `postgres_max_overflow`,
`postgres_pool_timeout`, `postgres_pool_recycle`) — that part needs no changes.

This closes the "No Postgres models / migrations" gap called out in `CLAUDE.md`.

## Scope

In scope:
- A SQLAlchemy 2.x declarative `Base` and one minimal `User` model in
  `app/models.py`, enough to prove the pool and Alembic migrations work
  end-to-end.
- Alembic wired up to autogenerate against that `Base.metadata`, using a
  dedicated **sync** driver for migrations only (app runtime stays asyncpg).
- One initial migration creating the `users` table.
- Updating the now-stale "No Postgres models / migrations" bullet in
  `CLAUDE.md`.

Out of scope (deferred to when each module's own design happens):
- Per-module tables (journal entries, finance transactions, todos, insight
  summaries).
- Any repository/DAO layer on top of the models.

## Design

### `app/models.py`

```python
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

`BigInteger` because Telegram user IDs exceed 32-bit range.

### Settings

Add `Settings.postgres_sync_dsn` next to the existing `postgres_dsn` property,
same fields, `postgresql+psycopg2://` scheme instead of `postgresql+asyncpg://`.
Alembic is the only consumer of this — the app's runtime engine
(`app/infra/db/session.py`) keeps using `postgres_dsn` (asyncpg) unchanged.

### Dependencies

Add to `pyproject.toml`:
- `alembic>=1.13`
- `psycopg2-binary>=2.9` (sync driver, migration-only)

### Alembic layout

Root-level `alembic/` + `alembic.ini`, matching where `pyproject.toml` and
`main.py` already live.

`alembic/env.py`:
- Imports `Base` from `app.models` and sets `target_metadata = Base.metadata`.
- Builds the engine from `get_settings().postgres_sync_dsn` (ignores the
  `sqlalchemy.url` placeholder in `alembic.ini`) so migrations always use the
  same config source as the app (`app/config/settings.py`), never a
  hand-maintained duplicate URL.
- Uses the standard sync `run_migrations_online`/`offline` pair — no asyncio
  wrapper needed since this is the dedicated sync path.

One initial migration, generated via `alembic revision --autogenerate -m
"create users table"`, checked in under `alembic/versions/`.

### CLAUDE.md update

Replace the "No Postgres models / migrations" gap bullet to reflect that
`Base`/`User` and Alembic now exist, while noting per-module tables are still
unmodeled — keeping the "gaps" section honest for future sessions.

## Testing

No test suite exists in this repo yet (per `CLAUDE.md`). Verification is
manual: run `alembic upgrade head` against a local/dockerized Postgres and
confirm the `users` table is created; confirm `uv run python main.py` still
boots (pool init unaffected).
