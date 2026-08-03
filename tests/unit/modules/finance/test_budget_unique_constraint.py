"""Finding 9: Postgres must reject a second 'overall' (category_id IS NULL)
budget row for the same user. That constraint is a partial unique INDEX
(postgresql_where="category_id IS NULL"), which SQLite — used by every other
finance test via the db_session fixture — can't express. So this one test
talks directly to the live docker-compose Postgres instance instead of the
SQLite db_session fixture.
"""

import asyncio

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings
from app.models import FinanceBudget, User
from app.modules.finance.tools.users import get_or_create_user_pk

# Unlikely-to-collide telegram_id so repeated runs don't stumble over
# leftover rows from a previous run that failed before cleanup.
_TEST_TELEGRAM_ID = 900_000_000_001


def _postgres_reachable() -> bool:
    """Best-effort connectivity probe so this test can be skipped (rather
    than erroring) on any machine/CI where `docker compose up -d postgres`
    hasn't been run — every other finance test uses the in-memory SQLite
    db_session fixture; this one legitimately needs live Postgres (see
    module docstring) but must degrade gracefully when it's absent."""

    async def _check() -> bool:
        try:
            engine = create_async_engine(get_settings().postgres_dsn)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return True
        except Exception:
            return False

    return asyncio.run(_check())


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="requires live Postgres (docker compose up -d postgres)",
)


@pytest.mark.asyncio
async def test_second_overall_budget_for_same_user_raises_integrity_error():
    settings = get_settings()
    engine = create_async_engine(settings.postgres_dsn)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_factory() as session:
            user_pk = await get_or_create_user_pk(session, str(_TEST_TELEGRAM_ID))
            await session.commit()

            session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=1_000_000))
            await session.commit()

            session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=2_000_000))
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        # Clean up so repeated test runs start from a clean slate.
        async with session_factory() as session:
            existing = await session.execute(select(User).where(User.telegram_id == _TEST_TELEGRAM_ID))
            user = existing.scalar_one_or_none()
            if user is not None:
                await session.execute(delete(FinanceBudget).where(FinanceBudget.user_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
        await engine.dispose()
