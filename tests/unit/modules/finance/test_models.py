import pytest
from datetime import date
from sqlalchemy import select

from app.models import FinanceBudget, FinanceCategory, FinanceDebt, FinanceGoal, FinanceLog, User


@pytest.mark.asyncio
async def test_models_import_and_create_tables(db_session):
    # If app.models imports and db_session's create_all succeeded, the schema is valid.
    result = await db_session.execute(select(FinanceLog))
    assert result.all() == []


@pytest.mark.asyncio
async def test_finance_log_round_trip(db_session):
    user = User(telegram_id=123456789)
    db_session.add(user)
    await db_session.flush()

    category = FinanceCategory(
        user_id=user.id, name="Ăn uống", description="", group="essential", is_seeded=True
    )
    db_session.add(category)
    await db_session.flush()

    log = FinanceLog(
        user_id=user.id,
        amount=45_000,
        description="Trà sữa",
        message="mua trà sữa 45k",
        log_type="expense",
        category_id=category.id,
        occurred_at=date(2026, 8, 3),
        extra={"raw_amount_text": "45k"},
    )
    db_session.add(log)
    await db_session.commit()

    fetched = (await db_session.execute(select(FinanceLog))).scalar_one()
    assert fetched.amount == 45_000
    assert fetched.extra == {"raw_amount_text": "45k"}
    assert fetched.category_id == category.id


@pytest.mark.asyncio
async def test_finance_category_unique_per_user_not_globally(db_session):
    user_a = User(telegram_id=1)
    user_b = User(telegram_id=2)
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    db_session.add(FinanceCategory(user_id=user_a.id, name="Khác", description="", group="other"))
    db_session.add(FinanceCategory(user_id=user_b.id, name="Khác", description="", group="other"))
    await db_session.commit()  # must not raise — same name, different users

    result = await db_session.execute(select(FinanceCategory))
    assert len(result.all()) == 2


@pytest.mark.asyncio
async def test_new_tables_exist(db_session):
    user = User(telegram_id=999)
    db_session.add(user)
    await db_session.flush()

    db_session.add(FinanceBudget(user_id=user.id, category_id=None, amount=3_000_000))
    db_session.add(FinanceGoal(user_id=user.id, description="Mua xe", status="active"))
    db_session.add(
        FinanceDebt(
            user_id=user.id,
            counterparty="Nam",
            direction="owed_to_me",
            amount_total=150_000,
            amount_outstanding=150_000,
            description="Lẩu",
            status="open",
        )
    )
    await db_session.commit()
