from datetime import date

import pytest

from app.modules.finance.tools.categories import find_category_by_name, seed_default_categories
from app.modules.finance.tools.queries import (
    search_total,
    search_transactions,
    spend_by_category,
    total_income,
    total_spend,
)
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk


async def _seed_two_transactions(db_session, user_pk):
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
            {
                "amount_vnd": 30_000,
                "raw_amount_text": "30k",
                "description": "Xe ôm",
                "log_type": "expense",
                "suggested_category": "Đi lại",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_total_spend_sums_expenses_in_range(db_session):
    user_pk = await get_or_create_user_pk(db_session, "20")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    total = await total_spend(
        db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31), category_id=None
    )
    assert total == 75_000


@pytest.mark.asyncio
async def test_total_spend_filters_by_category(db_session):
    user_pk = await get_or_create_user_pk(db_session, "21")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)
    an_uong = await find_category_by_name(db_session, user_pk, "Ăn uống")

    total = await total_spend(
        db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31), category_id=an_uong.id
    )
    assert total == 45_000


@pytest.mark.asyncio
async def test_spend_by_category_breaks_down_and_sorts(db_session):
    user_pk = await get_or_create_user_pk(db_session, "22")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    breakdown = await spend_by_category(db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31))
    assert breakdown[0] == {"category": "Ăn uống", "total": 45_000}
    assert breakdown[1] == {"category": "Đi lại", "total": 30_000}


@pytest.mark.asyncio
async def test_search_transactions_matches_description(db_session):
    user_pk = await get_or_create_user_pk(db_session, "23")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    results = await search_transactions(
        db_session, user_pk, "trà", start=date(2026, 8, 1), end=date(2026, 8, 31)
    )
    assert len(results) == 1
    assert results[0]["description"] == "Trà sữa"


@pytest.mark.asyncio
async def test_search_transactions_filters_by_period(db_session):
    """Finding 4: search_transactions must respect the period, not return
    all-time matches regardless of the extracted query range."""
    user_pk = await get_or_create_user_pk(db_session, "24")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa tháng 8",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
            {
                "amount_vnd": 50_000,
                "raw_amount_text": "50k",
                "description": "Trà sữa tháng 7",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-07-10",
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()

    results = await search_transactions(
        db_session, user_pk, "trà", start=date(2026, 8, 1), end=date(2026, 8, 31)
    )
    assert len(results) == 1
    assert results[0]["description"] == "Trà sữa tháng 8"


@pytest.mark.asyncio
async def test_search_total_is_not_capped_by_limit(db_session):
    """Finding 4: search_total must sum ALL matches in range, independent of
    any row limit applied to search_transactions."""
    user_pk = await get_or_create_user_pk(db_session, "25")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 10_000 * (i + 1),
                "raw_amount_text": f"{10 * (i + 1)}k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            }
            for i in range(3)
        ],
    )
    await db_session.commit()

    # 10_000 + 20_000 + 30_000 = 60_000, the true sum of all 3 matches.
    total = await search_total(db_session, user_pk, "trà", start=date(2026, 8, 1), end=date(2026, 8, 31))
    assert total == 60_000

    limited = await search_transactions(
        db_session, user_pk, "trà", start=date(2026, 8, 1), end=date(2026, 8, 31), limit=1
    )
    assert len(limited) == 1


def _txn(amount, description, category="Ăn uống", log_type="expense", occurred_at="2026-08-03"):
    return {
        "amount_vnd": amount,
        "raw_amount_text": f"{amount // 1000}k",
        "description": description,
        "log_type": log_type,
        "suggested_category": category,
        "occurred_at": occurred_at,
        "amount_adjusted": False,
    }


@pytest.mark.asyncio
async def test_total_income_sums_income_only(db_session):
    user_pk = await get_or_create_user_pk(db_session, "26")
    await create_transactions(
        db_session,
        user_pk,
        [
            _txn(10_000_000, "Lương tháng 8", category="Khác", log_type="income"),
            _txn(45_000, "Trà sữa"),
        ],
    )
    await db_session.commit()

    total = await total_income(db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31))
    assert total == 10_000_000


@pytest.mark.asyncio
async def test_spend_by_category_groups_uncategorized_as_khac(db_session):
    """A log whose category was deleted (category_id SET NULL) must still be
    counted, under "Khác", so the breakdown sums to the same as total_spend."""
    user_pk = await get_or_create_user_pk(db_session, "27")
    logs = await create_transactions(
        db_session, user_pk, [_txn(45_000, "Trà sữa"), _txn(20_000, "Linh tinh", category="Khác")]
    )
    logs[0].category_id = None
    await db_session.commit()

    breakdown = await spend_by_category(db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31))
    assert breakdown == [{"category": "Khác", "total": 65_000}]


@pytest.mark.asyncio
async def test_search_treats_like_wildcards_literally(db_session):
    user_pk = await get_or_create_user_pk(db_session, "28")
    await create_transactions(
        db_session, user_pk, [_txn(45_000, "Trà sữa"), _txn(90_000, "Áo sale 50%"), _txn(30_000, "a_b")]
    )
    await db_session.commit()

    start, end = date(2026, 8, 1), date(2026, 8, 31)
    percent = await search_transactions(db_session, user_pk, "%", start, end)
    assert [r["description"] for r in percent] == ["Áo sale 50%"]
    assert await search_total(db_session, user_pk, "%", start, end) == 90_000

    underscore = await search_transactions(db_session, user_pk, "_", start, end)
    assert [r["description"] for r in underscore] == ["a_b"]
