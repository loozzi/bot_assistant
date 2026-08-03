import pytest

from app.models import FinanceBudget
from app.modules.finance.node.log.budget_check import log_budget_check_node
from app.modules.finance.tools.budgets import (
    compute_budget_status,
    get_category_budget,
    get_overall_budget,
    month_to_date_spend,
)
from app.modules.finance.tools.categories import find_category_by_name, seed_default_categories
from app.modules.finance.tools.dates import today_ict
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk


def test_compute_budget_status_levels():
    assert compute_budget_status(500_000, 1_000_000)["level"] == "ok"
    assert compute_budget_status(850_000, 1_000_000)["level"] == "warn"
    assert compute_budget_status(1_200_000, 1_000_000)["level"] == "over"


@pytest.mark.asyncio
async def test_get_category_budget_ignores_overall_budget(db_session):
    """Finding 5: get_category_budget must NOT fall back to the overall
    budget — it's now a strictly category-scoped lookup."""
    user_pk = await get_or_create_user_pk(db_session, "10")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=5_000_000))
    db_session.add(FinanceBudget(user_id=user_pk, category_id=category.id, amount=1_000_000))
    await db_session.commit()

    budget = await get_category_budget(db_session, user_pk, category.id)
    assert budget.amount == 1_000_000


@pytest.mark.asyncio
async def test_get_category_budget_none_when_unset(db_session):
    """No category-specific budget exists, and there IS an overall budget —
    get_category_budget must return None, not fall back to the overall one."""
    user_pk = await get_or_create_user_pk(db_session, "11")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=5_000_000))
    await db_session.commit()

    budget = await get_category_budget(db_session, user_pk, category.id)
    assert budget is None


@pytest.mark.asyncio
async def test_get_overall_budget_returns_null_category_row(db_session):
    user_pk = await get_or_create_user_pk(db_session, "14")
    await seed_default_categories(db_session, user_pk)

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=5_000_000))
    await db_session.commit()

    budget = await get_overall_budget(db_session, user_pk)
    assert budget.amount == 5_000_000


@pytest.mark.asyncio
async def test_budget_check_node_warns_near_threshold(db_session):
    user_pk = await get_or_create_user_pk(db_session, "12")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    db_session.add(FinanceBudget(user_id=user_pk, category_id=category.id, amount=1_000_000))
    await db_session.commit()

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": today_ict().isoformat(),
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {
            "user_pk": user_pk,
            "session": db_session,
            "persisted_categories": ["Ăn uống"],
        }
    )
    assert len(result["budget_warnings"]) == 1
    assert "Ăn uống" in result["budget_warnings"][0]


@pytest.mark.asyncio
async def test_budget_check_node_silent_without_budget(db_session):
    user_pk = await get_or_create_user_pk(db_session, "13")
    await seed_default_categories(db_session, user_pk)

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": today_ict().isoformat(),
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {"user_pk": user_pk, "session": db_session, "persisted_categories": ["Ăn uống"]}
    )
    assert result["budget_warnings"] == []


@pytest.mark.asyncio
async def test_budget_check_overall_only_labels_as_tong_chi_tieu_with_overall_totals(db_session):
    """Finding 5: with ONLY an overall budget set (no category-specific
    budget), a warning must be labeled 'tổng chi tiêu' — never the specific
    category that happened to be logged — and the spent/budget figures shown
    must be the OVERALL totals, not the single category's spend."""
    user_pk = await get_or_create_user_pk(db_session, "15")
    await seed_default_categories(db_session, user_pk)

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=500_000))
    await db_session.commit()

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": today_ict().isoformat(),
                "amount_adjusted": False,
            },
            {
                "amount_vnd": 200_000,
                "raw_amount_text": "200k",
                "description": "Xe ôm tuần này",
                "log_type": "expense",
                "suggested_category": "Đi lại",
                "occurred_at": today_ict().isoformat(),
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {
            "user_pk": user_pk,
            "session": db_session,
            "persisted_categories": ["Ăn uống", "Đi lại"],
        }
    )

    assert len(result["budget_warnings"]) == 1
    warning = result["budget_warnings"][0]
    assert "tổng chi tiêu" in warning
    assert "Ăn uống" not in warning
    assert "Đi lại" not in warning
    # overall spend is 850k + 200k = 1.050.000đ against the 500.000đ overall budget —
    # not the 850.000đ category-only figure the old (buggy) code would have shown.
    assert "1.050.000đ" in warning
    assert "500.000đ" in warning


@pytest.mark.asyncio
async def test_budget_check_overall_and_unbudgeted_category_no_mislabeled_warning(db_session):
    """Finding 5: an overall budget plus spend in a category with no budget
    of its own must never produce a warning mislabeled with that category's
    name — the category loop must skip it (no budget = no per-category
    check), regardless of what the overall check does."""
    user_pk = await get_or_create_user_pk(db_session, "16")
    await seed_default_categories(db_session, user_pk)

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=10_000_000))
    await db_session.commit()

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": today_ict().isoformat(),
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {
            "user_pk": user_pk,
            "session": db_session,
            "persisted_categories": ["Ăn uống"],
        }
    )

    # Overall spend (850k) is well under the 10M overall budget, so no
    # warning at all — and critically, no warning naming "Ăn uống" even
    # though that's the category that was actually logged.
    for warning in result["budget_warnings"]:
        assert "Ăn uống" not in warning
    assert result["budget_warnings"] == []
