from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceBudget, FinanceLog

_WARN_RATIO = 0.8


async def get_category_budget(
    session: AsyncSession, user_pk: int, category_id: int
) -> FinanceBudget | None:
    """Category-specific budget only — no fallback to the overall budget."""
    return (
        await session.execute(
            select(FinanceBudget).where(
                FinanceBudget.user_id == user_pk, FinanceBudget.category_id == category_id
            )
        )
    ).scalar_one_or_none()


async def get_overall_budget(session: AsyncSession, user_pk: int) -> FinanceBudget | None:
    return (
        await session.execute(
            select(FinanceBudget).where(
                FinanceBudget.user_id == user_pk, FinanceBudget.category_id.is_(None)
            )
        )
    ).scalar_one_or_none()


async def total_month_to_date_spend(session: AsyncSession, user_pk: int, today: date) -> int:
    """Like month_to_date_spend but across every category (for the overall budget check)."""
    month_start = today.replace(day=1)
    total = (
        await session.execute(
            select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(
                FinanceLog.user_id == user_pk,
                FinanceLog.log_type == "expense",
                FinanceLog.occurred_at >= month_start,
                FinanceLog.occurred_at <= today,
            )
        )
    ).scalar_one()
    return int(total)


async def month_to_date_spend(
    session: AsyncSession, user_pk: int, category_id: int, today: date
) -> int:
    month_start = today.replace(day=1)
    total = (
        await session.execute(
            select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(
                FinanceLog.user_id == user_pk,
                FinanceLog.category_id == category_id,
                FinanceLog.log_type == "expense",
                FinanceLog.occurred_at >= month_start,
                FinanceLog.occurred_at <= today,
            )
        )
    ).scalar_one()
    return int(total)


def compute_budget_status(spent: int, budget_amount: int) -> dict:
    ratio = spent / budget_amount if budget_amount else 0.0
    if ratio > 1.0:
        level = "over"
    elif ratio >= _WARN_RATIO:
        level = "warn"
    else:
        level = "ok"
    return {"ratio": ratio, "level": level}
