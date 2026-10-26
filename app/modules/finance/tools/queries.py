from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory, FinanceLog

_UNCATEGORIZED = "Khác"


def _keyword_filter(keyword: str):
    """Case-insensitive substring match that treats LIKE wildcards literally."""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return FinanceLog.description.ilike(f"%{escaped}%", escape="\\")


async def _sum_logs(
    session: AsyncSession,
    user_pk: int,
    log_type: str,
    start: date,
    end: date,
    category_id: int | None = None,
) -> int:
    conditions = [
        FinanceLog.user_id == user_pk,
        FinanceLog.log_type == log_type,
        FinanceLog.occurred_at >= start,
        FinanceLog.occurred_at <= end,
    ]
    if category_id is not None:
        conditions.append(FinanceLog.category_id == category_id)

    total = (
        await session.execute(select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(*conditions))
    ).scalar_one()
    return int(total)


async def total_spend(
    session: AsyncSession, user_pk: int, start: date, end: date, category_id: int | None
) -> int:
    return await _sum_logs(session, user_pk, "expense", start, end, category_id)


async def total_income(
    session: AsyncSession, user_pk: int, start: date, end: date, category_id: int | None = None
) -> int:
    return await _sum_logs(session, user_pk, "income", start, end, category_id)


async def spend_by_category(
    session: AsyncSession, user_pk: int, start: date, end: date
) -> list[dict]:
    """Expense breakdown by category. Logs whose category was deleted
    (category_id SET NULL) are grouped under "Khác" so the rows always sum
    to total_spend for the same period."""
    category_name = func.coalesce(FinanceCategory.name, _UNCATEGORIZED)
    rows = await session.execute(
        select(category_name, func.sum(FinanceLog.amount))
        .select_from(FinanceLog)
        .outerjoin(FinanceCategory, FinanceLog.category_id == FinanceCategory.id)
        .where(
            FinanceLog.user_id == user_pk,
            FinanceLog.log_type == "expense",
            FinanceLog.occurred_at >= start,
            FinanceLog.occurred_at <= end,
        )
        .group_by(category_name)
        .order_by(func.sum(FinanceLog.amount).desc())
    )
    return [{"category": name, "total": int(total)} for name, total in rows.all()]


async def search_transactions(
    session: AsyncSession, user_pk: int, keyword: str, start: date, end: date, limit: int = 10
) -> list[dict]:
    rows = await session.execute(
        select(FinanceLog)
        .where(
            FinanceLog.user_id == user_pk,
            FinanceLog.log_type == "expense",
            FinanceLog.occurred_at >= start,
            FinanceLog.occurred_at <= end,
            _keyword_filter(keyword),
        )
        .order_by(FinanceLog.occurred_at.desc())
        .limit(limit)
    )
    return [
        {
            "description": log.description,
            "amount": log.amount,
            "occurred_at": log.occurred_at.isoformat(),
        }
        for log in rows.scalars().all()
    ]


async def search_total(session: AsyncSession, user_pk: int, keyword: str, start: date, end: date) -> int:
    """Unlimited sum for the same filter search_transactions uses, so the displayed total isn't truncated by the row limit."""
    total = (
        await session.execute(
            select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(
                FinanceLog.user_id == user_pk,
                FinanceLog.log_type == "expense",
                FinanceLog.occurred_at >= start,
                FinanceLog.occurred_at <= end,
                _keyword_filter(keyword),
            )
        )
    ).scalar_one()
    return int(total)
