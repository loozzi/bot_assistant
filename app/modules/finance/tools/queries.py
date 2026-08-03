from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory, FinanceLog


async def total_spend(
    session: AsyncSession, user_pk: int, start: date, end: date, category_id: int | None
) -> int:
    conditions = [
        FinanceLog.user_id == user_pk,
        FinanceLog.log_type == "expense",
        FinanceLog.occurred_at >= start,
        FinanceLog.occurred_at <= end,
    ]
    if category_id is not None:
        conditions.append(FinanceLog.category_id == category_id)

    total = (
        await session.execute(select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(*conditions))
    ).scalar_one()
    return int(total)


async def spend_by_category(
    session: AsyncSession, user_pk: int, start: date, end: date
) -> list[dict]:
    rows = await session.execute(
        select(FinanceCategory.name, func.coalesce(func.sum(FinanceLog.amount), 0))
        .join(FinanceLog, FinanceLog.category_id == FinanceCategory.id)
        .where(
            FinanceLog.user_id == user_pk,
            FinanceLog.log_type == "expense",
            FinanceLog.occurred_at >= start,
            FinanceLog.occurred_at <= end,
        )
        .group_by(FinanceCategory.name)
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
            FinanceLog.description.ilike(f"%{keyword}%"),
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
                FinanceLog.description.ilike(f"%{keyword}%"),
            )
        )
    ).scalar_one()
    return int(total)
