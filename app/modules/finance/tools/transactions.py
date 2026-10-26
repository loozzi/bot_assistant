from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory, FinanceLog

from .categories import find_category_by_name, seed_default_categories
from .queries import keyword_filter


async def create_transactions(
    session: AsyncSession, user_pk: int, transactions: list[dict]
) -> list[FinanceLog]:
    """Insert one FinanceLog row per parsed transaction dict.

    Ensures the user's categories are seeded first, then matches each
    transaction's suggested_category by name — falling back to "Khác" for
    anything the LLM invented that doesn't match a real category.
    """
    await seed_default_categories(session, user_pk)

    khac = await find_category_by_name(session, user_pk, "Khác")

    logs: list[FinanceLog] = []
    for txn in transactions:
        category = await find_category_by_name(session, user_pk, txn["suggested_category"])
        if category is None:
            category = khac

        log = FinanceLog(
            user_id=user_pk,
            amount=txn["amount_vnd"],
            description=txn["description"],
            message=txn["description"],
            log_type=txn["log_type"],
            category_id=category.id if category else None,
            occurred_at=date.fromisoformat(txn["occurred_at"]),
            extra={"raw_amount_text": txn["raw_amount_text"], "amount_adjusted": txn["amount_adjusted"]},
        )
        session.add(log)
        logs.append(log)

    await session.flush()
    return logs


async def find_logs(
    session: AsyncSession,
    user_pk: int,
    *,
    keyword: str | None = None,
    occurred_at: date | None = None,
    amount_vnd: int | None = None,
    limit: int = 5,
) -> list[FinanceLog]:
    """The user's logs matching every given filter, newest-entered first.

    With no filters at all this is simply "the most recent `limit` logs" —
    which is what "xoá giao dịch vừa rồi" should resolve to."""
    conditions = [FinanceLog.user_id == user_pk]
    if keyword:
        conditions.append(keyword_filter(keyword))
    if occurred_at is not None:
        conditions.append(FinanceLog.occurred_at == occurred_at)
    if amount_vnd is not None:
        conditions.append(FinanceLog.amount == amount_vnd)

    rows = await session.execute(
        select(FinanceLog).where(*conditions).order_by(FinanceLog.id.desc()).limit(limit)
    )
    return list(rows.scalars().all())


async def get_log(session: AsyncSession, user_pk: int, log_id: int) -> FinanceLog | None:
    """Scoped to user_pk so a log id from another user can never be touched."""
    return (
        await session.execute(
            select(FinanceLog).where(FinanceLog.id == log_id, FinanceLog.user_id == user_pk)
        )
    ).scalar_one_or_none()


async def category_name_of(session: AsyncSession, log: FinanceLog) -> str:
    if log.category_id is None:
        return "Khác"
    category = await session.get(FinanceCategory, log.category_id)
    return category.name if category else "Khác"


async def update_log(session: AsyncSession, user_pk: int, log: FinanceLog, changes: dict) -> FinanceLog:
    """Apply already-validated changes (keys: amount_vnd, description,
    category, occurred_at, log_type — each optional) to one log."""
    if "amount_vnd" in changes:
        log.amount = changes["amount_vnd"]
        log.extra = {**(log.extra or {}), "raw_amount_text": changes.get("raw_amount_text")}
    if "description" in changes:
        log.description = changes["description"]
    if "occurred_at" in changes:
        log.occurred_at = date.fromisoformat(changes["occurred_at"])
    if "log_type" in changes:
        log.log_type = changes["log_type"]
    if "category" in changes:
        await seed_default_categories(session, user_pk)
        category = await find_category_by_name(session, user_pk, changes["category"])
        if category is None:
            category = await find_category_by_name(session, user_pk, "Khác")
        log.category_id = category.id if category else None
    await session.flush()
    return log


async def delete_log(session: AsyncSession, log: FinanceLog) -> None:
    await session.delete(log)
    await session.flush()
