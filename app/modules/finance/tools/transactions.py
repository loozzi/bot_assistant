from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceLog

from .categories import find_category_by_name, seed_default_categories


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
