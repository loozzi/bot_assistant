from app.utils.logger import get_logger

from ...tools.categories import list_categories
from ...tools.transactions import create_transactions

logger = get_logger(__name__)


async def log_persist_node(state: dict) -> dict:
    """Insert one FinanceLog row per parsed transaction.

    Expects state["session"] (an open AsyncSession) and state["user_pk"] —
    supplied by the caller (agent.py's _log_persist_and_followups_node in
    production; test wrappers directly). Session lifecycle deliberately
    lives outside this function so budget_check/memory_write can reuse the
    same open session for the rest of the log branch's unit of work.
    """
    transactions = state.get("parsed_transactions", [])
    if not transactions:
        return {}

    session = state["session"]
    user_pk = state["user_pk"]

    logs = await create_transactions(session, user_pk, transactions)

    category_names = {c.id: c.name for c in await list_categories(session, user_pk)}
    persisted_categories = [category_names.get(log.category_id, "Khác") for log in logs]

    return {
        "persisted_log_ids": [log.id for log in logs],
        "persisted_categories": persisted_categories,
    }
