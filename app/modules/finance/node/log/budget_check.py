from collections import defaultdict
from datetime import date

from ...tools.budgets import (
    compute_budget_status,
    get_category_budget,
    get_overall_budget,
    month_to_date_spend,
    total_month_to_date_spend,
)
from ...tools.categories import find_category_by_name
from ...tools.dates import today_ict


async def log_budget_check_node(state: dict) -> dict:
    """Warn when the just-logged expenses push a category budget, or the
    overall monthly budget, into a higher level (ok -> warn -> over).

    Only a level *crossing* caused by this batch warns — spend that keeps a
    budget at the level it was already at stays silent, so the user isn't
    nagged on every transaction once they're past 80%. Income never counts.

    Requires state["session"] (an open AsyncSession) and state["user_pk"] —
    both set by log_persist_node before this node runs — plus
    state["parsed_transactions"] aligned with state["persisted_categories"].
    No-ops (empty warnings) for anything without a budget configured.
    """
    session = state["session"]
    user_pk = state["user_pk"]
    today = today_ict()

    # Amount this batch added to month-to-date spend, per category name.
    batch_spend: dict[str, int] = defaultdict(int)
    for txn, name in zip(state.get("parsed_transactions", []), state.get("persisted_categories", [])):
        if txn["log_type"] == "expense" and _counts_this_month(txn["occurred_at"], today):
            batch_spend[name] += txn["amount_vnd"]

    warnings: list[str] = []
    if not batch_spend:
        return {"budget_warnings": warnings}

    for name, added in batch_spend.items():
        category = await find_category_by_name(session, user_pk, name)
        if category is None:
            continue

        budget = await get_category_budget(session, user_pk, category.id)
        if budget is None:
            continue

        spent = await month_to_date_spend(session, user_pk, category.id, today)
        warnings.extend(_warning_if_crossed(name, spent - added, spent, budget.amount))

    overall_budget = await get_overall_budget(session, user_pk)
    if overall_budget is not None:
        overall_spent = await total_month_to_date_spend(session, user_pk, today)
        overall_before = overall_spent - sum(batch_spend.values())
        warnings.extend(
            _warning_if_crossed("tổng chi tiêu", overall_before, overall_spent, overall_budget.amount)
        )

    return {"budget_warnings": warnings}


def _counts_this_month(occurred_at: str, today: date) -> bool:
    """Mirror the month-to-date window the spend queries use."""
    return today.replace(day=1) <= date.fromisoformat(occurred_at) <= today


_LEVEL_RANK = {"ok": 0, "warn": 1, "over": 2}


def _warning_if_crossed(label: str, spent_before: int, spent: int, budget_amount: int) -> list[str]:
    before = compute_budget_status(spent_before, budget_amount)
    after = compute_budget_status(spent, budget_amount)
    if _LEVEL_RANK[after["level"]] <= _LEVEL_RANK[before["level"]]:
        return []
    return _render_warning(label, spent, budget_amount, after)


def _render_warning(label: str, spent: int, budget_amount: int, status: dict) -> list[str]:
    formatted = f"({spent:,}đ / {budget_amount:,}đ)".replace(",", ".")
    if status["level"] == "over":
        return [f"⚠️ Bạn đã vượt ngân sách {label} tháng này {formatted}"]
    if status["level"] == "warn":
        pct = int(status["ratio"] * 100)
        return [f"⚠️ Bạn đã dùng {pct}% ngân sách {label} tháng này {formatted}"]
    return []
