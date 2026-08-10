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
    """Warn when a just-logged category is near/over its own budget, and
    separately when overall monthly spend is near/over the overall budget.

    Requires state["session"] (an open AsyncSession) and state["user_pk"] —
    both set by log_persist_node before this node runs. No-ops (empty
    warnings) for anything without a budget configured.
    """
    session = state["session"]
    user_pk = state["user_pk"]
    category_names = set(state.get("persisted_categories", []))

    warnings: list[str] = []
    today = today_ict()

    for name in category_names:
        category = await find_category_by_name(session, user_pk, name)
        if category is None:
            continue

        budget = await get_category_budget(session, user_pk, category.id)
        if budget is None:
            continue

        spent = await month_to_date_spend(session, user_pk, category.id, today)
        status = compute_budget_status(spent, budget.amount)
        warnings.extend(_render_warning(name, spent, budget.amount, status))

    overall_budget = await get_overall_budget(session, user_pk)
    if overall_budget is not None:
        overall_spent = await total_month_to_date_spend(session, user_pk, today)
        status = compute_budget_status(overall_spent, overall_budget.amount)
        warnings.extend(_render_warning("tổng chi tiêu", overall_spent, overall_budget.amount, status))

    return {"budget_warnings": warnings}


def _render_warning(label: str, spent: int, budget_amount: int, status: dict) -> list[str]:
    formatted = f"({spent:,}đ / {budget_amount:,}đ)".replace(",", ".")
    if status["level"] == "over":
        return [f"⚠️ Bạn đã vượt ngân sách {label} tháng này {formatted}"]
    if status["level"] == "warn":
        pct = int(status["ratio"] * 100)
        return [f"⚠️ Bạn đã dùng {pct}% ngân sách {label} tháng này {formatted}"]
    return []
