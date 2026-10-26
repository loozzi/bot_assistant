from datetime import date

from ...tools.transactions import category_name_of, find_logs

_NOT_FOUND = "Mình không tìm thấy giao dịch nào khớp, bạn mô tả rõ hơn (tên, ngày, số tiền) nhé."


async def modify_find_node(state: dict) -> dict:
    """Look up saved logs matching the extracted target. Requires
    state["session"] and state["user_pk"] (injected by agent.py)."""
    session = state["session"]
    target = state.get("modify_target") or {}
    logs = await find_logs(
        session,
        state["user_pk"],
        keyword=(target.get("keyword") or "").strip() or None,
        occurred_at=date.fromisoformat(target["occurred_at"]) if target.get("occurred_at") else None,
        amount_vnd=target.get("amount_vnd"),
    )
    if not logs:
        return {"modify_candidates": [], "reply": _NOT_FOUND}

    # No filters at all means "the latest one" — don't make the user pick.
    if not any(target.get(key) for key in ("keyword", "occurred_at", "amount_vnd")):
        logs = logs[:1]

    candidates = [
        {
            "id": log.id,
            "description": log.description,
            "amount_vnd": log.amount,
            "occurred_at": log.occurred_at.isoformat(),
            "category": await category_name_of(session, log),
            "log_type": log.log_type,
        }
        for log in logs
    ]
    return {"modify_candidates": candidates}
