from datetime import date

from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..answers import is_affirmative


def render_log(log: dict) -> str:
    amount = f"{log['amount_vnd']:,}".replace(",", ".")
    day = date.fromisoformat(log["occurred_at"]).strftime("%d/%m")
    sign = "+" if log["log_type"] == "income" else ""
    return f"{log['description']} — {sign}{amount}đ — {log['category']} ({day})"


def apply_changes(log: dict, changes: dict) -> dict:
    """How the log will look after the update, for showing before saving."""
    return {**log, **{k: v for k, v in changes.items() if k != "raw_amount_text"}}


async def modify_select_node(state: dict) -> dict:
    """Several logs matched — ask which one. Accepts a button tap (the log
    id) or a typed list number."""
    candidates = state["modify_candidates"]
    lines = [f"{i}. {render_log(c)}" for i, c in enumerate(candidates, 1)]
    answer = interrupt(
        HumanReviewRequest(
            question="Có vài giao dịch khớp, bạn chọn giao dịch nào?\n\n" + "\n".join(lines),
            options=[{"label": str(i), "value": f"log:{c['id']}"} for i, c in enumerate(candidates, 1)],
        )
    )

    chosen = None
    text = str(answer).strip()
    if text.startswith("log:"):
        chosen = next((c for c in candidates if f"log:{c['id']}" == text), None)
    elif text.isdigit() and 1 <= int(text) <= len(candidates):
        chosen = candidates[int(text) - 1]

    if chosen is None:
        return {"reply": "Mình không rõ bạn chọn giao dịch nào, bạn thử lại nhé."}
    return {"modify_candidates": [chosen]}


async def modify_confirm_node(state: dict) -> dict:
    log = state["modify_candidates"][0]
    if state["modify_action"] == "delete":
        question = f"Xoá giao dịch này nhé?\n\n{render_log(log)}"
    else:
        after = apply_changes(log, state.get("modify_changes", {}))
        question = f"Sửa giao dịch này nhé?\n\n{render_log(log)}\n→ {render_log(after)}"

    answer = interrupt(
        HumanReviewRequest(question=question, options=[{"label": "Xác nhận", "value": "confirm"}])
    )
    if is_affirmative(answer):
        return {"modify_confirmed": True}
    return {"modify_confirmed": False, "reply": "Đã giữ nguyên giao dịch."}
