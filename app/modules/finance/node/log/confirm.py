from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..answers import is_affirmative


def _render_transaction_list(transactions: list[dict]) -> str:
    lines = []
    for i, txn in enumerate(transactions, 1):
        amount = f"{txn['amount_vnd']:,}".replace(",", ".")
        marker = " (đã điều chỉnh số tiền)" if txn["amount_adjusted"] else ""
        lines.append(f"{i}. {txn['description']} — {amount}đ — {txn['suggested_category']}{marker}")
    return "\n".join(lines)


async def log_confirm_node(state: dict) -> dict:
    """Ask the user to confirm the parsed transactions before writing them.

    If nothing was parsed, there's nothing to confirm — route straight to
    the "edit" (ask-to-restate) path without asking a question.

    Besides tapping a button, the user may type a correction straight away
    (e.g. "món 2 là 50k") — that free text comes back as the answer and the
    graph routes it to log_apply_edit. Any notice from the previous edit
    round (state["edit_notice"]) is shown above the list.
    """
    transactions = state.get("parsed_transactions", [])
    if not transactions:
        return {"confirm_answer": "edit"}

    notice = "\n".join(state.get("edit_notice") or [])
    question = (
        (f"{notice}\n\n" if notice else "")
        + "Xác nhận các giao dịch sau nhé:\n\n"
        + _render_transaction_list(transactions)
        + "\n\nCó thể gõ trực tiếp chỗ cần sửa, ví dụ \"món 2 là 50k\"."
    )
    answer = interrupt(
        HumanReviewRequest(
            question=question,
            options=[
                {"label": "Xác nhận", "value": "confirm"},
                {"label": "Sửa", "value": "edit"},
            ],
        )
    )
    if is_affirmative(answer):
        return {"confirm_answer": "confirm"}
    if answer == "edit":
        return {"confirm_answer": answer}
    return {"confirm_answer": answer, "edit_instruction": answer}
