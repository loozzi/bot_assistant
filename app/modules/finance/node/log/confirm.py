from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest


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
    """
    transactions = state.get("parsed_transactions", [])
    if not transactions:
        return {"confirm_answer": "edit"}

    question = "Xác nhận các giao dịch sau nhé:\n\n" + _render_transaction_list(transactions)
    answer = interrupt(
        HumanReviewRequest(
            question=question,
            options=[
                {"label": "Xác nhận", "value": "confirm"},
                {"label": "Sửa", "value": "edit"},
            ],
        )
    )
    return {"confirm_answer": answer}
