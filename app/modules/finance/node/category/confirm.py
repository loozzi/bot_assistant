from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..answers import is_affirmative


def needs_confirmation(params: dict) -> bool:
    """Only the operations that move logs between categories ask first."""
    return params["action"] in ("merge", "delete")


async def category_confirm_node(state: dict) -> dict:
    params = state["category_params"]
    if params["action"] == "merge":
        question = (
            f"Gộp danh mục '{params['name']}' vào '{params['target_name']}' nhé? "
            f"Mọi giao dịch của '{params['name']}' sẽ chuyển sang '{params['target_name']}'."
        )
    else:
        question = (
            f"Xoá danh mục '{params['name']}' nhé? Các giao dịch của nó sẽ chuyển sang 'Khác'."
        )

    answer = interrupt(
        HumanReviewRequest(question=question, options=[{"label": "Xác nhận", "value": "confirm"}])
    )
    if is_affirmative(answer):
        return {}
    return {"reply": "Đã giữ nguyên danh mục."}
