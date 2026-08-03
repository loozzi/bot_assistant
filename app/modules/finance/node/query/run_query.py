from datetime import date

from ...tools.categories import find_category_by_name
from ...tools.queries import search_total, search_transactions, spend_by_category, total_spend

_FALLBACK_REPLY = "Mình chưa hiểu câu hỏi này, bạn thử hỏi lại rõ hơn nhé."


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + "đ"


async def query_run_node(state: dict) -> dict:
    params = state.get("query_params")
    if not params:
        return {"reply": _FALLBACK_REPLY}

    session = state["session"]
    user_pk = state["user_pk"]
    start = date.fromisoformat(params["period_start"])
    end = date.fromisoformat(params["period_end"])

    if params["metric"] == "total":
        category_id = None
        if params.get("category"):
            category = await find_category_by_name(session, user_pk, params["category"])
            if category is None:
                return {"reply": f"Mình không tìm thấy danh mục '{params['category']}'."}
            category_id = category.id
        total = await total_spend(session, user_pk, start, end, category_id)
        label = f" cho {params['category']}" if params.get("category") else ""
        return {"reply": f"Bạn đã chi {_format_vnd(total)}{label} trong khoảng thời gian này."}

    if params["metric"] == "by_category":
        breakdown = await spend_by_category(session, user_pk, start, end)
        if not breakdown:
            return {"reply": "Chưa có giao dịch nào trong khoảng thời gian này."}
        lines = [f"• {row['category']}: {_format_vnd(row['total'])}" for row in breakdown]
        return {"reply": "Chi tiêu theo danh mục:\n" + "\n".join(lines)}

    if params["metric"] == "search":
        keyword = params.get("keyword") or ""
        results = await search_transactions(session, user_pk, keyword, start, end)
        if not results:
            return {"reply": f"Không tìm thấy giao dịch nào khớp với '{keyword}'."}
        total = await search_total(session, user_pk, keyword, start, end)
        lines = [f"• {r['description']}: {_format_vnd(r['amount'])} ({r['occurred_at']})" for r in results]
        return {"reply": f"Tổng {_format_vnd(total)} cho '{keyword}':\n" + "\n".join(lines)}

    return {"reply": _FALLBACK_REPLY}
