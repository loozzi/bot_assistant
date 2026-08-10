from ..state import FinancialState

NOT_IMPLEMENTED_MESSAGE = "Tính năng quản lý chi tiêu đang được phát triển, quay lại sau nhé!"


async def placeholder_node(state: FinancialState) -> dict:
    return {"reply": NOT_IMPLEMENTED_MESSAGE}
