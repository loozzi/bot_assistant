from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.category import CATEGORY_SYSTEM_PROMPT
from ...schema.category_params import CategoryParams
from ...tools.categories import category_names

logger = get_logger(__name__)

_MISSING = {
    "create": "Bạn muốn thêm danh mục tên gì?",
    "rename": "Bạn muốn đổi tên danh mục nào, thành tên gì? Ví dụ: \"đổi tên Mua sắm thành Shopping\".",
    "merge": "Bạn muốn gộp danh mục nào vào danh mục nào? Ví dụ: \"gộp Cà phê vào Ăn uống\".",
    "delete": "Bạn muốn xoá danh mục nào?",
}


def _missing_fields(params: CategoryParams) -> bool:
    if params.action == "list":
        return False
    if not (params.name or "").strip():
        return True
    return params.action in ("rename", "merge") and not (params.target_name or "").strip()


async def category_extract_params_node(state: dict) -> dict:
    try:
        llm = create_llm_client()
        extractor = llm.with_structured_output(CategoryParams, method="json_mode")
        params: CategoryParams = await extractor.ainvoke(
            [
                {
                    "role": "system",
                    "content": CATEGORY_SYSTEM_PROMPT.format(categories=", ".join(category_names(state))),
                },
                {"role": "user", "content": state["user_query"]},
            ]
        )
    except Exception as exc:
        logger.warning("finance_category_extract_failed", error=str(exc))
        return {"reply": "Mình chưa hiểu bạn muốn làm gì với danh mục, bạn nói rõ hơn nhé."}

    if _missing_fields(params):
        return {"reply": _MISSING[params.action]}
    return {"category_params": params.model_dump(mode="json")}
