from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.modify import MODIFY_SYSTEM_PROMPT, build_modify_user_message
from ...schema.modify_params import ModifyParams
from ...tools.amounts import parse_vnd
from ...tools.categories import category_names
from ...tools.dates import today_ict
from ..log.parse import _MAX_AMOUNT_VND, _MIN_AMOUNT_VND

logger = get_logger(__name__)

_NOT_UNDERSTOOD = "Mình chưa hiểu bạn muốn xoá/sửa giao dịch nào, bạn nói rõ hơn nhé."
_NO_CHANGES = "Bạn muốn sửa giao dịch đó thành gì? Ví dụ: \"sửa trà sữa hôm qua thành 50k\"."


async def modify_extract_params_node(state: dict) -> dict:
    """Work out delete-vs-update, which saved log is meant, and (for an
    update) the new values — with the new amount re-derived from its raw
    text by the deterministic parser, never taken from the LLM's number."""
    try:
        llm = create_llm_client()
        extractor = llm.with_structured_output(ModifyParams, method="json_mode")
        params: ModifyParams = await extractor.ainvoke(
            [
                {"role": "system", "content": MODIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_modify_user_message(
                        state["user_query"], today_ict().isoformat(), category_names(state)
                    ),
                },
            ]
        )
    except Exception as exc:
        logger.warning("finance_modify_extract_failed", error=str(exc))
        return {"reply": _NOT_UNDERSTOOD}

    changes: dict = {}
    if params.action == "update":
        raw = params.changes.raw_amount_text
        if raw:
            amount = parse_vnd(raw)
            if amount is None or not (_MIN_AMOUNT_VND <= amount <= _MAX_AMOUNT_VND):
                return {"reply": f"Số tiền '{raw}' có vẻ không hợp lý, bạn kiểm tra lại nhé."}
            changes["amount_vnd"] = amount
            changes["raw_amount_text"] = raw
        for field in ("description", "category", "log_type"):
            value = getattr(params.changes, field)
            if value:
                changes[field] = value
        if params.changes.occurred_at is not None:
            changes["occurred_at"] = params.changes.occurred_at.isoformat()
        if not changes:
            return {"reply": _NO_CHANGES}

    return {
        "modify_action": params.action,
        "modify_target": params.target.model_dump(mode="json"),
        "modify_changes": changes,
    }
