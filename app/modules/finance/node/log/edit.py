from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.edit import EDIT_SYSTEM_PROMPT, build_edit_user_message
from ...schema.parsed_transaction import ParseResult
from ...tools.categories import category_names
from ...tools.dates import today_ict
from .parse import verify_transactions

logger = get_logger(__name__)

MAX_EDIT_ROUNDS = 5

_ASK_EDIT_QUESTION = (
    "Bạn muốn sửa gì? Ví dụ: \"món 2 là 50k\", \"bỏ món 1\", \"trà sữa là hôm qua\".\n"
    "(Gõ /cancel để huỷ)"
)


async def log_ask_edit_node(state: dict) -> dict:
    """The user tapped "Sửa" — ask what to change, as free text."""
    answer = interrupt(HumanReviewRequest(question=_ASK_EDIT_QUESTION))
    return {"edit_instruction": answer}


async def log_apply_edit_node(state: dict) -> dict:
    """Apply a free-text correction to the pending (not yet saved)
    transactions, then hand back to log_confirm for another look.

    Amounts go through the same deterministic verification as log_parse.
    On LLM failure the list is left untouched and the user is told so on
    the next confirm prompt rather than the flow aborting.
    """
    rounds = state.get("edit_rounds", 0) + 1
    current = state.get("parsed_transactions", [])

    try:
        llm = create_llm_client()
        parser = llm.with_structured_output(ParseResult, method="json_mode")
        result: ParseResult = await parser.ainvoke(
            [
                {"role": "system", "content": EDIT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_edit_user_message(
                        current,
                        state.get("edit_instruction", ""),
                        today_ict().isoformat(),
                        category_names(state),
                    ),
                },
            ]
        )
    except Exception as exc:
        logger.warning("finance_edit_failed", error=str(exc))
        return {
            "edit_rounds": rounds,
            "edit_notice": ["Mình chưa áp dụng được yêu cầu sửa, bạn thử diễn đạt lại nhé."],
        }

    transactions, notes = verify_transactions(result)
    update: dict = {"parsed_transactions": transactions, "edit_rounds": rounds, "edit_notice": notes}
    if not transactions:
        update["reply"] = "Đã bỏ hết, không ghi giao dịch nào."
    return update
