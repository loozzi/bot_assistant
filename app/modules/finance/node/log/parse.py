from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.parse import PARSE_SYSTEM_PROMPT, build_parse_user_message
from ...schema.parsed_transaction import ParseResult
from ...tools.amounts import parse_vnd
from ...tools.dates import today_ict

logger = get_logger(__name__)

_MIN_AMOUNT_VND = 1_000
_MAX_AMOUNT_VND = 500_000_000


def _today_iso() -> str:
    return today_ict().isoformat()


async def log_parse_node(state: dict) -> dict:
    """Parse the user's message into transactions, with amounts verified
    against the deterministic parser (never trust the LLM's arithmetic)."""
    try:
        llm = create_llm_client()
        parser = llm.with_structured_output(ParseResult, method="json_mode")
        result: ParseResult = await parser.ainvoke(
            [
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_parse_user_message(state["user_query"], _today_iso()),
                },
            ]
        )
    except Exception as exc:
        logger.warning("finance_parse_failed", error=str(exc))
        return {
            "parsed_transactions": [],
            "unparsed_notes": ["Mình chưa đọc hiểu được giao dịch, bạn thử nói rõ hơn nhé."],
        }

    transactions: list[dict] = []
    notes = list(result.unparsed_notes)

    for txn in result.transactions:
        verified_amount = parse_vnd(txn.raw_amount_text)
        amount_adjusted = verified_amount is not None and verified_amount != txn.amount_vnd
        final_amount = verified_amount if verified_amount is not None else txn.amount_vnd

        if not (_MIN_AMOUNT_VND <= final_amount <= _MAX_AMOUNT_VND):
            notes.append(f"Bỏ qua '{txn.description}' vì số tiền có vẻ không hợp lý.")
            continue

        dumped = txn.model_dump(mode="json")
        dumped["amount_vnd"] = final_amount
        dumped["amount_adjusted"] = amount_adjusted
        transactions.append(dumped)

    return {"parsed_transactions": transactions, "unparsed_notes": notes}
