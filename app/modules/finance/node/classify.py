from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.classify import CLASSIFY_SYSTEM_PROMPT
from ..schema.classify_result import ClassifyResult

logger = get_logger(__name__)


# Writes before reads, so "trà sữa 45k, tháng này tiêu bao nhiêu?" answers
# the question with the new transaction already counted.
_ORDER = ["category", "modify", "log", "query", "budget", "goal", "split", "advice", "fallback"]
_MAX_INTENTS = 3


def order_intents(intents: list[str]) -> list[str]:
    unique = sorted(set(intents), key=_ORDER.index)
    if len(unique) > 1 and "fallback" in unique:
        unique.remove("fallback")
    return unique[:_MAX_INTENTS]


async def classify_node(state: dict) -> dict:
    """Pick the finance sub-intent(s) in the user's message.

    The first goes in sub_intent, the rest queue up in pending_intents and
    run one after another (see agent.py's next_intent node). Falls back to
    "fallback" on any LLM failure so the graph always has something to
    route on.
    """
    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ClassifyResult, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ]
        )
        ordered = order_intents(result.sub_intents)
        return {"sub_intent": ordered[0], "pending_intents": ordered[1:]}
    except Exception as exc:
        logger.warning("finance_classify_failed", error=str(exc))
        return {"sub_intent": "fallback", "pending_intents": []}
