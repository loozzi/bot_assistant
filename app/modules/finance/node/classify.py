from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.classify import CLASSIFY_SYSTEM_PROMPT
from ..schema.classify_result import ClassifyResult

logger = get_logger(__name__)


async def classify_node(state: dict) -> dict:
    """Pick one finance sub-intent for the user's message.

    Falls back to "fallback" on any LLM failure so the graph always has a
    valid sub_intent to route on.
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
        return {"sub_intent": result.sub_intent}
    except Exception as exc:
        logger.warning("finance_classify_failed", error=str(exc))
        return {"sub_intent": "fallback"}
