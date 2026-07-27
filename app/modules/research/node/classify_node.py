import re

from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.classify import CLASSIFY_SYSTEM_PROMPT
from ..schema.mode_classification import ModeClassification
from ..state import ResearchState

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")


async def classify_node(state: ResearchState) -> dict:
    """
    Classify the user's message into new_link / new_keyword / recall.

    A URL found in the message always forces new_link (unless the LLM
    classified it as recall — "did I already save this link" stays recall).
    On any LLM failure, falls back to new_link if a URL is present, else
    new_keyword — new_keyword mirrors search's current one-flow behavior.
    """
    url_match = _URL_PATTERN.search(state["user_query"])

    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ModeClassification, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ]
        )
        mode = result.mode
    except Exception as exc:
        logger.warning("classify_node_llm_call_failed", error=str(exc))
        mode = "new_link" if url_match else "new_keyword"

    if url_match and mode != "recall":
        mode = "new_link"

    output: dict = {"mode": mode}
    if url_match:
        output["url"] = url_match.group(0)
    return output
