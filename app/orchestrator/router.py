from app.core.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maps intent label → registered agent name
_INTENT_MAP: dict[str, str] = {
    "journal": "journal",
    "finance": "finance",
    "search": "search",
    "insight": "insight",
}


async def classify_intent(state: AgentState) -> str:
    """Return an intent label based on the latest user message.

    This is a keyword-based stub; replace with an LLM classifier in Phase 2.
    """
    last = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    if last is None:
        return "unknown"

    text = getattr(last, "content", "").lower()

    if any(w in text for w in ("cảm xúc", "nhật ký", "hôm nay", "mood", "journal", "feel")):
        return "journal"

    if any(w in text for w in ("tiêu", "chi tiêu", "mua", "tốn", "spend", "expense", "pay")):
        return "finance"

    if any(w in text for w in ("tìm", "search", "google", "tra", "lookup")):
        return "search"

    if any(w in text for w in ("phân tích", "xu hướng", "insight", "pattern", "trend")):
        return "insight"

    return "unknown"


def resolve_agent_name(intent: str) -> str | None:
    return _INTENT_MAP.get(intent)
