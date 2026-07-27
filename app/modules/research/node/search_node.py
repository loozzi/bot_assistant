from app.utils.logger import get_logger

from ..state import ResearchState
from ..tools import SearchToolError, tavily_search

logger = get_logger(__name__)


async def search_node(state: ResearchState) -> dict:
    """
    Perform a web search based on the user's query.

    If the search fails (missing API key, network error, etc.), returns an
    empty documents list so downstream nodes can handle it gracefully.
    """
    try:
        results = await tavily_search(state["user_query"])
    except SearchToolError as exc:
        logger.warning("research_search_node_failed", error=str(exc), query=state["user_query"])
        return {"documents": []}

    documents = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "raw_text": r.get("content", "")}
        for r in results
    ]
    return {"documents": documents}
