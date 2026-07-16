from ..state import SearchState
from ..tools import tavily_search, SearchToolError
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def search_node(state: SearchState) -> dict:
    """
    Perform a web search based on the user's query and return relevant documents.
    
    If the search fails (missing API key, network error, etc.), returns an empty
    documents list so downstream nodes can handle gracefully.
    """
    try:
        results = await tavily_search(state["user_query"])
    except SearchToolError as exc:
        logger.warning("search_node_failed", error=str(exc), query=state["user_query"])
        return {"documents": []}
    
    documents = []
    for result in results:
        documents.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "raw_text": result.get("content", ""),
        })
    
    return {"documents": documents}