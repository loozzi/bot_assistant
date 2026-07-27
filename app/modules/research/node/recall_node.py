from app.infra.providers.embedding import embed_text
from app.utils.logger import get_logger

from ..prompts.recall import NO_RECALL_RESULTS_MESSAGE, format_recall_results
from ..state import ResearchState
from ..tools import get_recall_min_score, get_recall_top_k, search_research_points

logger = get_logger(__name__)


async def recall_node(state: ResearchState) -> dict:
    """
    Answer a recall query by semantic search over previously saved research.

    Any failure (Qdrant unreachable/uninitialized, embedding error) is
    treated the same as "nothing found" — logs a warning and returns the
    canned no-results message rather than raising.
    """
    try:
        vector = await embed_text(state["user_query"], prefix="query: ")
        results = await search_research_points(
            user_id=state["user_id"],
            query_vector=vector,
            top_k=get_recall_top_k(),
            min_score=get_recall_min_score(),
        )
    except Exception as exc:
        logger.warning("recall_node_failed", error=str(exc))
        results = []

    if not results:
        return {"reply": NO_RECALL_RESULTS_MESSAGE}

    return {"reply": format_recall_results(results)}
