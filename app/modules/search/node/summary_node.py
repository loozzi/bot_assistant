from ..state import SearchState


async def summary_node(state: SearchState) -> dict:
    """
    Summarize the documents retrieved from the crawl node.
    """
    return {
        "summary": (
            f"[stub] Placeholder summary for: {state['user_query']!r} — "
            "search agent not yet connected to a real search/LLM provider."
        )
    }