from ..state import SearchState


async def crawl_node(state: SearchState) -> dict:
    """
    Crawl the web for documents based on the user's query.
    """
    return {"documents": state["documents"]}