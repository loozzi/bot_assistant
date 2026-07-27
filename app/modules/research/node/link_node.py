from ..state import ResearchState


async def link_node(state: ResearchState) -> dict:
    """
    Build a single-document placeholder for the classified URL.

    title/raw_text are filled in by crawl_node — there's nothing to search
    for since the user gave the link directly, so this always proceeds to
    a full crawl (see agent.py's edges).
    """
    url = state.get("url", "")
    return {"documents": [{"title": "", "url": url, "raw_text": ""}]}
