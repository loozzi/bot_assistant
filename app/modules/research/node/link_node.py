from ..state import ResearchState


async def link_node(state: ResearchState) -> dict:
    """
    Build a single-document placeholder for the classified URL.

    title/raw_text are filled in by crawl_node — there's nothing to search
    for since the user gave the link directly, so this always proceeds to
    a full crawl (see agent.py's edges). Returns no documents if there's no
    URL to work with (shouldn't normally happen, but classify_node's mode
    and url are set independently).
    """
    url = state.get("url", "")
    if not url:
        return {"documents": []}
    return {"documents": [{"title": "", "url": url, "raw_text": ""}]}
