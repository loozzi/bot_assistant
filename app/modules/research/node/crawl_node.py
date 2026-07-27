import asyncio

from app.utils.logger import get_logger

from ..state import ResearchState
from ..tools import CrawlToolError, jina_fetch

logger = get_logger(__name__)


async def crawl_node(state: ResearchState) -> dict:
    """
    Crawl the pages referenced in documents to extract their full content.

    If documents is empty, pass through unchanged. For each document,
    attempt to fetch full content via Jina; on failure, keep whatever
    raw_text it already had (empty for link_node, a Tavily snippet for
    search_node). Fetches run in parallel.
    """
    documents = state.get("documents", [])

    if not documents:
        return {"documents": documents}

    async def fetch_with_fallback(doc: dict) -> dict:
        url = doc.get("url", "")
        if not url:
            return doc

        try:
            content = await jina_fetch(url)
            doc = doc.copy()
            doc["raw_text"] = content
            return doc
        except CrawlToolError as exc:
            logger.warning("research_crawl_node_fetch_failed", url=url, error=str(exc))
            return doc

    updated_docs = await asyncio.gather(
        *[fetch_with_fallback(doc) for doc in documents]
    )

    return {"documents": updated_docs}
