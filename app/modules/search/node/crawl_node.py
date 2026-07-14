import asyncio

from ..state import SearchState
from ..tools import jina_fetch, CrawlToolError
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def crawl_node(state: SearchState) -> dict:
    """
    Crawl the web pages referenced in documents to extract their full content.
    
    If documents list is empty, pass through unchanged.
    For each document, attempt to fetch full content via Jina. If a single
    fetch fails, keep the document's existing raw_text (Tavily snippet).
    Multiple fetches run in parallel.
    """
    documents = state.get("documents", [])
    
    if not documents:
        return {"documents": documents}
    
    async def fetch_with_fallback(doc: dict) -> dict:
        """Fetch full content for a doc; keep original on failure."""
        url = doc.get("url", "")
        if not url:
            return doc
        
        try:
            content = await jina_fetch(url)
            doc = doc.copy()
            doc["raw_text"] = content
            return doc
        except CrawlToolError as exc:
            logger.warning("crawl_node_fetch_failed", url=url, error=str(exc))
            # Keep the original Tavily snippet
            return doc
    
    # Fetch all docs in parallel
    updated_docs = await asyncio.gather(
        *[fetch_with_fallback(doc) for doc in documents]
    )
    
    return {"documents": updated_docs}