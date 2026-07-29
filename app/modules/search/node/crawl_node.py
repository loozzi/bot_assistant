import asyncio
import hashlib

from ..state import SearchState
from ..tools import jina_fetch, CrawlToolError
from app.infra.memory.working import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CRAWL_CACHE_TTL = 86400  # 24h


def _cache_key(url: str) -> str:
    return f"crawl_url:{hashlib.md5(url.encode('utf-8')).hexdigest()}"


async def crawl_node(state: SearchState) -> dict:
    """
    Crawl the web pages referenced in documents to extract their full content.

    If documents list is empty, pass through unchanged.
    For each document, check the Redis cache first; on a miss, fetch full
    content via Jina and cache it for 24h. If a single fetch fails, keep the
    document's existing raw_text (Tavily snippet). Multiple fetches run in
    parallel. Redis being unavailable degrades to always fetching — it never
    blocks crawling.
    """
    documents = state.get("documents", [])

    if not documents:
        return {"documents": documents}

    async def fetch_with_fallback(doc: dict) -> dict:
        """Fetch full content for a doc (via cache or Jina); keep original on failure."""
        url = doc.get("url", "")
        if not url:
            return doc

        cache_key = _cache_key(url)
        try:
            redis = get_redis_client()
            cached = await redis.get(cache_key)
            if cached:
                doc = doc.copy()
                doc["raw_text"] = cached
                return doc
        except Exception as exc:
            logger.debug("crawl_cache_read_failed", url=url, error=str(exc))

        try:
            content = await jina_fetch(url)
        except CrawlToolError as exc:
            logger.warning("crawl_node_fetch_failed", url=url, error=str(exc))
            # Keep the original Tavily snippet
            return doc

        doc = doc.copy()
        doc["raw_text"] = content

        try:
            redis = get_redis_client()
            await redis.setex(cache_key, _CRAWL_CACHE_TTL, content)
        except Exception as exc:
            logger.debug("crawl_cache_write_failed", url=url, error=str(exc))

        return doc

    # Fetch all docs in parallel
    updated_docs = await asyncio.gather(
        *[fetch_with_fallback(doc) for doc in documents]
    )

    return {"documents": updated_docs}