import httpx

from app.config.settings import get_settings
from app.utils.logger import get_logger

from ._common import CrawlToolError, get_timeout_seconds

logger = get_logger(__name__)


async def jina_fetch(url: str) -> str:
    """
    Fetch and extract content from a URL using Jina Reader API.
    
    Args:
        url: The URL to fetch
    
    Returns:
        Extracted text content, truncated to 4000 characters
    
    Raises:
        CrawlToolError: If the request fails
    """
    settings = get_settings()
    timeout_seconds = get_timeout_seconds()
    
    headers = {}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"
    
    jina_url = f"https://r.jina.ai/{url}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
            content = response.text
            # Truncate to 4000 characters
            return content[:4000]
    except httpx.HTTPStatusError as exc:
        # Capture HTTP status error details
        error_msg = f"HTTP {exc.response.status_code}"
        try:
            resp_preview = exc.response.text[:500]
            if resp_preview:
                error_msg += f": {resp_preview}"
        except Exception:
            pass
        logger.warning("jina_request_failed", url=url, error=error_msg, status_code=exc.response.status_code)
        raise CrawlToolError(f"Jina fetch failed for {url}: {error_msg}") from exc
    except httpx.TimeoutException as exc:
        logger.warning("jina_request_failed", url=url, error="Request timeout", timeout_seconds=timeout_seconds)
        raise CrawlToolError(f"Jina fetch timeout for {url} (timeout={timeout_seconds}s)") from exc
    except httpx.HTTPError as exc:
        logger.warning("jina_request_failed", url=url, error=str(exc) or "Network error")
        raise CrawlToolError(f"Jina fetch failed for {url}: {exc}") from exc
    except Exception as exc:
        logger.warning("jina_unexpected_error", url=url, error=str(exc), error_type=type(exc).__name__)
        raise CrawlToolError(f"Unexpected error fetching {url}: {exc}") from exc
