import httpx

from app.config.settings import get_settings
from app.utils.logger import get_logger

from ._common import SearchToolError, get_max_results, get_timeout_seconds

logger = get_logger(__name__)


async def tavily_search(query: str) -> list[dict]:
    """
    Search the web using Tavily API.

    Args:
        query: The search query string

    Returns:
        List of dicts with keys: title, url, content

    Raises:
        SearchToolError: If API key is not set or the request fails
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        logger.warning("tavily_api_key_not_set")
        raise SearchToolError("TAVILY_API_KEY not configured")

    max_results = get_max_results()
    timeout_seconds = get_timeout_seconds()

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })
            return results
    except httpx.HTTPStatusError as exc:
        error_msg = f"HTTP {exc.response.status_code}"
        try:
            resp_preview = exc.response.text[:500]
            if resp_preview:
                error_msg += f": {resp_preview}"
        except Exception:
            pass
        logger.warning("tavily_request_failed", error=error_msg, status_code=exc.response.status_code)
        raise SearchToolError(f"Tavily API request failed: {error_msg}") from exc
    except httpx.TimeoutException as exc:
        logger.warning("tavily_request_failed", error="Request timeout", timeout_seconds=timeout_seconds)
        raise SearchToolError(f"Tavily API timeout (timeout={timeout_seconds}s)") from exc
    except httpx.HTTPError as exc:
        logger.warning("tavily_request_failed", error=str(exc) or "Network error")
        raise SearchToolError(f"Tavily API request failed: {exc}") from exc
    except Exception as exc:
        logger.warning("tavily_unexpected_error", error=str(exc), error_type=type(exc).__name__)
        raise SearchToolError(f"Unexpected error during Tavily search: {exc}") from exc
