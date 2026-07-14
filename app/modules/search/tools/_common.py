import functools

import yaml

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SearchToolError(Exception):
    """Raised when a Tavily search API call fails."""
    pass


class CrawlToolError(Exception):
    """Raised when a Jina crawl call fails."""
    pass


@functools.lru_cache(maxsize=1)
def _load_search_config() -> dict:
    """Load the search module's config.yaml to get tuning parameters."""
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    return cfg.get("tuning", {})


def get_max_results() -> int:
    """Get max_results from config.yaml, with fallback to 3."""
    config = _load_search_config()
    return config.get("max_results", 3)


def get_timeout_seconds() -> float:
    """Get timeout_seconds from config.yaml, with fallback to 10."""
    config = _load_search_config()
    return float(config.get("timeout_seconds", 10))
