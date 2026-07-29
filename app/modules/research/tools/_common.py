import functools
from pathlib import Path

import yaml

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SearchToolError(Exception):
    """Raised when a Tavily search API call fails."""


class CrawlToolError(Exception):
    """Raised when a Jina crawl call fails."""


@functools.lru_cache(maxsize=1)
def _load_research_config() -> dict:
    """Load the research module's config.yaml (cached)."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    return yaml.safe_load(config_path.read_text())


def get_max_results() -> int:
    return _load_research_config().get("tuning", {}).get("max_results", 3)


def get_timeout_seconds() -> float:
    return float(_load_research_config().get("tuning", {}).get("timeout_seconds", 10))


def get_recall_top_k() -> int:
    return _load_research_config().get("tuning", {}).get("recall_top_k", 5)


def get_recall_min_score() -> float:
    return float(_load_research_config().get("tuning", {}).get("recall_min_score", 0.5))


def get_default_importance() -> float:
    return float(_load_research_config().get("memory", {}).get("default_importance", 0.5))
