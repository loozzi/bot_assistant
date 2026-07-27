# Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `research` module that saves link/keyword research summaries to Qdrant and answers recall questions about previously saved research, following the `search` module's subgraph pattern.

**Architecture:** `app/modules/research/agent.py` wraps an internal LangGraph `StateGraph` (`classify → {link, search→confirm, recall} → crawl → summarize_and_save → END`) behind `BaseAgent`, using the same `run_interruptible_subgraph` HITL bridge `search` uses. A new shared `app/infra/providers/embedding.py` provides local sentence-transformers embeddings for both saving (`summarize_and_save_node`) and recall (`recall_node`).

**Tech Stack:** Python 3.11, LangGraph (`StateGraph`, `interrupt`), Pydantic structured LLM output, `sentence-transformers` (new dependency), Qdrant (`qdrant-client`), `httpx` for Tavily/Jina calls.

## Global Constraints

- No cross-module imports — `research` must not import from `app.modules.search` (per CLAUDE.md dependency rules). Tavily/Jina wrappers are duplicated into `research/tools/`, not shared.
- No Postgres table for research items — Qdrant is the only persistence layer (per spec's non-goals).
- No automated test suite exists in this repo (no pytest/ruff config) — every task's verification step is a manual `uv run python -c "..."` smoke check, not a pytest run.
- Every external call (Tavily, Jina, LLM, Qdrant, embedding) must degrade gracefully — log a warning and return a safe fallback, never raise out of a node.
- `agent.py` module-level `graph = builder.compile(checkpointer=get_checkpointer())` requires the checkpointer to already be initialized (`init_checkpointer()`, which needs Redis reachable) before that module is imported — this only matters for Task 11's verification.
- Config values (routing, tuning, memory defaults) live in `app/modules/research/config.yaml`, loaded via the same `yaml.safe_load` + `functools.lru_cache` pattern `search` uses — never hardcode tuning numbers in node code.

---

### Task 1: Embedding provider + dependency

**Files:**
- Modify: `pyproject.toml:7-34` (add `sentence-transformers` dependency)
- Modify: `app/config/settings.py:27` (fix `embedding_model` default to a loadable HuggingFace repo id)
- Create: `app/infra/providers/embedding.py`

**Interfaces:**
- Consumes: `app.config.settings.get_settings()` (existing) — `settings.embedding_model`, `settings.embedding_device`.
- Produces: `async def embed_text(text: str, *, prefix: str) -> list[float]` — used by Task 9 (`summarize_and_save_node`) and Task 10 (`recall_node`). `prefix` is always `"passage: "` (content being stored) or `"query: "` (recall query), per the e5 model's asymmetric training convention.

- [ ] **Step 1: Add the `sentence-transformers` dependency**

Edit `pyproject.toml`, inside the `dependencies` list, add a new line after `"qdrant-client>=1.9",`:

```toml
    # Vector DB
    "qdrant-client>=1.9",
    # Embeddings
    "sentence-transformers>=3.0",
```

- [ ] **Step 2: Install it**

Run: `uv sync`
Expected: completes without error; `sentence-transformers` (and `torch`) appear in `uv.lock`.

- [ ] **Step 3: Fix the embedding model setting to a real HuggingFace repo id**

In `app/config/settings.py`, line 27 currently reads:

```python
    embedding_model: str = "multilingual-e5-large"
```

Change it to:

```python
    embedding_model: str = "intfloat/multilingual-e5-large"
```

This is the actual loadable HuggingFace repo id — the bare name isn't a valid `SentenceTransformer()` argument. Nothing loaded this setting before now, so this was latent.

- [ ] **Step 4: Write the embedding provider**

Create `app/infra/providers/embedding.py`:

```python
import asyncio
import functools

from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    logger.info(
        "embedding_model_loading",
        model=settings.embedding_model,
        device=settings.embedding_device,
    )
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info("embedding_model_loaded", model=settings.embedding_model)
    return model


async def embed_text(text: str, *, prefix: str) -> list[float]:
    """Embed `prefix + text` with the configured sentence-transformers model.

    `prefix` must be "passage: " for content being stored and "query: " for
    a recall query, per the e5 model's asymmetric training convention.
    """
    model = _load_model()
    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(None, model.encode, prefix + text)
    return vector.tolist()
```

- [ ] **Step 5: Verify it loads and embeds**

Run: `uv run python -c "import asyncio; from app.infra.providers.embedding import embed_text; v = asyncio.run(embed_text('hello world', prefix='query: ')); print(len(v))"`
Expected: prints `1024`. First run downloads the model from HuggingFace (needs network, may take a minute); subsequent runs are fast and cached under `~/.cache/huggingface`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/config/settings.py app/infra/providers/embedding.py
git commit -m "feat: add shared embedding provider (sentence-transformers)"
```

---

### Task 2: Research module scaffolding — state, config

**Files:**
- Create: `app/modules/research/__init__.py`
- Create: `app/modules/research/config.yaml`
- Create: `app/modules/research/state.py`
- Create: `app/modules/research/node/__init__.py` (empty placeholder, filled in by later tasks)
- Create: `app/modules/research/tools/__init__.py` (empty placeholder, filled in by Task 3)
- Create: `app/modules/research/schema/__init__.py` (empty placeholder, filled in by Task 4)
- Create: `app/modules/research/prompts/__init__.py` (empty placeholder, filled in by Task 4)

**Interfaces:**
- Produces: `ResearchState` TypedDict (`user_id`, `user_query`, `mode`, `url`, `documents`, `skip_crawl`, `reply`) and `DocumentInfo` TypedDict (`title`, `url`, `raw_text`) — used by every node task (5–10) and `agent.py` (Task 11). `config` dict loaded from `config.yaml` — consumed by `tools/_common.py` (Task 3) via its own independent load, and by `agent.py` (Task 11) for `agent.name`/`agent.description`.

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p "app/modules/research/node" "app/modules/research/tools" "app/modules/research/schema" "app/modules/research/prompts"
touch "app/modules/research/__init__.py" "app/modules/research/node/__init__.py" "app/modules/research/tools/__init__.py" "app/modules/research/schema/__init__.py" "app/modules/research/prompts/__init__.py"
```

- [ ] **Step 2: Write `config.yaml`**

Create `app/modules/research/config.yaml`:

```yaml
# app/modules/research/config.yaml
# Declarative metadata the orchestrator uses to classify intent and route to
# this agent. Loaded by orchestrator/registry.py at startup — do not put
# secrets or environment-specific values here (those belong in .env).

agent:
  name: research
  enabled: true
  description: >
    Saves web links or keyword research (crawled and summarized) for later
    recall, and answers questions about previously saved research.

routing:
  keywords:
    vi: [lưu lại, nghiên cứu, tra cứu tài liệu, đọc lại, tóm tắt bài, đường link]
    en: [research, save this, read later, summarize this article, remember this link]
  examples:
    - "Lưu link này lại giúp tôi: https://example.com/article"
    - "Nghiên cứu giúp tôi về xu hướng AI 2026 rồi lưu lại"
    - "Tôi từng đọc gì về chủ đề blockchain chưa nhỉ?"
    - "Summarize and save this article: https://example.com/post"
  sticky: false
  priority: 20

memory:
  source_type: research
  default_importance: 0.5

tuning:
  max_results: 3
  timeout_seconds: 10
  recall_top_k: 5
  recall_min_score: 0.5
```

- [ ] **Step 3: Write `state.py`**

Create `app/modules/research/state.py`:

```python
from typing import Literal

from typing_extensions import NotRequired, TypedDict


class DocumentInfo(TypedDict):
    title: str
    url: str
    raw_text: str


class ResearchState(TypedDict):
    user_id: str
    user_query: str
    mode: NotRequired[Literal["new_link", "new_keyword", "recall"]]
    url: NotRequired[str]
    documents: list[DocumentInfo]
    skip_crawl: NotRequired[bool]
    reply: str
```

- [ ] **Step 4: Verify config and state import cleanly**

Run: `uv run python -c "import yaml; from pathlib import Path; cfg = yaml.safe_load(Path('app/modules/research/config.yaml').read_text()); print(cfg['agent']['name'], cfg['memory']['default_importance'], cfg['tuning']['recall_top_k'])"`
Expected: `research 0.5 5`

Run: `uv run python -c "from app.modules.research.state import ResearchState, DocumentInfo; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add app/modules/research
git commit -m "feat: scaffold research module (config, state)"
```

---

### Task 3: Tools — Tavily, Jina, shared config helpers

**Files:**
- Create: `app/modules/research/tools/_common.py`
- Create: `app/modules/research/tools/tavily.py`
- Create: `app/modules/research/tools/jina.py`
- Modify: `app/modules/research/tools/__init__.py` (export the tool functions)

**Interfaces:**
- Consumes: `app.config.settings.get_settings()` (existing), `app.utils.logger.get_logger` (existing).
- Produces: `async def tavily_search(query: str) -> list[dict]` (keys: `title`, `url`, `content`), `async def jina_fetch(url: str) -> str`, `class SearchToolError(Exception)`, `class CrawlToolError(Exception)`, `get_max_results() -> int`, `get_timeout_seconds() -> float`, `get_recall_top_k() -> int`, `get_recall_min_score() -> float`, `get_default_importance() -> float` — used by Task 7 (`search_node`, `link_node`), Task 8 (`crawl_node`), Task 9 (`summarize_and_save_node`), Task 10 (`recall_node`).

- [ ] **Step 1: Write `_common.py`**

Create `app/modules/research/tools/_common.py`:

```python
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
```

- [ ] **Step 2: Write `tavily.py`**

Create `app/modules/research/tools/tavily.py` (identical logic to `app/modules/search/tools/tavily.py`, reading research's own config helpers):

```python
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
```

- [ ] **Step 3: Write `jina.py`**

Create `app/modules/research/tools/jina.py` (identical logic to `app/modules/search/tools/jina.py`):

```python
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
            return content[:4000]
    except httpx.HTTPStatusError as exc:
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
```

- [ ] **Step 4: Export from `tools/__init__.py`**

Replace the contents of `app/modules/research/tools/__init__.py`:

```python
from ._common import (
    CrawlToolError,
    SearchToolError,
    get_default_importance,
    get_max_results,
    get_recall_min_score,
    get_recall_top_k,
    get_timeout_seconds,
)
from .jina import jina_fetch
from .tavily import tavily_search

__all__ = [
    "tavily_search",
    "jina_fetch",
    "SearchToolError",
    "CrawlToolError",
    "get_max_results",
    "get_timeout_seconds",
    "get_recall_top_k",
    "get_recall_min_score",
    "get_default_importance",
]
```

- [ ] **Step 5: Verify imports and config helpers**

Run: `uv run python -c "from app.modules.research.tools import tavily_search, jina_fetch, SearchToolError, CrawlToolError, get_max_results, get_recall_top_k, get_default_importance; print(get_max_results(), get_recall_top_k(), get_default_importance())"`
Expected: `3 5 0.5`

- [ ] **Step 6: Commit**

```bash
git add app/modules/research/tools
git commit -m "feat: add research Tavily/Jina tool wrappers"
```

---

### Task 4: Schema and prompts

**Files:**
- Create: `app/modules/research/schema/mode_classification.py`
- Create: `app/modules/research/schema/research_summary.py`
- Modify: `app/modules/research/schema/__init__.py`
- Create: `app/modules/research/prompts/classify.py`
- Create: `app/modules/research/prompts/summary.py`
- Create: `app/modules/research/prompts/recall.py`

**Interfaces:**
- Produces: `class ModeClassification(BaseModel)` with `mode: Literal["new_link", "new_keyword", "recall"]` — consumed by Task 6 (`classify_node`). `class ResearchSummary(BaseModel)` with `answer: str`, `key_points: list[str]`, `topics: list[str]`, method `.render() -> str` — consumed by Task 9. `CLASSIFY_SYSTEM_PROMPT: str` — Task 6. `SUMMARY_SYSTEM_PROMPT: str`, `NO_RESULTS_MESSAGE: str`, `format_fallback_summary(documents: list[dict]) -> str` — Task 9. `NO_RECALL_RESULTS_MESSAGE: str`, `format_recall_results(results: list[dict]) -> str` — Task 10.

- [ ] **Step 1: Write `schema/mode_classification.py`**

Create `app/modules/research/schema/mode_classification.py`:

```python
from typing import Literal

from pydantic import BaseModel


class ModeClassification(BaseModel):
    """Structured output for classify_node: which research flow applies."""
    mode: Literal["new_link", "new_keyword", "recall"]
```

- [ ] **Step 2: Write `schema/research_summary.py`**

Create `app/modules/research/schema/research_summary.py`:

```python
from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    """Structured output from the summarize-and-save LLM call."""
    answer: str
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Render as a formatted string for the Telegram reply."""
        lines = [self.answer]

        if self.key_points:
            lines.append("")
            for point in self.key_points:
                lines.append(f"• {point}")

        return "\n".join(lines)
```

- [ ] **Step 3: Export from `schema/__init__.py`**

Replace the contents of `app/modules/research/schema/__init__.py`:

```python
from .mode_classification import ModeClassification
from .research_summary import ResearchSummary

__all__ = ["ModeClassification", "ResearchSummary"]
```

- [ ] **Step 4: Write `prompts/classify.py`**

Create `app/modules/research/prompts/classify.py`:

```python
CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for a personal research assistant.

Classify the user's latest message into exactly one of:
- "new_link": the message contains a URL to read, save, or summarize
- "new_keyword": the message asks to research or look up a topic by keywords (no URL given)
- "recall": the message asks what the user previously read, saved, or researched about a topic

Respond only with JSON of the form {"mode": "new_link" | "new_keyword" | "recall"}."""
```

- [ ] **Step 5: Write `prompts/summary.py`**

Create `app/modules/research/prompts/summary.py`:

```python
NO_RESULTS_MESSAGE = (
    "Không tìm thấy nội dung để tóm tắt. "
    "Vui lòng thử lại với link hoặc từ khóa khác."
)

SUMMARY_SYSTEM_PROMPT = """You are a research assistant that summarizes web content for later recall.

Your task:
1. Read the provided documents
2. Answer the user's query based ONLY on the content in these documents
3. If the documents don't contain enough information to answer confidently, say so
4. Always respond in the same language as the user's query
5. Be concise and factual

Format your response as JSON with these fields:
- answer: A direct, concise answer to the user's question
- key_points: A list of 2-4 important supporting points (can be empty if not relevant)
- topics: A list of 2-5 short topic tags describing this content, for future retrieval"""


def format_fallback_summary(documents: list[dict]) -> str:
    """
    Create a fallback summary when the LLM call fails.

    Shows document titles and URLs without LLM processing.

    Args:
        documents: List of document dicts with keys: title, url, raw_text

    Returns:
        Formatted text with title + URL list for top 3 documents
    """
    if not documents:
        return NO_RESULTS_MESSAGE

    lines = ["Nội dung đã tìm được:"]
    for doc in documents[:3]:
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        if url:
            lines.append(f"• [{title}]({url})")
        else:
            lines.append(f"• {title}")

    return "\n".join(lines)
```

- [ ] **Step 6: Write `prompts/recall.py`**

Create `app/modules/research/prompts/recall.py`:

```python
NO_RECALL_RESULTS_MESSAGE = (
    "Chưa có gì được lưu về chủ đề này. "
    "Hãy gửi link hoặc từ khóa để tôi nghiên cứu và lưu lại."
)


def format_recall_results(results: list[dict]) -> str:
    """
    Format saved research items directly as a list, no LLM call.

    Args:
        results: List of dicts with keys: title, summary, url, timestamp, score

    Returns:
        Formatted text listing each saved item.
    """
    lines = ["Đây là những gì bạn đã từng tìm hiểu:"]
    for r in results:
        title = r.get("title") or "Untitled"
        summary = r.get("summary", "")
        url = r.get("url", "")
        timestamp = (r.get("timestamp") or "")[:10]

        lines.append("")
        header = f"• {title}"
        if timestamp:
            header += f" ({timestamp})"
        lines.append(header)
        if summary:
            lines.append(f"  {summary}")
        if url:
            lines.append(f"  {url}")

    return "\n".join(lines)
```

- [ ] **Step 7: Verify**

Run: `uv run python -c "
from app.modules.research.schema import ModeClassification, ResearchSummary
from app.modules.research.prompts.classify import CLASSIFY_SYSTEM_PROMPT
from app.modules.research.prompts.summary import NO_RESULTS_MESSAGE, format_fallback_summary
from app.modules.research.prompts.recall import NO_RECALL_RESULTS_MESSAGE, format_recall_results

m = ModeClassification(mode='recall')
s = ResearchSummary(answer='ans', key_points=['a', 'b'], topics=['x'])
print(m.mode)
print(s.render())
print(format_fallback_summary([]))
print(format_recall_results([{'title': 'T', 'summary': 'S', 'url': 'U', 'timestamp': '2026-07-27T00:00:00'}]))
"`
Expected: prints `recall`, then `ans\n\n• a\n• b`, then the `NO_RESULTS_MESSAGE` text, then a formatted list containing `• T (2026-07-27)`.

- [ ] **Step 8: Commit**

```bash
git add app/modules/research/schema app/modules/research/prompts
git commit -m "feat: add research schema and prompts"
```

---

### Task 5: Qdrant storage helpers

**Files:**
- Create: `app/modules/research/tools/qdrant_store.py`

**Interfaces:**
- Consumes: `app.infra.db.vector.get_qdrant_client()` (existing, raises `RuntimeError` if `init_vector_db()` hasn't run), `app.config.settings.get_settings()` (existing) for `settings.qdrant_collection`.
- Produces: `async def upsert_research_point(*, user_id: str, url: str, title: str, summary: str, topics: list[str], importance: float, vector: list[float]) -> None` and `async def search_research_points(*, user_id: str, query_vector: list[float], top_k: int, min_score: float) -> list[dict]` (each dict has keys `title`, `summary`, `url`, `timestamp`, `score`) — used by Task 9 and Task 10.

- [ ] **Step 1: Write `qdrant_store.py`**

Create `app/modules/research/tools/qdrant_store.py`:

```python
import uuid
from datetime import datetime, timezone

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client

SOURCE_TYPE = "research"


async def upsert_research_point(
    *,
    user_id: str,
    url: str,
    title: str,
    summary: str,
    topics: list[str],
    importance: float,
    vector: list[float],
) -> None:
    """Store one research item as a Qdrant point, scoped to user_id + source_type."""
    client = get_qdrant_client()
    settings = get_settings()

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "user_id": user_id,
            "source_type": SOURCE_TYPE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance": importance,
            "topics": topics,
            "url": url,
            "title": title,
            "summary": summary,
        },
    )
    await client.upsert(collection_name=settings.qdrant_collection, points=[point])


async def search_research_points(
    *,
    user_id: str,
    query_vector: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Semantic search over this user's saved research items."""
    client = get_qdrant_client()
    settings = get_settings()

    results = await client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="source_type", match=MatchValue(value=SOURCE_TYPE)),
            ]
        ),
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "title": r.payload.get("title", ""),
            "summary": r.payload.get("summary", ""),
            "url": r.payload.get("url", ""),
            "timestamp": r.payload.get("timestamp", ""),
            "score": r.score,
        }
        for r in results
    ]
```

- [ ] **Step 2: Add to `tools/__init__.py`**

Edit `app/modules/research/tools/__init__.py` — add the import and export:

```python
from ._common import (
    CrawlToolError,
    SearchToolError,
    get_default_importance,
    get_max_results,
    get_recall_min_score,
    get_recall_top_k,
    get_timeout_seconds,
)
from .jina import jina_fetch
from .qdrant_store import search_research_points, upsert_research_point
from .tavily import tavily_search

__all__ = [
    "tavily_search",
    "jina_fetch",
    "SearchToolError",
    "CrawlToolError",
    "get_max_results",
    "get_timeout_seconds",
    "get_recall_top_k",
    "get_recall_min_score",
    "get_default_importance",
    "upsert_research_point",
    "search_research_points",
]
```

- [ ] **Step 3: Verify import (functional test requires a running Qdrant, deferred to Task 9/10 verification)**

Run: `uv run python -c "from app.modules.research.tools import upsert_research_point, search_research_points; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add app/modules/research/tools
git commit -m "feat: add research Qdrant storage helpers"
```

---

### Task 6: classify_node

**Files:**
- Create: `app/modules/research/node/classify_node.py`

**Interfaces:**
- Consumes: `app.infra.providers.llm_client.create_llm_client()` (existing), `ModeClassification` (Task 4), `CLASSIFY_SYSTEM_PROMPT` (Task 4), `ResearchState` (Task 2).
- Produces: `async def classify_node(state: ResearchState) -> dict` returning `{"mode": ..., "url": ...}` (url key present only if a URL was found in the message) — consumed by `agent.py`'s `_route_after_classify` (Task 11).

- [ ] **Step 1: Write `classify_node.py`**

Create `app/modules/research/node/classify_node.py`:

```python
import re

from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.classify import CLASSIFY_SYSTEM_PROMPT
from ..schema.mode_classification import ModeClassification
from ..state import ResearchState

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")


async def classify_node(state: ResearchState) -> dict:
    """
    Classify the user's message into new_link / new_keyword / recall.

    A URL found in the message always forces new_link (unless the LLM
    classified it as recall — "did I already save this link" stays recall).
    On any LLM failure, falls back to new_link if a URL is present, else
    new_keyword — new_keyword mirrors search's current one-flow behavior.
    """
    url_match = _URL_PATTERN.search(state["user_query"])

    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ModeClassification, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ]
        )
        mode = result.mode
    except Exception as exc:
        logger.warning("classify_node_llm_call_failed", error=str(exc))
        mode = "new_link" if url_match else "new_keyword"

    if url_match and mode != "recall":
        mode = "new_link"

    output: dict = {"mode": mode}
    if url_match:
        output["url"] = url_match.group(0)
    return output
```

- [ ] **Step 2: Verify fallback behavior**

Run (works with no LLM provider configured, since the LLM call then fails and the fallback branch is exercised deterministically):

`uv run python -c "
import asyncio
from app.modules.research.node.classify_node import classify_node

async def main():
    r1 = await classify_node({'user_id': 'u', 'user_query': 'Đọc giúp tôi https://example.com/a', 'documents': [], 'reply': ''})
    r2 = await classify_node({'user_id': 'u', 'user_query': 'Tìm giúp tôi thông tin về AI 2026', 'documents': [], 'reply': ''})
    print(r1)
    print(r2)

asyncio.run(main())
"`

Expected (if `LLM_PROVIDER`/`LLM_API_KEY` are unset or invalid in `.env`, so the LLM call raises and the fallback runs): `{'mode': 'new_link', 'url': 'https://example.com/a'}` then `{'mode': 'new_keyword'}`. If a working LLM is configured, the LLM's own classification is used instead — check the printed `mode` is one of the three valid literals and that the first result still has `'url'` set (the URL-forcing rule always applies).

- [ ] **Step 3: Commit**

```bash
git add app/modules/research/node/classify_node.py
git commit -m "feat: add research classify_node"
```

---

### Task 7: link_node and search_node

**Files:**
- Create: `app/modules/research/node/link_node.py`
- Create: `app/modules/research/node/search_node.py`

**Interfaces:**
- Consumes: `ResearchState`/`DocumentInfo` (Task 2), `tavily_search`, `SearchToolError` (Task 3).
- Produces: both return `{"documents": list[DocumentInfo]}` — consumed by `crawl_node` (Task 8) and `agent.py`'s edges (Task 11).

- [ ] **Step 1: Write `link_node.py`**

Create `app/modules/research/node/link_node.py`:

```python
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
```

- [ ] **Step 2: Write `search_node.py`**

Create `app/modules/research/node/search_node.py`:

```python
from app.utils.logger import get_logger

from ..state import ResearchState
from ..tools import SearchToolError, tavily_search

logger = get_logger(__name__)


async def search_node(state: ResearchState) -> dict:
    """
    Perform a web search based on the user's query.

    If the search fails (missing API key, network error, etc.), returns an
    empty documents list so downstream nodes can handle it gracefully.
    """
    try:
        results = await tavily_search(state["user_query"])
    except SearchToolError as exc:
        logger.warning("research_search_node_failed", error=str(exc), query=state["user_query"])
        return {"documents": []}

    documents = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "raw_text": r.get("content", "")}
        for r in results
    ]
    return {"documents": documents}
```

- [ ] **Step 3: Verify**

Run: `uv run python -c "
import asyncio
from app.modules.research.node.link_node import link_node

async def main():
    r = await link_node({'user_id': 'u', 'user_query': 'q', 'url': 'https://example.com/a', 'documents': [], 'reply': ''})
    print(r)

asyncio.run(main())
"`
Expected: `{'documents': [{'title': '', 'url': 'https://example.com/a', 'raw_text': ''}]}`

Run: `uv run python -c "
import asyncio
from app.modules.research.node.search_node import search_node

async def main():
    r = await search_node({'user_id': 'u', 'user_query': 'AI trends 2026', 'documents': [], 'reply': ''})
    print(r)

asyncio.run(main())
"`
Expected: if `TAVILY_API_KEY` is unset in `.env`, prints `{'documents': []}`. If it's set, prints a `documents` list with real search results — either way, no exception should propagate.

- [ ] **Step 4: Commit**

```bash
git add app/modules/research/node/link_node.py app/modules/research/node/search_node.py
git commit -m "feat: add research link_node and search_node"
```

---

### Task 8: confirm_node and crawl_node

**Files:**
- Create: `app/modules/research/node/confirm_node.py`
- Create: `app/modules/research/node/crawl_node.py`

**Interfaces:**
- Consumes: `ResearchState` (Task 2), `app.core.hitl.HumanReviewRequest` / `langgraph.types.interrupt` (existing), `jina_fetch`, `CrawlToolError` (Task 3).
- Produces: `async def confirm_node(state: ResearchState) -> dict` returning `{}` or `{"skip_crawl": bool}`; `async def crawl_node(state: ResearchState) -> dict` returning `{"documents": list[DocumentInfo]}` — both consumed by `agent.py`'s edges (Task 11).

- [ ] **Step 1: Write `confirm_node.py`**

Create `app/modules/research/node/confirm_node.py`:

```python
from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..state import ResearchState


def _wants_full_crawl(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized == "yes" or "có" in normalized


async def confirm_node(state: ResearchState) -> dict:
    """
    Ask whether to crawl full page content or just use search snippets.

    Only reached from the new_keyword flow (link_node always crawls, since
    the user gave an explicit single link with nothing to compare against).
    If there's nothing to confirm about (search returned no documents),
    skip the question entirely.
    """
    documents = state.get("documents", [])
    if not documents:
        return {}

    answer = interrupt(
        HumanReviewRequest(
            question=(
                f"Tìm thấy {len(documents)} kết quả. "
                "Đọc chi tiết từng trang hay chỉ tóm tắt nhanh?"
            ),
            options=[
                {"label": "Đọc chi tiết", "value": "yes"},
                {"label": "Tóm tắt nhanh", "value": "no"},
            ],
        )
    )
    return {"skip_crawl": not _wants_full_crawl(answer)}
```

- [ ] **Step 2: Write `crawl_node.py`**

Create `app/modules/research/node/crawl_node.py`:

```python
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
```

- [ ] **Step 3: Verify pure-passthrough paths**

Run: `uv run python -c "
import asyncio
from app.modules.research.node.confirm_node import confirm_node
from app.modules.research.node.crawl_node import crawl_node

async def main():
    c = await confirm_node({'user_id': 'u', 'user_query': 'q', 'documents': [], 'reply': ''})
    cr = await crawl_node({'user_id': 'u', 'user_query': 'q', 'documents': [], 'reply': ''})
    print(c, cr)

asyncio.run(main())
"`
Expected: `{} {'documents': []}`

- [ ] **Step 4: Commit**

```bash
git add app/modules/research/node/confirm_node.py app/modules/research/node/crawl_node.py
git commit -m "feat: add research confirm_node and crawl_node"
```

---

### Task 9: summarize_and_save_node

**Files:**
- Create: `app/modules/research/node/summarize_save_node.py`

**Interfaces:**
- Consumes: `ResearchState` (Task 2), `ResearchSummary` (Task 4), `SUMMARY_SYSTEM_PROMPT`/`NO_RESULTS_MESSAGE`/`format_fallback_summary` (Task 4), `embed_text` (Task 1), `upsert_research_point`, `get_default_importance` (Task 3/5), `app.infra.providers.llm_client.create_llm_client()` (existing).
- Produces: `async def summarize_and_save_node(state: ResearchState) -> dict` returning `{"reply": str}` — the graph's terminal node for the `new_link`/`new_keyword` flows (Task 11).

- [ ] **Step 1: Write `summarize_save_node.py`**

Create `app/modules/research/node/summarize_save_node.py`:

```python
from app.infra.providers.embedding import embed_text
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.summary import NO_RESULTS_MESSAGE, SUMMARY_SYSTEM_PROMPT, format_fallback_summary
from ..schema.research_summary import ResearchSummary
from ..state import ResearchState
from ..tools import get_default_importance, upsert_research_point

logger = get_logger(__name__)


async def summarize_and_save_node(state: ResearchState) -> dict:
    """
    Summarize the documents, then best-effort save the result to Qdrant.

    If no documents are available, returns NO_RESULTS_MESSAGE without any
    LLM call. If the LLM structured-output call fails, falls back to a
    plain title/URL list. If saving to Qdrant fails for any reason, logs a
    warning and still returns the summary — the user's answer must not be
    lost just because persistence failed.
    """
    documents = state.get("documents", [])
    user_query = state.get("user_query", "")

    if not documents:
        return {"reply": NO_RESULTS_MESSAGE}

    doc_content_lines = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "")
        url = doc.get("url", "")
        raw_text = doc.get("raw_text", "")
        truncated = raw_text[:2000] if raw_text else ""

        doc_content_lines.append(f"Document {i}:")
        if title:
            doc_content_lines.append(f"Title: {title}")
        if url:
            doc_content_lines.append(f"URL: {url}")
        if truncated:
            doc_content_lines.append(f"Content: {truncated}")
        doc_content_lines.append("")

    doc_block = "\n".join(doc_content_lines)
    user_message = f"Query: {user_query}\n\n{doc_block}"

    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ResearchSummary, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        reply = result.render()
        summary_text = result.answer
        topics = result.topics
    except Exception as exc:
        logger.warning("summarize_save_node_llm_call_failed", error=str(exc))
        reply = format_fallback_summary(documents)
        summary_text = reply
        topics = []

    title = documents[0].get("title") or user_query
    url = documents[0].get("url", "")

    try:
        vector = await embed_text(summary_text, prefix="passage: ")
        await upsert_research_point(
            user_id=state["user_id"],
            url=url,
            title=title,
            summary=summary_text,
            topics=topics,
            importance=get_default_importance(),
            vector=vector,
        )
    except Exception as exc:
        logger.warning("research_save_failed", error=str(exc))

    return {"reply": reply}
```

- [ ] **Step 2: Verify the no-documents path**

Run: `uv run python -c "
import asyncio
from app.modules.research.node.summarize_save_node import summarize_and_save_node
from app.modules.research.prompts.summary import NO_RESULTS_MESSAGE

async def main():
    r = await summarize_and_save_node({'user_id': 'u', 'user_query': 'q', 'documents': [], 'reply': ''})
    assert r == {'reply': NO_RESULTS_MESSAGE}, r
    print('ok')

asyncio.run(main())
"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add app/modules/research/node/summarize_save_node.py
git commit -m "feat: add research summarize_and_save_node"
```

---

### Task 10: recall_node

**Files:**
- Create: `app/modules/research/node/recall_node.py`

**Interfaces:**
- Consumes: `ResearchState` (Task 2), `NO_RECALL_RESULTS_MESSAGE`/`format_recall_results` (Task 4), `embed_text` (Task 1), `search_research_points`, `get_recall_top_k`, `get_recall_min_score` (Task 3/5).
- Produces: `async def recall_node(state: ResearchState) -> dict` returning `{"reply": str}` — the graph's terminal node for the `recall` flow (Task 11).

- [ ] **Step 1: Write `recall_node.py`**

Create `app/modules/research/node/recall_node.py`:

```python
from app.infra.providers.embedding import embed_text
from app.utils.logger import get_logger

from ..prompts.recall import NO_RECALL_RESULTS_MESSAGE, format_recall_results
from ..state import ResearchState
from ..tools import get_recall_min_score, get_recall_top_k, search_research_points

logger = get_logger(__name__)


async def recall_node(state: ResearchState) -> dict:
    """
    Answer a recall query by semantic search over previously saved research.

    Any failure (Qdrant unreachable/uninitialized, embedding error) is
    treated the same as "nothing found" — logs a warning and returns the
    canned no-results message rather than raising.
    """
    try:
        vector = await embed_text(state["user_query"], prefix="query: ")
        results = await search_research_points(
            user_id=state["user_id"],
            query_vector=vector,
            top_k=get_recall_top_k(),
            min_score=get_recall_min_score(),
        )
    except Exception as exc:
        logger.warning("recall_node_failed", error=str(exc))
        results = []

    if not results:
        return {"reply": NO_RECALL_RESULTS_MESSAGE}

    return {"reply": format_recall_results(results)}
```

- [ ] **Step 2: Verify graceful failure when Qdrant isn't initialized**

Run: `uv run python -c "
import asyncio
from app.modules.research.node.recall_node import recall_node
from app.modules.research.prompts.recall import NO_RECALL_RESULTS_MESSAGE

async def main():
    r = await recall_node({'user_id': 'u', 'user_query': 'blockchain', 'documents': [], 'reply': ''})
    print(r)

asyncio.run(main())
"`
Expected: `{'reply': 'Chưa có gì được lưu về chủ đề này. Hãy gửi link hoặc từ khóa để tôi nghiên cứu và lưu lại.'}` — since this script never calls `init_vector_db()`, `get_qdrant_client()` raises `RuntimeError`, which `recall_node` catches and turns into the canned message. (The embedding model still loads for real, so this may take a moment on first run.)

- [ ] **Step 3: Commit**

```bash
git add app/modules/research/node/recall_node.py
git commit -m "feat: add research recall_node"
```

---

### Task 11: agent.py wiring and registry verification

**Files:**
- Modify: `app/modules/research/node/__init__.py`
- Create: `app/modules/research/agent.py`

**Interfaces:**
- Consumes: every node from Tasks 6–10, `ResearchState` (Task 2), `config` (Task 2's `config.yaml`), `app.core.base_agent.{AgentInput, AgentOutput, BaseAgent}` (existing), `app.core.hitl.run_interruptible_subgraph` (existing), `app.infra.memory.checkpointer.get_checkpointer` (existing).
- Produces: module-level `agent = ResearchAgent()` and `graph` — the contract `app/orchestrator/registry.py:discover_and_register()` picks up automatically (no registry code changes needed, matching how `search` was wired).

- [ ] **Step 1: Export nodes from `node/__init__.py`**

Replace the contents of `app/modules/research/node/__init__.py`:

```python
from .classify_node import classify_node
from .confirm_node import confirm_node
from .crawl_node import crawl_node
from .link_node import link_node
from .recall_node import recall_node
from .search_node import search_node
from .summarize_save_node import summarize_and_save_node

__all__ = [
    "classify_node",
    "confirm_node",
    "crawl_node",
    "link_node",
    "recall_node",
    "search_node",
    "summarize_and_save_node",
]
```

- [ ] **Step 2: Write `agent.py`**

Create `app/modules/research/agent.py`:

```python
from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer

from .node import (
    classify_node,
    confirm_node,
    crawl_node,
    link_node,
    recall_node,
    search_node,
    summarize_and_save_node,
)
from .state import ResearchState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=ResearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("classify", classify_node)
builder.add_node("link", link_node)
builder.add_node("search", search_node)
builder.add_node("confirm", confirm_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summarize_and_save", summarize_and_save_node)
builder.add_node("recall", recall_node)

builder.set_entry_point("classify")


def _route_after_classify(state: ResearchState) -> str:
    mode = state.get("mode", "new_keyword")
    if mode == "new_link":
        return "link"
    if mode == "recall":
        return "recall"
    return "search"


builder.add_conditional_edges("classify", _route_after_classify, ["link", "search", "recall"])
builder.add_edge("link", "crawl")
builder.add_edge("search", "confirm")


def _route_after_confirm(state: ResearchState) -> str:
    if not state.get("documents"):
        return "summarize_and_save"
    return "summarize_and_save" if state.get("skip_crawl") else "crawl"


builder.add_conditional_edges("confirm", _route_after_confirm, ["crawl", "summarize_and_save"])
builder.add_edge("crawl", "summarize_and_save")
builder.add_edge("summarize_and_save", END)
builder.add_edge("recall", END)

graph = builder.compile(checkpointer=get_checkpointer())


class ResearchAgent(BaseAgent):
    """Wraps the classify/link/search/confirm/crawl/summarize/recall subgraph behind BaseAgent."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: ResearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": [],
            "reply": "",
        }
        result = await run_interruptible_subgraph(
            graph,
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}


agent = ResearchAgent()
```

- [ ] **Step 3: Start Redis (needed for the checkpointer this module compiles against at import time)**

Run: `docker compose up -d redis`
Expected: container starts (or is already running).

- [ ] **Step 4: Verify the graph builds and the registry auto-discovers it**

Run:

```bash
uv run python -c "
import asyncio
from app.infra.memory.checkpointer import init_checkpointer

async def main():
    await init_checkpointer()
    from app.modules.research.agent import agent, graph
    print(agent.name)
    print(sorted(graph.nodes.keys()))

    from app.orchestrator import registry
    registry.discover_and_register()
    print(sorted(registry.all_agents().keys()))

asyncio.run(main())
"
```

Expected: prints `research`, then a node list including `classify`, `link`, `search`, `confirm`, `crawl`, `summarize_and_save`, `recall` (plus LangGraph's internal `__start__`), then a registered-agents list containing both `search` and `research`.

- [ ] **Step 5: Commit**

```bash
git add app/modules/research/node/__init__.py app/modules/research/agent.py
git commit -m "feat: wire research agent graph"
```

---

## Post-implementation note

This plan intentionally leaves the following out of scope (per the design spec's "Open items deferred" section) — do not add them speculatively:
- Hybrid dense+sparse search / cross-encoder re-ranking.
- Importance decay / weekly compression jobs.
- A structured Postgres index for research items.
