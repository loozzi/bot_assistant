# HITL Subgraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give LangGraph's `interrupt()`/`Command(resume=...)` mechanism a working home in this codebase — a generic bridge helper any subgraph-style module can use to pause mid-turn for human input via Telegram (button or free-text), plus a proof-of-concept in `search` (confirm before crawling).

**Architecture:** Outer orchestrator graph and each module subgraph share one Redis-backed `AsyncRedisSaver` checkpointer, keyed by distinct thread IDs (`user_id` for the outer graph, `f"{user_id}:{agent_name}"` per subgraph). A reusable helper (`app/core/hitl.py::run_interruptible_subgraph`) bridges a subgraph's own pause up to the parent graph's checkpoint via a second `interrupt()` call, so the whole turn (not just the subgraph) can pause across two separate Telegram messages without losing subgraph progress on resume.

**Tech Stack:** LangGraph 1.2.6 (`interrupt`, `Command`, `AsyncRedisSaver` from `langgraph-checkpoint-redis`), Redis 8 (RediSearch/RedisJSON required by the checkpoint library), aiogram 3.

## Global Constraints

- **Do not run `git commit` at any point in this plan.** The user asked to implement without auto-committing — every task ends with the working tree left as-is for manual review/commit, not a commit step.
- **No pytest in this repo** (per CLAUDE.md — `tests/` has only empty `__init__.py`, no `[tool.pytest]` config). Each task's "test" step is a standalone verification script run via `uv run python`, matching how the existing `search` module's nodes were built without a test suite. Do not add `pytest` or any test config as a side effect of this plan.
- **Python 3.11+ is a hard requirement** — `interrupt()` is unsafe on 3.10 in an async graph (empirically confirmed; see `docs/superpowers/specs/2026-07-15-hitl-subgraph-design.md` amendment). Already applied in Task 1.
- **Redis 8+ (RediSearch/RedisJSON) is a hard requirement** for `AsyncRedisSaver` — plain `redis:7-alpine` fails with `unknown command 'FT._LIST'`. Already applied in Task 1.
- All new/modified `.env`-driven config must go through `app/config/settings.py`'s `Settings` — no ad-hoc `os.environ` reads (matches existing convention).
- Follow the vertical-slice import rules from CLAUDE.md: `modules/` never imports another `modules/*`; `infra/` never imports `modules/*`.

---

### Task 1: Runtime prerequisites — Python 3.11 + Redis 8 (already applied, verify only)

**Files:**
- Modify: `.python-version` (already changed: `3.10` → `3.11`)
- Modify: `pyproject.toml` (already changed: `requires-python = ">=3.11"`, added `langgraph-checkpoint-redis>=0.5.1` to `dependencies`)
- Modify: `docker-compose.yml:21` (already changed: `redis:7-alpine` → `redis:8-alpine`)

**Interfaces:**
- Produces: a `.venv` running Python 3.11.x with `langgraph-checkpoint-redis` installed; a local `redis:8-alpine` container reachable on the port from `.env`'s `REDIS_PORT`.

- [x] **Step 1: Confirm the interpreter and dependency changes are in place**

Run: `cat .python-version && grep requires-python pyproject.toml && grep langgraph-checkpoint-redis pyproject.toml`
Expected output:
```
3.11
requires-python = ">=3.11"
    "langgraph-checkpoint-redis>=0.5.1",
```

- [x] **Step 2: Confirm `uv sync` resolves cleanly on 3.11**

Run: `uv run python --version`
Expected: `Python 3.11.15` (or any 3.11.x)

- [x] **Step 3: Confirm the Redis image change**

Run: `grep -A1 "redis:" docker-compose.yml | head -3`
Expected: `image: redis:8-alpine`

- [ ] **Step 4: Recreate the local Redis container on the new image**

Run: `docker compose up -d --force-recreate redis`
Expected: container `secondbrain_redis` recreated and healthy (`docker compose ps` shows `Up ... (healthy)`). The existing `redis_data` volume carries over — no data loss, Redis 8 reads Redis 7's RDB/AOF format.

- [ ] **Step 5: Confirm RediSearch/RedisJSON are present on the running container**

Run: `docker exec secondbrain_redis redis-cli -a "$(grep REDIS_PASSWORD .env | cut -d= -f2)" MODULE LIST`
Expected: a list including modules named `search` and `ReJSON` (exact format: repeated `name`/`ver`/`path` blocks — presence of `search` and `ReJSON` is what matters, not exact versions).

No commit — this task only touches config files already staged from the design/validation phase; leave them as-is for the final review.

---

### Task 2: Checkpointer infra — `app/infra/memory/checkpointer.py`

**Files:**
- Create: `app/infra/memory/checkpointer.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `app.config.settings.Settings` (`redis_ttl_session: int`, seconds), `app.infra.memory.working.get_redis_client() -> aioredis.Redis` (must already be initialized — this module never opens its own Redis connection).
- Produces: `init_checkpointer(settings: Settings | None = None) -> None`, `close_checkpointer() -> None`, `get_checkpointer() -> AsyncRedisSaver` — used by `app/core/hitl.py` (Task 3), `app/orchestrator/graph.py` (Task 4), and `app/modules/search/agent.py` (Task 5).

- [ ] **Step 1: Write the verification script**

Create `/tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_checkpointer.py`:

```python
import asyncio

from app.config.settings import get_settings
from app.infra.memory.checkpointer import close_checkpointer, get_checkpointer, init_checkpointer
from app.infra.memory.working import close_redis, init_redis


async def main() -> None:
    settings = get_settings()
    await init_redis(settings)
    await init_checkpointer(settings)

    saver = get_checkpointer()
    print("ttl_config:", saver.ttl_config)
    assert saver.ttl_config == {"default_ttl": settings.redis_ttl_session // 60, "refresh_on_read": True}

    await close_checkpointer()
    await close_redis()
    print("VERIFY_CHECKPOINTER_OK")


asyncio.run(main())
```

- [ ] **Step 2: Run it to confirm it fails (module doesn't exist yet)**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_checkpointer.py`
Expected: `ModuleNotFoundError: No module named 'app.infra.memory.checkpointer'`

- [ ] **Step 3: Create `app/infra/memory/checkpointer.py`**

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config.settings import Settings, get_settings
from app.infra.memory.working import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_saver: AsyncRedisSaver | None = None


async def init_checkpointer(settings: Settings | None = None) -> None:
    global _saver

    if _saver is not None:
        return

    cfg = settings or get_settings()
    saver = AsyncRedisSaver(
        redis_client=get_redis_client(),
        ttl={"default_ttl": cfg.redis_ttl_session // 60, "refresh_on_read": True},
    )
    await saver.asetup()
    _saver = saver
    logger.info("checkpointer_initialized")


async def close_checkpointer() -> None:
    global _saver

    if _saver is None:
        return

    _saver = None
    logger.info("checkpointer_closed")


def get_checkpointer() -> AsyncRedisSaver:
    if _saver is None:
        raise RuntimeError("Checkpointer not initialized — call init_checkpointer() first")
    return _saver
```

`close_checkpointer()` doesn't close a connection itself — `AsyncRedisSaver` reuses the shared client from `infra/memory/working.py`, which `close_redis()` already tears down. It just drops the module-level reference, mirroring `working.py`'s own shape.

- [ ] **Step 4: Run the verification script to confirm it passes**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_checkpointer.py`
Expected: prints `ttl_config: {'default_ttl': 240, 'refresh_on_read': True}` then `VERIFY_CHECKPOINTER_OK`. Requires the Task 1 Redis container to be reachable per `.env`.

- [ ] **Step 5: Wire lifecycle into `app/main.py`**

In `app/main.py`, add the import next to the other `infra` imports:

```python
from app.infra.memory.checkpointer import close_checkpointer, init_checkpointer
```

In `on_startup`, insert `init_checkpointer` **after** `init_redis` and **before** `registry.discover_and_register()` (module `agent.py` files compile their subgraph — needing the checkpointer — at import time, which `discover_and_register()` triggers):

```python
async def on_startup(bot: Bot) -> None:
    settings = get_settings()

    await init_db(settings)
    await init_redis(settings)
    await init_checkpointer(settings)
    await init_vector_db(settings)

    registry.discover_and_register()
    ...
```

In `on_shutdown`, add `close_checkpointer()` before `close_redis()`:

```python
async def on_shutdown(bot: Bot) -> None:
    ...
    await close_checkpointer()
    await close_redis()
    await close_vector_db()
    await close_db()
    ...
```

No commit.

---

### Task 3: Shared HITL contract + bridge helper — `app/core/hitl.py`

**Files:**
- Create: `app/core/hitl.py`

**Interfaces:**
- Consumes: `app.infra.memory.checkpointer` only indirectly — this file takes an already-compiled `CompiledStateGraph` as a parameter, it does not import the checkpointer module itself.
- Produces: `HumanReviewOption` (TypedDict: `label: str`, `value: str`), `HumanReviewRequest` (TypedDict: `question: str`, `options: NotRequired[list[HumanReviewOption]]`), `run_interruptible_subgraph(graph, initial_state, *, thread_id) -> dict[str, Any]` — used by `app/modules/search/node/confirm_node.py` and `app/modules/search/agent.py` (Task 5), and by `app/bot/keyboards.py` (Task 6).

Deliberate deviation from CLAUDE.md's "`app/core/` holds contracts only" rule — the bridge helper is colocated with its contracts here for simplicity (explicit user direction); note this exception in CLAUDE.md once the whole plan lands.

- [ ] **Step 1: Write the verification script**

Create `/tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_hitl_bridge.py`:

```python
import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.core.hitl import HumanReviewRequest, run_interruptible_subgraph

saver = InMemorySaver()


class ChildState(TypedDict):
    documents: list
    skip_crawl: bool
    summary: str


async def search_node(state):
    return {"documents": ["doc1", "doc2"]}


async def confirm_node(state):
    answer = interrupt(HumanReviewRequest(question="Crawl full pages?", options=[
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]))
    return {"skip_crawl": answer != "yes"}


async def crawl_node(state):
    return {"documents": [d + "-crawled" for d in state["documents"]]}


async def summary_node(state):
    return {"summary": f"summary of {state['documents']}"}


def route_after_confirm(state):
    return "summary" if state.get("skip_crawl") else "crawl"


child = StateGraph(ChildState)
child.add_node("search", search_node)
child.add_node("confirm", confirm_node)
child.add_node("crawl", crawl_node)
child.add_node("summary", summary_node)
child.set_entry_point("search")
child.add_edge("search", "confirm")
child.add_conditional_edges("confirm", route_after_confirm, ["crawl", "summary"])
child.add_edge("crawl", "summary")
child.add_edge("summary", END)
child_graph = child.compile(checkpointer=saver)


class ParentState(TypedDict):
    user_id: str
    agent_reply: str


async def dispatch_agent(state):
    result = await run_interruptible_subgraph(
        child_graph,
        {"documents": [], "skip_crawl": False, "summary": ""},
        thread_id=f"{state['user_id']}:search",
    )
    return {"agent_reply": result["summary"]}


parent = StateGraph(ParentState)
parent.add_node("dispatch_agent", dispatch_agent)
parent.set_entry_point("dispatch_agent")
parent.add_edge("dispatch_agent", END)
parent_graph = parent.compile(checkpointer=saver)


async def main():
    config = {"configurable": {"thread_id": "user1"}}
    r1 = await parent_graph.ainvoke({"user_id": "user1", "agent_reply": ""}, config=config)
    assert "__interrupt__" in r1, r1

    r2 = await parent_graph.ainvoke(Command(resume="no"), config=config)
    assert r2["agent_reply"] == "summary of ['doc1', 'doc2']", r2  # crawl skipped

    config2 = {"configurable": {"thread_id": "user2"}}
    r3 = await parent_graph.ainvoke({"user_id": "user2", "agent_reply": ""}, config=config2)
    assert "__interrupt__" in r3, r3
    r4 = await parent_graph.ainvoke(Command(resume="yes"), config=config2)
    assert r4["agent_reply"] == "summary of ['doc1-crawled', 'doc2-crawled']", r4  # crawl ran

    print("VERIFY_HITL_BRIDGE_OK")


asyncio.run(main())
```

- [ ] **Step 2: Run it to confirm it fails (module doesn't exist yet)**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_hitl_bridge.py`
Expected: `ModuleNotFoundError: No module named 'app.core.hitl'`

- [ ] **Step 3: Create `app/core/hitl.py`**

```python
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import NotRequired, TypedDict


class HumanReviewOption(TypedDict):
    label: str
    value: str


class HumanReviewRequest(TypedDict):
    question: str
    options: NotRequired[list[HumanReviewOption]]


async def run_interruptible_subgraph(
    graph: CompiledStateGraph,
    initial_state: dict[str, Any],
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Run a checkpointed subgraph, bridging any pause up to the parent graph.

    Must be awaited from inside a node of a checkpointed parent graph — the
    bridging `interrupt()` call below relies on that parent's checkpoint
    context. Supports exactly one pending pause per invocation; a subgraph
    that needs to pause a second time within the same run isn't supported yet.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot.next:
        # Paused by a previous attempt at this same call site.
        pending = snapshot.interrupts[0].value
        answer = interrupt(pending)  # returns instantly on replay, else re-pauses the parent
        result = await graph.ainvoke(Command(resume=answer), config=config)
    else:
        result = await graph.ainvoke(initial_state, config=config)

    snapshot = await graph.aget_state(config)
    if snapshot.next:
        # Just paused for the first time — bridge it up to the parent graph.
        interrupt(snapshot.interrupts[0].value)

    return result
```

- [ ] **Step 4: Run the verification script to confirm it passes**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_hitl_bridge.py`
Expected: `VERIFY_HITL_BRIDGE_OK`

No commit.

---

### Task 4: Outer graph checkpointer wiring — `app/orchestrator/graph.py`

**Files:**
- Modify: `app/orchestrator/graph.py:142-149` (the `_compiled`/`get_compiled_graph` block)

**Interfaces:**
- Consumes: `app.infra.memory.checkpointer.get_checkpointer()` (Task 2).
- Produces: `get_compiled_graph()` now returns a graph compiled with a checkpointer — `bot/handlers.py` (Task 6) relies on this for `graph.aget_state(config)` / `Command(resume=...)` to work at the outer level.

- [ ] **Step 1: Write the verification script**

Create `/tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_outer_checkpointed.py`:

```python
import asyncio

from app.config.settings import get_settings
from app.infra.memory.checkpointer import close_checkpointer, init_checkpointer
from app.infra.memory.working import close_redis, init_redis
from app.orchestrator.graph import get_compiled_graph


async def main() -> None:
    settings = get_settings()
    await init_redis(settings)
    await init_checkpointer(settings)

    graph = get_compiled_graph()
    assert graph.checkpointer is not None, "outer graph must be compiled with a checkpointer"

    config = {"configurable": {"thread_id": "verify-outer-user"}}
    snapshot = await graph.aget_state(config)
    print("fresh thread snapshot.next:", snapshot.next)
    assert snapshot.next == ()

    await close_checkpointer()
    await close_redis()
    print("VERIFY_OUTER_CHECKPOINTED_OK")


asyncio.run(main())
```

- [ ] **Step 2: Run it to confirm it fails (`graph.checkpointer` is `None` today)**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_outer_checkpointed.py`
Expected: `AssertionError: outer graph must be compiled with a checkpointer`

- [ ] **Step 3: Modify `app/orchestrator/graph.py`**

Add the import at the top alongside the other `app.*` imports:

```python
from app.infra.memory.checkpointer import get_checkpointer
```

Change the compile call:

```python
_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=get_checkpointer())
    return _compiled
```

(Only the `.compile()` line changes — `build_graph()` itself is untouched.)

- [ ] **Step 4: Run the verification script to confirm it passes**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_outer_checkpointed.py`
Expected: prints `fresh thread snapshot.next: ()` then `VERIFY_OUTER_CHECKPOINTED_OK`

No commit.

---

### Task 5: Search module PoC — confirm-before-crawl

**Files:**
- Modify: `app/modules/search/state.py`
- Create: `app/modules/search/node/confirm_node.py`
- Modify: `app/modules/search/node/__init__.py`
- Modify: `app/modules/search/agent.py`

**Interfaces:**
- Consumes: `app.core.hitl.HumanReviewRequest`, `run_interruptible_subgraph` (Task 3); `app.infra.memory.checkpointer.get_checkpointer` (Task 2).
- Produces: `SearchAgent.run()` keeps its existing `BaseAgent` signature (`AgentInput -> AgentOutput`) — no contract change visible to the orchestrator.

- [ ] **Step 1: Modify `app/modules/search/state.py`**

```python
from typing import TypedDict

from typing_extensions import NotRequired


class DocumentInfo(TypedDict):
    title: str
    url: str
    raw_text: str


class SearchState(TypedDict):
    user_id: str
    user_query: str
    documents: list[DocumentInfo]
    summary: str
    skip_crawl: NotRequired[bool]
```

- [ ] **Step 2: Create `app/modules/search/node/confirm_node.py`**

```python
from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..state import SearchState


def _wants_full_crawl(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized == "yes" or "có" in normalized


async def confirm_node(state: SearchState) -> dict:
    """
    Ask the user whether to crawl full page content or just use search snippets.

    If there's nothing to confirm about (search returned no documents), skip
    the question entirely — crawling would have nothing to do either way.
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

- [ ] **Step 3: Modify `app/modules/search/node/__init__.py`**

```python
from .confirm_node import confirm_node
from .crawl_node import crawl_node
from .search_node import search_node
from .summary_node import summary_node


__all__ = ["confirm_node", "crawl_node", "search_node", "summary_node"]
```

- [ ] **Step 4: Modify `app/modules/search/agent.py`**

```python
from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer

from .node import confirm_node, crawl_node, search_node, summary_node
from .state import SearchState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=SearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("search", search_node)
builder.add_node("confirm", confirm_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summary", summary_node)

builder.set_entry_point("search")
builder.add_edge("search", "confirm")


def _route_after_confirm(state: SearchState) -> str:
    if not state.get("documents"):
        return "summary"
    return "summary" if state.get("skip_crawl") else "crawl"


builder.add_conditional_edges("confirm", _route_after_confirm, ["crawl", "summary"])
builder.add_edge("crawl", "summary")
builder.add_edge("summary", END)

graph = builder.compile(checkpointer=get_checkpointer())


class SearchAgent(BaseAgent):
    """Wraps the search/crawl/summary subgraph behind the BaseAgent contract."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: SearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": [],
            "summary": "",
        }
        result = await run_interruptible_subgraph(
            graph,
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["summary"]}


agent = SearchAgent()
```

- [ ] **Step 5: Write the verification script**

Requires `TAVILY_API_KEY`/`JINA_API_KEY`/LLM settings in `.env` (already present in this repo) and Redis reachable (Task 1). Create `/tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_search_poc.py`:

```python
import asyncio

from langgraph.types import Command

from app.config.settings import get_settings
from app.infra.memory.checkpointer import close_checkpointer, init_checkpointer
from app.infra.memory.working import close_redis, init_redis
from app.modules.search.agent import graph


async def main() -> None:
    settings = get_settings()
    await init_redis(settings)
    await init_checkpointer(settings)

    thread_id = "verify-search-poc:search"
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "user_id": "verify-search-poc",
        "user_query": "thời tiết Hà Nội hôm nay",
        "documents": [],
        "summary": "",
    }
    r1 = await graph.ainvoke(initial, config=config)
    assert "__interrupt__" in r1, f"expected a pause, got: {r1}"
    payload = r1["__interrupt__"][0].value
    print("interrupt question:", payload["question"])
    assert payload["options"] == [
        {"label": "Đọc chi tiết", "value": "yes"},
        {"label": "Tóm tắt nhanh", "value": "no"},
    ]

    r2 = await graph.ainvoke(Command(resume="no"), config=config)
    assert "summary" in r2 and r2["summary"], f"expected a summary, got: {r2}"
    print("summary (crawl skipped):", r2["summary"][:200])

    await close_checkpointer()
    await close_redis()
    print("VERIFY_SEARCH_POC_OK")


asyncio.run(main())
```

- [ ] **Step 6: Run it to confirm it fails (confirm node doesn't exist yet before Step 2-4 land)**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_search_poc.py`
Expected (before Steps 2-4 applied): `ImportError: cannot import name 'confirm_node'` — if Steps 1-4 are already applied by the time this runs, skip straight to Step 7.

- [ ] **Step 7: Run the verification script to confirm it passes**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_search_poc.py`
Expected: prints the Vietnamese question, then a non-empty summary, then `VERIFY_SEARCH_POC_OK`. If `TAVILY_API_KEY` is invalid/rate-limited, `documents` may come back empty — in that case `r1` won't contain `__interrupt__` (confirm_node's own graceful no-documents skip) and this script's first assertion will fail with a clear message; that's a live-API issue, not a code bug, and doesn't block the rest of the plan — re-run once the key issue is resolved, or verify manually via Task 7's fuller script instead.

No commit.

---

### Task 6: Bot handlers — resume flow, `/cancel`, HITL keyboard

**Files:**
- Modify: `app/bot/keyboards.py`
- Modify: `app/bot/handlers.py`

**Interfaces:**
- Consumes: `app.core.hitl.HumanReviewOption` (Task 3), `app.orchestrator.graph.get_compiled_graph()` (Task 4), `app.orchestrator.registry.all_agents()`, `app.infra.memory.checkpointer.get_checkpointer()` (Task 2).
- Produces: user-facing `/cancel` command, `hitl:<value>` and `hitl_cancel` callback routes.

- [ ] **Step 1: Add `hitl_keyboard` to `app/bot/keyboards.py`**

Add the import at the top:

```python
from app.core.hitl import HumanReviewOption
```

Add the function (anywhere after the existing keyboard builders):

```python
def hitl_keyboard(options: list[HumanReviewOption]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for option in options:
        builder.button(text=option["label"], callback_data=f"hitl:{option['value']}")
    builder.button(text="Huỷ", callback_data="hitl_cancel")
    builder.adjust(2)
    return builder.as_markup()
```

- [ ] **Step 2: Verify the keyboard builder in isolation**

Run:
```bash
uv run python -c "
from app.bot.keyboards import hitl_keyboard
kb = hitl_keyboard([{'label': 'Đọc chi tiết', 'value': 'yes'}, {'label': 'Tóm tắt nhanh', 'value': 'no'}])
rows = [[b.text for b in row] for row in kb.inline_keyboard]
print(rows)
callback_data = [[b.callback_data for b in row] for row in kb.inline_keyboard]
print(callback_data)
"
```
Expected: two rows total (adjust(2) packs 3 buttons as 2+1), first row `['Đọc chi tiết', 'Tóm tắt nhanh']`, callback_data `[['hitl:yes', 'hitl:no'], ['hitl_cancel']]`.

- [ ] **Step 3: Rewrite `app/bot/handlers.py`**

Full file:

```python
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from langchain_core.messages import HumanMessage
from langgraph.types import Command as ResumeCommand

from app.bot.keyboards import hitl_keyboard, main_menu_keyboard
from app.core.state import AgentState
from app.infra.memory.checkpointer import get_checkpointer
from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = Router(name="main")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message, user_id: str) -> None:
    await message.answer(
        "👋 Xin chào! Mình là trợ lý cá nhân của bạn.\n\n"
        "Bạn có thể:\n"
        "• 📓 Ghi nhật ký & cảm xúc\n"
        "• 💰 Theo dõi chi tiêu\n"
        "• 🔍 Tìm kiếm thông tin\n"
        "• 📊 Xem phân tích xu hướng\n\n"
        "Hoặc chỉ cần nhắn tin tự nhiên — mình sẽ hiểu!",
        reply_markup=main_menu_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 <b>Hướng dẫn sử dụng</b>\n\n"
        "<b>Nhật ký:</b> Kể về ngày của bạn, cảm xúc, suy nghĩ\n"
        "<b>Chi tiêu:</b> 'Mua cà phê 45k', 'Tốn 200k tiền ăn'\n"
        "<b>Tìm kiếm:</b> 'Tìm ...', 'Search ...'\n"
        "<b>Phân tích:</b> 'Phân tích chi tiêu tháng này'\n\n"
        "Dùng /menu để xem menu chính. Dùng /cancel để huỷ câu hỏi đang chờ.",
        parse_mode="HTML",
    )


@router.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    await message.answer("Chọn chức năng:", reply_markup=main_menu_keyboard())


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, user_id: str) -> None:
    await _clear_pending_review(user_id)
    await message.answer("Đã huỷ yêu cầu trước đó. Bạn cần gì tiếp theo?")


# ---------------------------------------------------------------------------
# Shared turn-running helpers
# ---------------------------------------------------------------------------

async def _clear_pending_review(user_id: str) -> None:
    checkpointer = get_checkpointer()
    await checkpointer.adelete_thread(user_id)
    for agent_name in registry.all_agents():
        await checkpointer.adelete_thread(f"{user_id}:{agent_name}")


async def _run_turn(
    user_id: str,
    *,
    new_text: str | None = None,
    resume_answer: str | None = None,
) -> AgentState:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": user_id}}

    if resume_answer is not None:
        return await graph.ainvoke(ResumeCommand(resume=resume_answer), config=config)

    state: AgentState = {
        "messages": [HumanMessage(content=new_text)],
        "user_id": user_id,
    }
    return await graph.ainvoke(state, config=config)


async def _reply_or_prompt(send, result: AgentState) -> None:
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        options = payload.get("options")
        keyboard = hitl_keyboard(options) if options else None
        await send(payload["question"], reply_markup=keyboard)
        return

    last = next(
        (m for m in reversed(result["messages"]) if getattr(m, "type", None) == "ai"),
        None,
    )
    reply = last.content if last else "Mình chưa có câu trả lời cho điều này."
    await send(reply)


# ---------------------------------------------------------------------------
# Main message handler — delegates to orchestrator
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_text(message: Message, user_id: str) -> None:
    assert message.text is not None

    thinking = await message.answer("⏳ Đang xử lý...")

    try:
        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": user_id}}
        snapshot = await graph.aget_state(config)

        if snapshot.next:
            result = await _run_turn(user_id, resume_answer=message.text)
        else:
            result = await _run_turn(user_id, new_text=message.text)

        await thinking.delete()
        await _reply_or_prompt(message.answer, result)

    except Exception as exc:
        logger.exception("orchestrator_error", user_id=user_id, error=str(exc))
        await thinking.delete()
        await message.answer("❌ Có lỗi xảy ra. Bạn thử lại sau nhé!")


# ---------------------------------------------------------------------------
# Callback queries
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "cancel")
async def cb_cancel(callback: CallbackQuery) -> None:
    await callback.message.delete()  # type: ignore[union-attr]
    await callback.answer("Đã huỷ.")


@router.callback_query(F.data == "confirm")
async def cb_confirm(callback: CallbackQuery) -> None:
    await callback.answer("Đã xác nhận.")


@router.callback_query(F.data.startswith("menu:"))
async def cb_menu(callback: CallbackQuery, user_id: str) -> None:
    module = callback.data.split(":")[1]  # type: ignore[union-attr]
    prompts = {
        "journal": "Hãy kể về ngày hôm nay của bạn.",
        "finance": "Bạn muốn ghi chi tiêu gì?",
        "search": "Bạn muốn tìm kiếm gì?",
        "insight": "Bạn muốn phân tích gì?",
    }
    await callback.message.answer(prompts.get(module, "Bạn cần gì?"))  # type: ignore[union-attr]
    await callback.answer()


@router.callback_query(F.data == "hitl_cancel")
async def cb_hitl_cancel(callback: CallbackQuery, user_id: str) -> None:
    await _clear_pending_review(user_id)
    await callback.message.edit_text("Đã huỷ yêu cầu trước đó.")  # type: ignore[union-attr]
    await callback.answer()


@router.callback_query(F.data.startswith("hitl:"))
async def cb_hitl(callback: CallbackQuery, user_id: str) -> None:
    value = callback.data.split(":", 1)[1]  # type: ignore[union-attr]
    await callback.answer()

    try:
        result = await _run_turn(user_id, resume_answer=value)
        await callback.message.delete()  # type: ignore[union-attr]
        await _reply_or_prompt(callback.message.answer, result)  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("orchestrator_error", user_id=user_id, error=str(exc))
        await callback.message.answer("❌ Có lỗi xảy ra. Bạn thử lại sau nhé!")  # type: ignore[union-attr]
```

- [ ] **Step 4: Sanity-import the module**

Run: `uv run python -c "import app.bot.handlers; print('OK')"`
Expected: `OK` (no import errors — this only checks syntax/imports, not runtime behavior; full behavior is exercised in Task 7).

No commit.

---

### Task 6.5: Bug found via live Telegram testing — per-user turn serialization

**Discovered post-implementation**, via real Telegram usage: sending two messages within ~7s of each other (before the first search+LLM turn finished) reliably crashed both with `IndexError: tuple index out of range` at `app/core/hitl.py:36`.

**Root cause (confirmed by reproduction script, not guessed):** nothing serialized graph turns per `user_id`. Two concurrent `handle_text` invocations for the same user both call `graph.ainvoke()`/`aget_state()` on the *same* Redis checkpoint `thread_id` at once, racing on reads/writes and corrupting the checkpoint — one call reads `snapshot.next` truthy (written by the other call) before that other call's `Interrupt` object is flushed, so `snapshot.interrupts` is still empty and `snapshot.interrupts[0]` throws.

**Fix:** one `asyncio.Lock` per `user_id` (module-level dict in `app/bot/handlers.py`), guarding every block that touches the graph or checkpointer (`handle_text`, `cmd_cancel`, `cb_hitl`, `cb_hitl_cancel`). Different users remain fully concurrent; the same user's turns now strictly serialize.

- [x] Reproduced with a concurrent-`asyncio.gather` script hitting the exact same traceback as production.
- [x] Applied the lock fix in `app/bot/handlers.py`.
- [x] Re-ran the reproduction script through the real `handle_text` branching logic — no exception, second (queued) call correctly treated as answering the first call's pending question.
- [x] Re-ran Task 7's `verify_e2e.py` — still passes, sequential behavior unaffected.

No commit.

---

### Task 7: End-to-end verification (no live Telegram needed)

**Files:**
- None modified — this task only exercises Tasks 1-6 together via a script that calls the same functions `bot/handlers.py` calls, standing in for real `Message`/`CallbackQuery` objects.

**Interfaces:**
- Consumes: everything produced by Tasks 2-6.

- [ ] **Step 1: Write the full-cycle verification script**

Create `/tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_e2e.py`:

```python
import asyncio

from app.bot.handlers import _clear_pending_review, _run_turn
from app.config.settings import get_settings
from app.infra.memory.checkpointer import close_checkpointer, init_checkpointer
from app.infra.memory.working import close_redis, init_redis
from app.orchestrator import registry


async def main() -> None:
    settings = get_settings()
    await init_redis(settings)
    await init_checkpointer(settings)
    registry.discover_and_register()

    user_id = "verify-e2e-user"
    await _clear_pending_review(user_id)  # clean slate even on re-runs

    # Turn 1: fresh message that should route to search and pause.
    r1 = await _run_turn(user_id, new_text="Tìm tin tức mới nhất về AI hôm nay")
    print("turn1 has interrupt:", "__interrupt__" in r1)
    if "__interrupt__" not in r1:
        print("SKIP: search didn't pause (likely no documents from Tavily) — nothing further to verify")
        await close_checkpointer()
        await close_redis()
        return
    question = r1["__interrupt__"][0].value["question"]
    print("question:", question)

    # Turn 2: resume via free-text (not a button value) — exercises the
    # "both channels" resume decision from the design.
    r2 = await _run_turn(user_id, resume_answer="không cần, tóm tắt thôi")
    assert "__interrupt__" not in r2, f"should have completed, got: {r2}"
    last_ai = next(m for m in reversed(r2["messages"]) if getattr(m, "type", None) == "ai")
    print("final reply (truncated):", last_ai.content[:200])

    # Turn 3: /cancel semantics — start a new pause, then clear it, then
    # confirm the next message is treated as a brand new turn (no interrupt
    # replay of the stale question).
    r3 = await _run_turn(user_id, new_text="Tìm tin tức mới nhất về AI hôm nay")
    if "__interrupt__" in r3:
        await _clear_pending_review(user_id)
        from app.orchestrator.graph import get_compiled_graph

        graph = get_compiled_graph()
        snapshot = await graph.aget_state({"configurable": {"thread_id": user_id}})
        assert snapshot.next == (), f"expected cleared state, got next={snapshot.next}"
        print("cancel cleared pending state correctly")

    await close_checkpointer()
    await close_redis()
    print("VERIFY_E2E_OK")


asyncio.run(main())
```

- [ ] **Step 2: Run it**

Run: `uv run python /tmp/claude-1000/-home-loozzi-bot-assistant/cf5aa6ce-5c06-48ca-800d-71cb665b61d8/scratchpad/verify_e2e.py`
Expected: `VERIFY_E2E_OK` at the end. Read through the printed question/reply to confirm they're coherent Vietnamese text, not just that assertions passed — LLM output correctness isn't fully mechanically checkable.

- [ ] **Step 3: Manual smoke test note (not automatable here)**

This script proves the mechanism works without a live Telegram connection. Actually running `uv run python main.py` and messaging the real bot (button tap **and** free-text reply to the same pending question, plus `/cancel`) is the last-mile check the user should do themselves before considering this done — flag this explicitly when reporting completion, don't claim it as verified without it.

No commit.
