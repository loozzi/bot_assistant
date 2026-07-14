# Search Agent + Registry Auto-Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `search` the first module that actually runs end-to-end through the orchestrator, and make `orchestrator/registry.py` auto-discover/register any module that provides one, instead of requiring hand-written registration.

**Architecture:** `search`'s existing 3-node internal LangGraph (`search → crawl → summary`) stays as-is, but gets a real entry point/terminal edge and is wrapped by a new `SearchAgent(BaseAgent)` whose `run()` translates `AgentState` in and out at the boundary. `orchestrator/registry.py` gains `discover_and_register()`, which walks `app/modules/*`, imports `<module>.agent` if it exists, and registers the module-level `agent` object it exposes (gated by that module's own `config.agent.enabled`). `app/main.py` calls discovery once at startup, before the graph is warmed up.

**Tech Stack:** Python 3.10, LangGraph (`StateGraph`), `langchain_core.messages`, `pkgutil`/`importlib` (stdlib), PyYAML (already a transitive dep via `langgraph`/other packages — confirmed importable in the venv).

## Global Constraints

- No pytest or any test runner exists in this repo yet, and adding one is explicitly out of scope for this change (per the approved spec). Every task is verified with a manual `uv run python -c "..."` snippet instead of an automated test — run it and confirm the exact output shown.
- Node stub bodies must not perform any network or LLM calls — placeholder data only (per spec §Non-goals).
- Do not touch `AgentState`/`BaseState` schema definitions, journal/finance/insight/todo modules, or extract `_last_human_message` into `utils/` — all explicitly out of scope (per spec).
- Run every verification command from the repo root (`/home/loozzi/bot_assistant`) using `uv run python -c "..."` unless a step says otherwise.

---

### Task 1: Search node stubs return placeholder data

**Files:**
- Modify: `app/modules/search/node/search_node.py`
- Modify: `app/modules/search/node/crawl_node.py`
- Modify: `app/modules/search/node/summary_node.py`

**Interfaces:**
- Consumes: `SearchState` (`app/modules/search/state.py`) — has `user_query: str`, `documents: list[DocumentInfo]`, `summary: str`, plus `BaseState` fields (`messages`, `user_id`, `chat_id`, `agent_outputs`, `errors`). `DocumentInfo` is a `TypedDict` with `title: str`, `url: str`, `raw_text: str`.
- Produces: each node returns a partial-state `dict` LangGraph merges into `SearchState`. `search_node` produces `documents`; `crawl_node` produces `documents`; `summary_node` produces `summary` — later tasks (Task 2's `SearchAgent`) read `result["summary"]` from the compiled graph's output.

- [ ] **Step 1: Replace `search_node.py` body**

Write `app/modules/search/node/search_node.py`:

```python
from ..state import SearchState


async def search_node(state: SearchState) -> dict:
    """
    Perform a search based on the user's query and return relevant documents.
    """
    return {
        "documents": [
            {
                "title": "[stub] Placeholder result",
                "url": "https://example.invalid/stub",
                "raw_text": (
                    "This is placeholder content — search agent is not yet "
                    "wired to a real search provider."
                ),
            }
        ]
    }
```

- [ ] **Step 2: Replace `crawl_node.py` body**

Write `app/modules/search/node/crawl_node.py`:

```python
from ..state import SearchState


async def crawl_node(state: SearchState) -> dict:
    """
    Crawl the web for documents based on the user's query.
    """
    return {"documents": state["documents"]}
```

- [ ] **Step 3: Replace `summary_node.py` body**

Write `app/modules/search/node/summary_node.py`:

```python
from ..state import SearchState


async def summary_node(state: SearchState) -> dict:
    """
    Summarize the documents retrieved from the crawl node.
    """
    return {
        "summary": (
            f"[stub] Placeholder summary for: {state['user_query']!r} — "
            "search agent not yet connected to a real search/LLM provider."
        )
    }
```

- [ ] **Step 4: Manually verify each node in isolation**

Run:

```bash
uv run python -c "
import asyncio

from app.modules.search.node import search_node, crawl_node, summary_node

async def main():
    s = await search_node({})
    print('search_node ->', s)

    c = await crawl_node({'documents': s['documents']})
    print('crawl_node ->', c)

    m = await summary_node({'documents': c['documents'], 'user_query': 'gold price today'})
    print('summary_node ->', m)

asyncio.run(main())
"
```

Expected output (three lines):
```
search_node -> {'documents': [{'title': '[stub] Placeholder result', 'url': 'https://example.invalid/stub', 'raw_text': 'This is placeholder content — search agent is not yet wired to a real search provider.'}]}
crawl_node -> {'documents': [{'title': '[stub] Placeholder result', 'url': 'https://example.invalid/stub', 'raw_text': 'This is placeholder content — search agent is not yet wired to a real search provider.'}]}
summary_node -> {'summary': "[stub] Placeholder summary for: 'gold price today' — search agent not yet connected to a real search/LLM provider."}
```

- [ ] **Step 5: Commit**

```bash
git add app/modules/search/node/search_node.py app/modules/search/node/crawl_node.py app/modules/search/node/summary_node.py
git commit -m "feat(search): return placeholder data from search/crawl/summary node stubs"
```

---

### Task 2: `SearchAgent(BaseAgent)` wraps the internal graph

**Files:**
- Modify: `app/modules/search/agent.py`

**Interfaces:**
- Consumes: `BaseAgent` ABC (`app/core/base_agent.py` — abstract `name` property, abstract `async def run(self, state: AgentState) -> AgentState`). `AgentState` (`app/core/state.py` — `TypedDict` with `messages: list`, `user_id: str`, `intent: str`, `retrieved_memories: list[dict]`, `metadata: dict`). `SearchState`/node functions from Task 1.
- Produces: module-level `agent = SearchAgent()` (an instance of `BaseAgent`) and module-level `config` (the dict loaded from `config.yaml`) — both are the exact attributes Task 3's `discover_and_register()` looks up via `getattr(module, "agent", None)` and `getattr(module, "config", {})`.

The current file has two bugs blocking this task that must be fixed as part of it:
1. `yaml.safe_load(open("./config.yaml", "r"))` only resolves when the process cwd happens to be `app/modules/search/` — breaks under normal execution from the repo root.
2. The `StateGraph` has no entry point and no edge to `END`, so `builder.compile()` raises `ValueError: Graph must have an entrypoint: add at least one edge from START to another node` (verified by running it as-is).

- [ ] **Step 1: Rewrite `app/modules/search/agent.py`**

```python
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.core.base_agent import BaseAgent
from app.core.state import AgentState

from .node import crawl_node, search_node, summary_node
from .state import SearchState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=SearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

# Add nodes
builder.add_node("search", search_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summary", summary_node)

# Add edges
builder.set_entry_point("search")
builder.add_edge("search", "crawl")
builder.add_edge("crawl", "summary")
builder.add_edge("summary", END)

graph = builder.compile()


def _last_human_message(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return message.content
    return ""


class SearchAgent(BaseAgent):
    """Wraps the search/crawl/summary subgraph behind the BaseAgent contract."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, state: AgentState) -> AgentState:
        initial: SearchState = {
            "messages": state["messages"],
            "user_id": state["user_id"],
            "chat_id": state["user_id"],
            "user_query": _last_human_message(state["messages"]),
            "documents": [],
            "summary": "",
        }
        result = await graph.ainvoke(initial)
        reply = AIMessage(content=result["summary"])
        return {**state, "messages": state["messages"] + [reply]}


agent = SearchAgent()
```

- [ ] **Step 2: Manually verify the compiled subgraph runs standalone**

Run:

```bash
uv run python -c "
import asyncio

from app.modules.search.agent import graph

async def main():
    initial = {
        'messages': [],
        'user_id': 'u1',
        'chat_id': 'u1',
        'user_query': 'gold price today',
        'documents': [],
        'summary': '',
    }
    result = await graph.ainvoke(initial)
    print(result['summary'])

asyncio.run(main())
"
```

Expected output (one line):
```
[stub] Placeholder summary for: 'gold price today' — search agent not yet connected to a real search/LLM provider.
```

- [ ] **Step 3: Manually verify `SearchAgent.run()` against the `AgentState` contract**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.modules.search.agent import agent

async def main():
    state = {
        'messages': [HumanMessage(content='Tìm giúp tôi giá vàng hôm nay')],
        'user_id': 'u1',
        'intent': 'search',
        'retrieved_memories': [],
        'metadata': {},
    }
    result = await agent.run(state)
    print(agent.name)
    print(result['messages'][-1].content)

asyncio.run(main())
"
```

Expected output (two lines):
```
search
[stub] Placeholder summary for: 'Tìm giúp tôi giá vàng hôm nay' — search agent not yet connected to a real search/LLM provider.
```

- [ ] **Step 4: Commit**

```bash
git add app/modules/search/agent.py
git commit -m "feat(search): wrap search/crawl/summary subgraph in a SearchAgent(BaseAgent)"
```

---

### Task 3: Registry auto-discovers and registers module agents

**Files:**
- Modify: `app/orchestrator/registry.py`

**Interfaces:**
- Consumes: `app.modules.<name>.agent` modules that expose `agent: BaseAgent` and `config: dict` at module level (contract established in Task 2 for `search`; any future module follows the same shape). `BaseAgent.name` (`app/core/base_agent.py`).
- Produces: `discover_and_register() -> None` — a new public function. After calling it, `all_agents()` (already existing) reflects every discovered, enabled module agent. This is what Task 4's `main.py` calls at startup.

- [ ] **Step 1: Add `discover_and_register()` to `app/orchestrator/registry.py`**

Replace the full file contents with:

```python
import importlib
import pkgutil
from pathlib import Path

from app.core.base_agent import BaseAgent
from app.utils.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, BaseAgent] = {}
_MODULES_PATH = Path(__file__).resolve().parent.parent / "modules"


def register(agent: BaseAgent) -> None:
    if agent.name in _registry:
        raise ValueError(f"Agent '{agent.name}' is already registered")
    _registry[agent.name] = agent
    logger.info("agent_registered", agent=agent.name)


def get(name: str) -> BaseAgent:
    if name not in _registry:
        raise KeyError(f"No agent registered with name '{name}'")
    return _registry[name]


def all_agents() -> dict[str, BaseAgent]:
    return dict(_registry)


def discover_and_register() -> None:
    """Import every app.modules.<name>.agent module and register its `agent`.

    Modules without an agent.py are skipped (expected for modules that
    haven't been implemented yet, e.g. journal/finance/insight/todo today).
    A module's config.yaml `agent.enabled: false` prevents registration
    without requiring the module to be deleted.
    """
    for _, module_name, is_pkg in pkgutil.iter_modules([str(_MODULES_PATH)]):
        if not is_pkg:
            continue

        agent_module_path = _MODULES_PATH / module_name / "agent.py"
        if not agent_module_path.exists():
            logger.debug("module_has_no_agent", module=module_name)
            continue

        module = importlib.import_module(f"app.modules.{module_name}.agent")
        agent = getattr(module, "agent", None)
        if not isinstance(agent, BaseAgent):
            logger.warning("module_agent_attribute_missing_or_invalid", module=module_name)
            continue

        module_config = getattr(module, "config", {})
        enabled = module_config.get("agent", {}).get("enabled", True)
        if not enabled:
            logger.info("module_agent_disabled", module=module_name)
            continue

        register(agent)
```

- [ ] **Step 2: Manually verify discovery finds and registers `search`, skips everything else**

Run:

```bash
uv run python -c "
from app.orchestrator import registry

registry.discover_and_register()
print(sorted(registry.all_agents()))
print(registry.get('search').name)
"
```

Expected output (two lines):
```
['search']
search
```

(`journal`/`finance`/`todo` are skipped because they have no `agent.py` yet; `insight` has no `__init__.py` so `pkgutil` doesn't even see it as a package — both are expected, not bugs.)

- [ ] **Step 3: Manually verify calling `discover_and_register()` twice doesn't crash**

Run:

```bash
uv run python -c "
from app.orchestrator import registry

registry.discover_and_register()
registry.discover_and_register()
print(sorted(registry.all_agents()))
"
```

Expected: raises `ValueError: Agent 'search' is already registered` on the second call — this is the existing `register()` guard firing correctly (re-importing an already-imported module returns the cached module object, so `discover_and_register()` is not idempotent; that's fine since `main.py` will only call it once at startup — confirmed in Task 4).

- [ ] **Step 4: Commit**

```bash
git add app/orchestrator/registry.py
git commit -m "feat(orchestrator): auto-discover and register module agents from app/modules"
```

---

### Task 4: Wire discovery into bot startup

**Files:**
- Modify: `app/main.py:1-51` (imports block and `on_startup`)

**Interfaces:**
- Consumes: `registry.discover_and_register()` (Task 3).

- [ ] **Step 1: Add the registry import**

In `app/main.py`, the current imports are:

```python
from app.bot.handlers import router as main_router
from app.bot.middlewares import LoggingMiddleware, RateLimitMiddleware, UserContextMiddleware
from app.bot.scheduler import init_scheduler
from app.config.settings import get_settings
from app.config.logging import setup_logging
from app.infra.db.session import close_db, init_db
from app.infra.db.vector import close_vector_db, init_vector_db
from app.infra.memory.working import close_redis, init_redis
from app.orchestrator.graph import get_compiled_graph
from app.utils.logger import get_logger
```

Add one import, keeping the existing `app.orchestrator` grouping:

```python
from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph
```

(i.e. insert `from app.orchestrator import registry` immediately above the existing `from app.orchestrator.graph import get_compiled_graph` line.)

- [ ] **Step 2: Call discovery before the graph warm-up**

Current `on_startup`:

```python
async def on_startup(bot: Bot) -> None:
    settings = get_settings()

    await init_db(settings)
    await init_redis(settings)
    await init_vector_db(settings)

    # Warm-up the compiled graph so the first request isn't slow
    get_compiled_graph()

    scheduler = init_scheduler()
    scheduler.start()

    me = await bot.get_me()
    logger.info("bot_started", username=me.username)
```

Change to:

```python
async def on_startup(bot: Bot) -> None:
    settings = get_settings()

    await init_db(settings)
    await init_redis(settings)
    await init_vector_db(settings)

    registry.discover_and_register()

    # Warm-up the compiled graph so the first request isn't slow
    get_compiled_graph()

    scheduler = init_scheduler()
    scheduler.start()

    me = await bot.get_me()
    logger.info("bot_started", username=me.username)
```

- [ ] **Step 3: Manually verify the startup sequence (without DB/Redis/Telegram) — registry + graph warm-up**

Run:

```bash
uv run python -c "
from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph

registry.discover_and_register()
graph = get_compiled_graph()
print(sorted(registry.all_agents()))
print(type(graph).__name__)
"
```

Expected output (two lines):
```
['search']
CompiledStateGraph
```

- [ ] **Step 4: Manually verify the full orchestrator path end-to-end (router → registry → SearchAgent)**

This is the deliverable's real proof: a message that keyword-matches "search" flows through `route` → `dispatch` → `SearchAgent.run()` → the internal subgraph → back out as the bot's reply.

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph

registry.discover_and_register()
graph = get_compiled_graph()

async def main():
    state = {
        'messages': [HumanMessage(content='Tìm giúp tôi giá vàng hôm nay')],
        'user_id': 'u1',
        'intent': '',
        'retrieved_memories': [],
        'metadata': {},
    }
    result = await graph.ainvoke(state)
    print(result['intent'])
    print(result['messages'][-1].content)

asyncio.run(main())
"
```

Expected output (two lines):
```
search
[stub] Placeholder summary for: 'Tìm giúp tôi giá vàng hôm nay' — search agent not yet connected to a real search/LLM provider.
```

- [ ] **Step 5: Verify `app/main.py` still imports cleanly (catches syntax/import errors without needing a real bot token)**

Run:

```bash
uv run python -c "import app.main; print('import ok')"
```

Expected output:
```
import ok
```

- [ ] **Step 6: Commit**

```bash
git add app/main.py
git commit -m "feat(bot): discover and register module agents at startup"
```

---

## Self-Review Notes

- **Spec coverage:** §1 SearchAgent → Task 2. §2 node stubs → Task 1. §3 config path fix → Task 2 Step 1 (folded in, since it's the same file and blocks the graph from compiling at all). §4 registry discovery → Task 3. §5 main.py wiring → Task 4. The spec's "Testing" section (manual verification of discovery + a stub reply through the graph) is covered by Task 3 Step 2 and Task 4 Step 4.
- **Extra fix not in the original spec text:** the spec's code sketch for `SearchAgent` implicitly assumed `graph.ainvoke()` would just work; actually running it (done during planning) surfaced that `builder` had no entry point and no edge to `END`, so `compile()` raised immediately. Folded the fix into Task 2 Step 1 since it's required for the task's own deliverable to run at all, and flagged inline in that task.
- **Type/name consistency:** `SearchAgent.name` returns `config["agent"]["name"]` (`"search"`, from `app/modules/search/config.yaml`) — matches what Task 3's `discover_and_register()` registers under and what Task 4's verification asserts (`registry.get('search')`, `result['intent'] == 'search'` via the existing keyword router already matching "tìm"/"search"). `agent = SearchAgent()` and `config = yaml.safe_load(...)` module-level names in Task 2 exactly match the `getattr(module, "agent", ...)` / `getattr(module, "config", ...)` lookups in Task 3.
- **No placeholders:** every step has full, runnable code and literal expected output.
