# Search agent as BaseAgent template + auto-discovering registry

Date: 2026-07-14
Status: Approved

## Problem

Two related gaps block any module from actually running through the orchestrator:

1. No module implements `BaseAgent`. `app/modules/search/agent.py` builds its own standalone `StateGraph` but never subclasses `BaseAgent`, so it can't be dispatched by `orchestrator/graph.py`. Its node stubs (`search_node`/`crawl_node`/`summary_node`) all `return {}`.
2. Nothing calls `orchestrator/registry.register()`. Even if an intent classifies correctly, `orchestrator/registry.py:get(agent_name)` raises `KeyError` because the registry is always empty.

This spec implements a minimal, real end-to-end path for the `search` module and a generic registry mechanism other modules (journal, finance, insight, todo) can plug into the same way once they get an `agent.py`.

## Goals

- Turn `search` into a working example of "module with its own internal multi-step graph, exposed through `BaseAgent`" — the pattern future multi-step modules copy.
- Make `orchestrator/registry.py` automatically discover and register any module that provides an agent, instead of requiring hand-written registration per module.
- Keep the search node bodies as clearly-fake stubs (no network calls, no LLM calls) — this is about wiring the plumbing, not building real search.

## Non-goals

- Real web search / crawling / summarization (no LLM provider exists yet — out of scope).
- Adding pytest or any test infrastructure (explicitly deferred by user).
- Reconciling `AgentState` vs `BaseState` into one schema (documented gap in CLAUDE.md; not part of this change).
- Implementing journal/finance/insight/todo agents (no `agent.py` for them yet — registry discovery simply skips modules without one).

## Design

### 1. `SearchAgent(BaseAgent)` — `app/modules/search/agent.py`

The module keeps its existing internal `StateGraph` (`search → crawl → summary`, using `SearchState`/`BaseState`). It is compiled once at import time as today (`graph = builder.compile()`).

A new `SearchAgent` class subclasses `BaseAgent` and becomes the module's public entry point:

```python
class SearchAgent(BaseAgent):
    @property
    def name(self) -> str:
        return config["agent"]["name"]  # "search"

    async def run(self, state: AgentState) -> AgentState:
        last_human = _last_human_message(state["messages"])
        initial: SearchState = {
            "messages": state["messages"],
            "user_id": state["user_id"],
            "chat_id": state["user_id"],  # AgentState has no chat_id yet
            "user_query": last_human,
            "documents": [],
            "summary": "",
        }
        result = await graph.ainvoke(initial)
        reply = AIMessage(content=result["summary"])
        return {**state, "messages": state["messages"] + [reply]}


agent = SearchAgent()
```

`_last_human_message` is a small local helper mirroring the one already duplicated in `orchestrator/router.py` and `bot/handlers.py` (reverse-scan `messages` for `type == "human"`). Duplicating it locally (not extracting to `utils/`) matches the existing pattern in the codebase — an extraction is a separate cleanup, not part of this change.

`agent = SearchAgent()` at module level is the discovery contract: registry looks for an attribute literally named `agent` on `app.modules.<name>.agent`, plus the existing module-level `config` dict for the enabled flag (see §4).

### 2. Node stubs — `app/modules/search/node/*.py`

Each node stays a single async function taking `SearchState`, but returns clearly-fake placeholder data instead of `{}`:

- `search_node`: returns one fake `DocumentInfo` (`title="[stub] Placeholder result"`, `url="https://example.invalid/stub"`, `raw_text="This is placeholder content — search agent is not yet wired to a real search provider."`) under `documents`.
- `crawl_node`: passes `documents` through unchanged (in a real implementation this would fetch full page content per URL; stub has nothing to fetch).
- `summary_node`: returns `summary` as an f-string: `f"[stub] Placeholder summary for: {state['user_query']!r} — search agent not yet connected to a real search/LLM provider."`

No network calls, no new dependencies.

### 3. Config path fix — `app/modules/search/agent.py`

Current code: `yaml.safe_load(open("./config.yaml", "r"))` — breaks unless the process cwd happens to be `app/modules/search/`. Fix:

```python
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())
```

This is required for registry discovery to work at all, since it imports `app.modules.search.agent` from the process's normal cwd (repo root).

### 4. Registry auto-discovery — `app/orchestrator/registry.py`

Add:

```python
import importlib
import pkgutil
from pathlib import Path

_MODULES_PATH = Path(__file__).resolve().parent.parent / "modules"


def discover_and_register() -> None:
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

Behavior:
- Only packages under `app/modules/` are considered (`is_pkg` — skips stray files).
- A module without `agent.py` is silently skipped at debug level — this is the expected state for journal/finance/insight/todo today, not an error.
- A module with `agent.py` but no valid `agent` attribute logs a warning (misconfiguration) but doesn't crash startup.
- Reuses the `config` dict `agent.py` already loads from `config.yaml` (rather than re-parsing the file) to read `agent.enabled`; missing config or key defaults to enabled. This makes `config = yaml.safe_load(...)` at module level part of the same discovery contract as `agent = SearchAgent()`.
- Reuses the existing `register()` — no change to its signature or the already-correct duplicate-registration guard.

### 5. Wiring — `app/main.py`

In `on_startup`, call discovery before warming the compiled graph:

```python
from app.orchestrator import registry
...
registry.discover_and_register()
get_compiled_graph()
```

## Testing

No automated tests (explicitly out of scope for this change). Manual verification:
1. `uv run python main.py` (or a short script) imports `app.orchestrator.registry`, calls `discover_and_register()`, and confirms `registry.all_agents()` contains `"search"`.
2. Send a search-classified message through the compiled graph (or call `SearchAgent().run(state)` directly) and confirm it returns an `AgentState` whose last message is the stub summary string, with no exceptions.

## Follow-ups (not in this change)

- Add `agent.py` for journal/finance/insight/todo once each has real logic — they'll pick up registration automatically via `discover_and_register()`.
- Replace stub nodes with real search/crawl/summarize once an LLM provider and search tool exist.
- Reconcile `AgentState`/`BaseState` duplication.
- Set up pytest and backfill tests for this change and future ones.
