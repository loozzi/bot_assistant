# LLM-Based Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `orchestrator/router.py`'s hardcoded keyword-matching `classify_intent()` with an LLM-based classifier driven by each module's `config.yaml`, and make `orchestrator/graph.py`'s `_dispatch()` degrade gracefully instead of crashing when the LLM confidently routes to a real module that has no registered agent yet.

**Architecture:** `router.py` gains `_load_module_specs()` (globs `app/modules/*/config.yaml` for `description`/`examples`/`keywords`), `_build_system_prompt()`, and `_build_intent_schema()` (a dynamic Pydantic model constraining output to the discovered module names + `"unknown"`). `classify_intent()` calls `app/infra/providers/llm_client.py:create_llm_client()` with `.with_structured_output(...)`, and any exception — or the LLM legitimately picking `"unknown"` — collapses to `intent = "unknown"`. `resolve_agent_name()` is simplified to look modules up in the same discovered spec dict instead of a hand-maintained map. `graph.py:_dispatch()` wraps `registry.get(agent_name)` in `try/except KeyError`, reusing the existing "Mình chưa hiểu ý bạn" fallback message.

**Tech Stack:** Python 3.10, LangChain (`BaseChatModel.with_structured_output`, `langchain_openai.ChatOpenAI` under the hood via `create_llm_client()`), Pydantic v2 (`create_model`, `Literal`), PyYAML.

## Global Constraints

- No pytest or any test runner exists in this repo; this change does not add one. Verify every task with the manual `uv run python -c "..."` commands specified in that task — run them and confirm the exact output shown.
- **Do not run `git add` or `git commit` at any point in this plan.** The user reviews and commits everything themselves at the end. Every implementer must leave changes as uncommitted working-tree edits.
- Do not touch `app/infra/providers/*` (`openai.py`, `llm_client.py`) — already implemented, out of scope (per spec).
- Do not implement sticky-session logic (`config.yaml`'s `sticky`/`sticky_turns`) or the `priority` tie-breaker — explicitly out of scope (per spec).
- Do not add new `Settings` fields — reuse `llm_model`/`llm_temperature`/`llm_max_tokens`/`llm_provider`/`llm_api_key`/`llm_base_url`, already read by `create_llm_client()`.
- Classification input stays the latest human message only — no full conversation history.
- Run every verification command from the repo root (`/home/loozzi/bot_assistant`) using `uv run python -c "..."` unless a step says otherwise. A live LLM endpoint is already configured in `.env` (`LLM_PROVIDER=openai`, `LLM_BASE_URL` pointing at a local OpenAI-compatible proxy) — the manual verification steps that call the LLM require that endpoint to be reachable; if it isn't, report the exact connection error rather than skipping the step.

---

### Task 1: LLM-based `classify_intent()` and `resolve_agent_name()` in `router.py`

**Files:**
- Modify: `app/orchestrator/router.py` (full-file rewrite)

**Interfaces:**
- Consumes: `AgentState` (`app/core/state.py` — `messages: list`, `user_id: str`, `intent: str`, `retrieved_memories: list[dict]`, `metadata: dict`). `create_llm_client() -> BaseChatModel` (`app/infra/providers/llm_client.py` — already implemented, do not modify). Each `app/modules/<name>/config.yaml` has the shape `agent: {name, enabled, description}` and `routing: {keywords: {vi: [...], en: [...]}, examples: [...]}` (see any existing module's `config.yaml` for the exact shape — do not assume fields not present there).
- Produces: `async def classify_intent(state: AgentState) -> str` (unchanged signature — still async, still called as `await router.classify_intent(state)` from `graph.py:_route`) and `def resolve_agent_name(intent: str) -> str | None` (unchanged signature) — both consumed by Task 2's `graph.py` and already consumed today by `graph.py:_route`/`_dispatch`, which this task does not touch.

- [ ] **Step 1: Rewrite `app/orchestrator/router.py`**

Replace the full file contents with:

```python
import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, create_model

from app.core.state import AgentState
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MODULES_PATH = Path(__file__).resolve().parent.parent / "modules"


@functools.lru_cache(maxsize=1)
def _load_module_specs() -> dict[str, dict]:
    """Read every app/modules/<name>/config.yaml into {name: {description, examples, keywords}}."""
    specs: dict[str, dict] = {}
    for config_path in sorted(_MODULES_PATH.glob("*/config.yaml")):
        cfg = yaml.safe_load(config_path.read_text())
        agent_cfg = cfg.get("agent", {})
        if not agent_cfg.get("enabled", True):
            continue
        name = agent_cfg.get("name", config_path.parent.name)
        routing_cfg = cfg.get("routing", {})
        specs[name] = {
            "description": agent_cfg.get("description", "").strip(),
            "examples": routing_cfg.get("examples", []),
            "keywords": routing_cfg.get("keywords", {}),
        }
    return specs


def _build_system_prompt(specs: dict[str, dict]) -> str:
    sections = []
    for name, spec in specs.items():
        lines = [f"### {name}", spec["description"]]
        if spec["examples"]:
            lines.append("Examples:")
            lines.extend(f"- {ex}" for ex in spec["examples"])
        keywords = [*spec["keywords"].get("vi", []), *spec["keywords"].get("en", [])]
        if keywords:
            lines.append(f"Keywords: {', '.join(keywords)}")
        sections.append("\n".join(lines))

    modules_block = "\n\n".join(sections)
    return (
        "You are an intent classifier for a Telegram personal-assistant bot. "
        "Given the user's latest message, choose exactly one module that should "
        'handle it. If no module clearly applies, choose "unknown".\n\n'
        f"Available modules:\n\n{modules_block}"
    )


def _build_intent_schema(specs: dict[str, dict]) -> type[BaseModel]:
    labels = (*specs.keys(), "unknown")
    return create_model("IntentClassification", intent=(Literal[labels], ...))


async def classify_intent(state: AgentState) -> str:
    """Return an intent label based on the latest user message, via LLM classification."""
    last = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    if last is None:
        return "unknown"

    specs = _load_module_specs()
    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(_build_intent_schema(specs))
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": _build_system_prompt(specs)},
                {"role": "user", "content": getattr(last, "content", "")},
            ]
        )
        return result.intent
    except Exception as exc:
        logger.warning("intent_classification_failed", error=str(exc))
        return "unknown"


def resolve_agent_name(intent: str) -> str | None:
    if intent == "unknown":
        return None
    return intent if intent in _load_module_specs() else None
```

- [ ] **Step 2: Manually verify `_load_module_specs()` discovers all 5 modules, including `insight`**

Run:

```bash
uv run python -c "
from app.orchestrator.router import _load_module_specs

specs = _load_module_specs()
print(sorted(specs))
print(specs['search']['description'][:40])
print(bool(specs['search']['examples']))
print(bool(specs['search']['keywords']))
"
```

Expected output (four lines):
```
['finance', 'insight', 'journal', 'search', 'todo']
Searches the web for current informatio
True
True
```

(`insight` appears despite having no `__init__.py` — `_load_module_specs()` globs `config.yaml` files directly, unlike `registry.discover_and_register()`'s `pkgutil`-based scan. This is expected, not a bug — see the design spec's "Intentional asymmetry" note.)

- [ ] **Step 3: Manually verify a search-classified message routes correctly end-to-end via the LLM**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator.router import classify_intent

async def main():
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1',
        'intent': '',
        'retrieved_memories': [],
        'metadata': {},
    }
    intent = await classify_intent(state)
    print(intent)

asyncio.run(main())
"
```

Expected output:
```
search
```

If this fails with a connection error, the local LLM endpoint configured in `.env` (`LLM_BASE_URL`) is not reachable — report the exact error rather than editing around it; this is an environment precondition, not a code bug.

- [ ] **Step 4: Manually verify a journal-style message classifies to `"journal"`, and `resolve_agent_name` resolves it correctly**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator.router import classify_intent, resolve_agent_name

async def main():
    state = {
        'messages': [HumanMessage(content='Hôm nay tôi cảm thấy khá mệt mỏi')],
        'user_id': 'u1',
        'intent': '',
        'retrieved_memories': [],
        'metadata': {},
    }
    intent = await classify_intent(state)
    print(intent)
    print(resolve_agent_name(intent))

asyncio.run(main())
"
```

Expected output (two lines):
```
journal
journal
```

- [ ] **Step 5: Manually verify the failure path returns `"unknown"` without raising**

Run (temporarily overriding the API key/base URL with invalid values via env vars, without touching `.env`):

```bash
uv run python -c "
import asyncio
import os

os.environ['LLM_API_KEY'] = 'invalid-key-for-testing'
os.environ['LLM_BASE_URL'] = 'http://localhost:1/does-not-exist'

from langchain_core.messages import HumanMessage

from app.orchestrator.router import classify_intent

async def main():
    state = {
        'messages': [HumanMessage(content='anything')],
        'user_id': 'u1',
        'intent': '',
        'retrieved_memories': [],
        'metadata': {},
    }
    intent = await classify_intent(state)
    print(intent)

asyncio.run(main())
"
```

Expected output:
```
unknown
```

(No traceback — `classify_intent` must catch the connection failure internally and return `"unknown"`, per the design's resilience boundary.)

- [ ] **Step 6: Manually verify `resolve_agent_name` handles `"unknown"` and unrecognized labels**

Run:

```bash
uv run python -c "
from app.orchestrator.router import resolve_agent_name

print(resolve_agent_name('unknown'))
print(resolve_agent_name('not_a_real_module'))
print(resolve_agent_name('search'))
"
```

Expected output (three lines):
```
None
None
search
```

- [ ] **Step 7: Do not commit**

Leave `app/orchestrator/router.py` as an uncommitted working-tree change. Do not run `git add` or `git commit`.

---

### Task 2: `graph.py._dispatch()` degrades gracefully for a known-but-unregistered module

**Files:**
- Modify: `app/orchestrator/graph.py:16-26` (the `_dispatch` function body)

**Interfaces:**
- Consumes: `router.resolve_agent_name(intent: str) -> str | None` (Task 1). `registry.get(name: str) -> BaseAgent` (`app/orchestrator/registry.py` — existing, unmodified; raises `KeyError` if `name` isn't registered — currently only `"search"` is registered).
- Produces: no new public interface — `_dispatch` is a private graph node function, already wired into `build_graph()` in this same file.

- [ ] **Step 1: Edit `_dispatch()` in `app/orchestrator/graph.py`**

Current content of the function (lines 16-26):

```python
async def _dispatch(state: AgentState) -> AgentState:
    agent_name = router.resolve_agent_name(state["intent"])

    if agent_name is None:
        logger.warning("unresolved_intent", intent=state["intent"])
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?")
        return {**state, "messages": state["messages"] + [fallback]}

    agent = registry.get(agent_name)
    return await agent.run(state)
```

Replace with:

```python
async def _dispatch(state: AgentState) -> AgentState:
    agent_name = router.resolve_agent_name(state["intent"])

    if agent_name is None:
        logger.warning("unresolved_intent", intent=state["intent"])
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?")
        return {**state, "messages": state["messages"] + [fallback]}

    try:
        agent = registry.get(agent_name)
    except KeyError:
        logger.warning("resolved_agent_not_registered", agent=agent_name)
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?")
        return {**state, "messages": state["messages"] + [fallback]}

    return await agent.run(state)
```

No other part of `app/orchestrator/graph.py` changes — `_route`, `build_graph`, `get_compiled_graph`, and the module-level `_compiled` singleton stay exactly as they are.

- [ ] **Step 2: Manually verify a known-but-unregistered module (`"journal"`) no longer crashes `_dispatch`**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator.graph import _dispatch

async def main():
    state = {
        'messages': [HumanMessage(content='Hôm nay tôi cảm thấy khá mệt mỏi')],
        'user_id': 'u1',
        'intent': 'journal',
        'retrieved_memories': [],
        'metadata': {},
    }
    result = await _dispatch(state)
    print(result['messages'][-1].content)

asyncio.run(main())
"
```

Expected output:
```
Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?
```

(No traceback. `intent='journal'` is passed directly here — bypassing `classify_intent` — to test `_dispatch`'s `KeyError` handling in isolation, independent of Task 1's LLM call.)

- [ ] **Step 3: Manually verify `"search"` (registered) still dispatches normally — no regression**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator import registry
from app.orchestrator.graph import _dispatch

registry.discover_and_register()

async def main():
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1',
        'intent': 'search',
        'retrieved_memories': [],
        'metadata': {},
    }
    result = await _dispatch(state)
    print(result['messages'][-1].content)

asyncio.run(main())
"
```

Expected output (one line, the stub search summary):
```
[stub] Placeholder summary for: 'Search for the latest news on AI regulation' — search agent not yet connected to a real search/LLM provider.
```

If `registry.discover_and_register()` raises `ImportError` or the `search` agent isn't found: check that `app/modules/search/agent.py` on disk is the version from the prior "search agent + registry" work (it should already be an uncommitted working-tree change from before this plan) — report `BLOCKED` with the exact error rather than reimplementing it; that file is out of this task's scope.

- [ ] **Step 4: Manually verify the full orchestrator path end-to-end — LLM routing → dispatch, both tasks composed**

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
    # search — LLM classifies, dispatches to the real SearchAgent
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1', 'intent': '', 'retrieved_memories': [], 'metadata': {},
    }
    result = await graph.ainvoke(state)
    print(result['intent'])
    print(result['messages'][-1].content)

    # journal — LLM classifies to a real module with no registered agent yet
    state = {
        'messages': [HumanMessage(content='Hôm nay tôi cảm thấy khá mệt mỏi')],
        'user_id': 'u1', 'intent': '', 'retrieved_memories': [], 'metadata': {},
    }
    result = await graph.ainvoke(state)
    print(result['intent'])
    print(result['messages'][-1].content)

asyncio.run(main())
"
```

Expected output (four lines):
```
search
[stub] Placeholder summary for: 'Search for the latest news on AI regulation' — search agent not yet connected to a real search/LLM provider.
journal
Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?
```

- [ ] **Step 5: Do not commit**

Leave `app/orchestrator/graph.py` as an uncommitted working-tree change. Do not run `git add` or `git commit`.

---

## Self-Review Notes

- **Spec coverage:** §1 module spec discovery, §2 system prompt, §3 structured-output `classify_intent`, §4 `resolve_agent_name` simplification → all in Task 1 Step 1 (single-file rewrite, as the spec's code blocks are all within `router.py`). §5 `_dispatch` fix → Task 2 Step 1. The spec's "Testing" section's four scenarios map 1:1 to Task 1 Steps 3–4 (search/journal classification), Task 1 Step 5 (LLM failure → `"unknown"`), Task 1 Step 2 (module discovery including `insight`), and Task 2 Steps 2–4 (dispatch behavior, both crash-fix and no-regression).
- **Placeholder scan:** no TBD/TODO; every step has literal expected output.
- **Type/name consistency:** `classify_intent(state: AgentState) -> str` and `resolve_agent_name(intent: str) -> str | None` signatures are unchanged from the current file, so `graph.py`'s existing call sites (`await router.classify_intent(state)` in `_route`, `router.resolve_agent_name(state["intent"])` in `_dispatch`) need no changes beyond Task 2's `_dispatch` body edit. `_load_module_specs()`, `_build_system_prompt()`, `_build_intent_schema()` are private (`_`-prefixed) and used only within `router.py`, consistent with the file's existing convention (`_INTENT_MAP` was also private).
- **Task independence:** Task 2 Step 2 deliberately drives `_dispatch` with a hand-built `intent='journal'` state rather than depending on Task 1's LLM classification, so Task 2 can be implemented, verified, and reviewed independently of whether Task 1's live LLM endpoint is reachable at that moment. Task 1 Step 4 already covers the LLM producing `"journal"` in the first place. Only the composed end-to-end check (Task 2 Step 4) requires both tasks' code to be present.
