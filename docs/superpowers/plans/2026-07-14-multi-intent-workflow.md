# Multi-Intent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the orchestrator's single-intent, single-agent flow with a multi-intent, parallel-dispatch flow, while fully isolating module state/I/O from the orchestrator's internal bookkeeping.

**Architecture:** `AgentState` (orchestrator-only) gains `intents`/`agent_outputs`/`errors`; `BaseState` is deleted. `BaseAgent.run()` changes from `(AgentState) -> AgentState` to `(AgentInput) -> AgentOutput` — a minimal, module-agnostic I/O contract. The graph grows from 2 nodes (`route`→`dispatch`) to 6 (`ingest`→`rehydrate_context`→`route`→ fan-out via LangGraph `Send` →`dispatch_agent`(×N)→`format_response`→`episodic_writer`→`END`).

**Tech Stack:** Python 3.10, LangGraph (`Send` API for dynamic parallel fan-out, `StateGraph` reducers via `Annotated`), LangChain (`ChatOpenAI.with_structured_output(..., method="json_mode")`).

## Global Constraints

- No pytest configured; verify every task with the manual `uv run python -c "..."` commands below.
- **Do not run `git add` or `git commit`.** Leave changes uncommitted for the user to review and commit themselves.
- Do not modify `app/infra/providers/*`, `app/infra/memory/working.py`, or `app/infra/db/vector.py` — `rehydrate_context`/`episodic_writer` stay honest no-op stub nodes this round (per the spec's Non-goals); real memory-layer wiring is a separate future spec.
- Do not implement sticky-session behavior (`config.yaml`'s `sticky`/`sticky_turns`).
- Do not modify `app/modules/search/node/search_node.py`, `crawl_node.py`, or `summary_node.py` — confirmed compatible as-is (they only read `user_query`/`documents`).
- Do not modify `app/orchestrator/registry.py` or `app/main.py` — `discover_and_register()` and its startup call are unaffected by this plan.
- Copy all code blocks below verbatim — they are the full, final file contents (not diffs) unless explicitly marked as a partial edit.

---

### Task 1: Core contracts — `AgentState` + isolated `BaseAgent` I/O

**Files:**
- Modify: `app/core/state.py` (full rewrite)
- Modify: `app/core/base_agent.py` (full rewrite)

**Interfaces:**
- Produces: `AgentState` TypedDict (`messages`, `user_id`, `intents`, `retrieved_memories`, `agent_outputs`, `errors`, `metadata`) — consumed by Task 3 (`router.py`), Task 4 (`graph.py`), Task 5 (`handlers.py`).
- Produces: `AgentInput` TypedDict (`user_id`, `message`, `retrieved_memories`), `AgentOutput` TypedDict (`reply`), and `BaseAgent.run(self, input: AgentInput) -> AgentOutput` — consumed by Task 2 (`search/agent.py`) and Task 4 (`graph.py`).
- `BaseState` is deleted entirely from `app/core/state.py`.

- [ ] **Step 1: Rewrite `app/core/state.py`**

```python
import operator
from typing import Annotated, Any, TypedDict
from typing_extensions import NotRequired

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    intents: list[str]
    retrieved_memories: NotRequired[list[dict]]
    agent_outputs: Annotated[dict[str, Any], operator.or_]
    errors: Annotated[list[str], operator.add]
    metadata: NotRequired[dict[str, Any]]
```

- [ ] **Step 2: Rewrite `app/core/base_agent.py`**

```python
from abc import ABC, abstractmethod
from typing import TypedDict


class AgentInput(TypedDict):
    user_id: str
    message: str
    retrieved_memories: list[dict]


class AgentOutput(TypedDict):
    reply: str


class BaseAgent(ABC):
    """All feature agents must subclass this and implement `run`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier used for routing and logging."""
        ...

    @abstractmethod
    async def run(self, input: AgentInput) -> AgentOutput:
        """Process minimal input and return the agent's own reply only."""
        ...
```

- [ ] **Step 3: Manually verify**

Run:

```bash
uv run python -c "
import asyncio

from app.core.state import AgentState
from app.core.base_agent import AgentInput, AgentOutput, BaseAgent

assert 'intents' in AgentState.__annotations__
assert 'agent_outputs' in AgentState.__annotations__
assert 'errors' in AgentState.__annotations__
assert 'retrieved_memories' in AgentState.__annotations__

try:
    from app.core.state import BaseState
    raise SystemExit('FAIL: BaseState still exists — should be deleted')
except ImportError:
    pass


class Dummy(BaseAgent):
    @property
    def name(self) -> str:
        return 'dummy'

    async def run(self, input: AgentInput) -> AgentOutput:
        return {'reply': f\"echo: {input['message']}\"}


async def main():
    out = await Dummy().run({'user_id': 'u1', 'message': 'hi', 'retrieved_memories': []})
    assert out == {'reply': 'echo: hi'}, out
    print('OK')

asyncio.run(main())
"
```

Expected output: `OK` (no assertion errors, no `ImportError` bypassed).

- [ ] **Step 4: Do not commit**

Leave both files as uncommitted working-tree changes.

---

### Task 2: Isolate the `search` module behind the new `BaseAgent` contract

**Files:**
- Modify: `app/modules/search/state.py` (full rewrite)
- Modify: `app/modules/search/agent.py` (full rewrite)

**Interfaces:**
- Consumes: `AgentInput`, `AgentOutput`, `BaseAgent` from `app.core.base_agent` (Task 1).
- Produces: `SearchAgent.run(input: AgentInput) -> AgentOutput`; module-level `agent = SearchAgent()` singleton — name and location unchanged, so `registry.discover_and_register()` (untouched) keeps finding it exactly as before.

- [ ] **Step 1: Rewrite `app/modules/search/state.py`**

```python
from typing import TypedDict


class DocumentInfo(TypedDict):
    title: str
    url: str
    raw_text: str


class SearchState(TypedDict):
    user_id: str
    user_query: str
    documents: list[DocumentInfo]
    summary: str
```

- [ ] **Step 2: Rewrite `app/modules/search/agent.py`**

```python
from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent

from .node import crawl_node, search_node, summary_node
from .state import SearchState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=SearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("search", search_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summary", summary_node)

builder.set_entry_point("search")
builder.add_edge("search", "crawl")
builder.add_edge("crawl", "summary")
builder.add_edge("summary", END)

graph = builder.compile()


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
        result = await graph.ainvoke(initial)
        return {"reply": result["summary"]}


agent = SearchAgent()
```

Do not modify `app/modules/search/node/search_node.py`, `crawl_node.py`, or `summary_node.py` — they only read `state["user_query"]`/`state["documents"]`, both still present in the new `SearchState`.

- [ ] **Step 3: Manually verify**

Run:

```bash
uv run python -c "
import asyncio

from app.modules.search.agent import agent

async def main():
    result = await agent.run({'user_id': 'u1', 'message': 'test query', 'retrieved_memories': []})
    assert set(result.keys()) == {'reply'}, result
    assert 'test query' in result['reply'], result
    print(result['reply'])

asyncio.run(main())
"
```

Expected output: one line containing `[stub] Placeholder summary for: 'test query' — search agent not yet connected to a real search/LLM provider.`

- [ ] **Step 4: Do not commit**

---

### Task 3: Multi-intent classification in `router.py`

**Files:**
- Modify: `app/orchestrator/router.py` (full rewrite)

**Interfaces:**
- Consumes: `AgentState` from `app.core.state` (Task 1).
- Produces: `classify_intents(state: AgentState) -> list[str]`, `resolve_agent_names(intents: list[str]) -> list[str]` — consumed by Task 4 (`graph.py`). Replaces the old `classify_intent`/`resolve_agent_name` (singular) entirely — no call sites for the old names should remain after this task.

- [ ] **Step 1: Rewrite `app/orchestrator/router.py`**

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
        "Given the user's latest message, choose every module that applies — "
        "most messages need only one, but choose more than one when the "
        'message clearly asks for multiple distinct things. If no module '
        'clearly applies, choose "unknown".\n\n'
        f"Available modules:\n\n{modules_block}\n\n"
        'Respond only with a JSON object of the form {"intents": ["<module_name>", ...]}.'
    )


def _build_intent_schema(specs: dict[str, dict]) -> type[BaseModel]:
    labels = (*specs.keys(), "unknown")
    return create_model("IntentClassification", intents=(list[Literal[labels]], ...))


async def classify_intents(state: AgentState) -> list[str]:
    """Return one or more intent labels for the latest user message, via LLM classification."""
    last = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    if last is None:
        return ["unknown"]

    specs = _load_module_specs()
    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(_build_intent_schema(specs), method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": _build_system_prompt(specs)},
                {"role": "user", "content": getattr(last, "content", "")},
            ]
        )
        return result.intents or ["unknown"]
    except Exception as exc:
        logger.warning("intent_classification_failed", error=str(exc))
        return ["unknown"]


def resolve_agent_names(intents: list[str]) -> list[str]:
    specs = _load_module_specs()
    seen: list[str] = []
    for intent in intents:
        if intent != "unknown" and intent in specs and intent not in seen:
            seen.append(intent)
    return seen
```

- [ ] **Step 2: Manually verify structural correctness (stubbed LLM)**

Run:

```bash
uv run python -c "
import asyncio
from unittest.mock import patch

from langchain_core.messages import HumanMessage

import app.orchestrator.router as router_mod


class FakeClassifier:
    def __init__(self, schema):
        self._schema = schema

    async def ainvoke(self, messages):
        sys_content = messages[0]['content']
        assert '\"intents\"' in sys_content, 'system prompt missing intents JSON-shape instruction'
        return self._schema(intents=['search', 'journal'])


class FakeLLM:
    def with_structured_output(self, schema, method=None):
        return FakeClassifier(schema)


async def main():
    state = {
        'messages': [HumanMessage(content='search for X and journal about Y')],
        'user_id': 'u1', 'intents': [], 'retrieved_memories': [],
        'agent_outputs': {}, 'errors': [], 'metadata': {},
    }
    with patch.object(router_mod, 'create_llm_client', return_value=FakeLLM()):
        intents = await router_mod.classify_intents(state)
        print(intents)

    print(router_mod.resolve_agent_names(intents))
    print(router_mod.resolve_agent_names(['unknown']))

asyncio.run(main())
"
```

Expected output (three lines):
```
['search', 'journal']
['search', 'journal']
[]
```

- [ ] **Step 3: Manually verify against the live endpoint (if reachable)**

Run:

```bash
uv run python -c "
import asyncio

from langchain_core.messages import HumanMessage

from app.orchestrator.router import classify_intents

async def main():
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1', 'intents': [], 'retrieved_memories': [],
        'agent_outputs': {}, 'errors': [], 'metadata': {},
    }
    print(await classify_intents(state))

asyncio.run(main())
"
```

Expected: `['search']` if the endpoint is reachable and classifies correctly, or `['unknown']` with no traceback if not — both are acceptable results for this task (the second confirms the failure path still degrades gracefully).

- [ ] **Step 4: Do not commit**

---

### Task 4: Rewrite the orchestrator graph for multi-intent parallel dispatch

**Files:**
- Modify: `app/orchestrator/graph.py` (full rewrite)

**Interfaces:**
- Consumes: `AgentState`, `AgentInput` (Task 1); `router.classify_intents`, `router.resolve_agent_names` (Task 3); `registry.get` (unchanged); `create_llm_client` (unchanged, `app/infra/providers/llm_client.py`).
- Produces: `build_graph()`, `get_compiled_graph()` — consumed by Task 5 (`bot/handlers.py`). Signatures unchanged from today, so Task 5's call site doesn't need to change beyond its input dict.

- [ ] **Step 1: Rewrite `app/orchestrator/graph.py`**

```python
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.core.base_agent import AgentInput
from app.core.state import AgentState
from app.infra.providers.llm_client import create_llm_client
from app.orchestrator import registry, router
from app.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_REPLY = "Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?"
PARTIAL_FAILURE_NOTE = "\n\n(Mình chưa xử lý được một phần yêu cầu của bạn.)"


def _last_human_message(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return message.content
    return ""


async def _ingest(state: AgentState) -> AgentState:
    """Fill defaults for any field the caller didn't supply."""
    return {
        "messages": state.get("messages", []),
        "user_id": state["user_id"],
        "intents": state.get("intents", []),
        "retrieved_memories": state.get("retrieved_memories", []),
        "agent_outputs": state.get("agent_outputs", {}),
        "errors": state.get("errors", []),
        "metadata": state.get("metadata", {}),
    }


async def _rehydrate_context(state: AgentState) -> AgentState:
    """Stub — no working/episodic memory retrieval implemented yet (see CLAUDE.md)."""
    logger.debug("rehydrate_context_stub", user_id=state["user_id"])
    return {}


async def _route(state: AgentState) -> AgentState:
    intents = await router.classify_intents(state)
    return {"intents": intents}


def _fan_out(state: AgentState):
    names = router.resolve_agent_names(state["intents"])
    if not names:
        return "format_response"

    agent_input: AgentInput = {
        "user_id": state["user_id"],
        "message": _last_human_message(state["messages"]),
        "retrieved_memories": state.get("retrieved_memories", []),
    }
    return [Send("dispatch_agent", {"_agent_name": name, "_input": agent_input}) for name in names]


async def _dispatch_agent(payload: dict) -> dict:
    agent_name = payload["_agent_name"]
    agent_input: AgentInput = payload["_input"]

    try:
        agent = registry.get(agent_name)
    except KeyError:
        logger.warning("resolved_agent_not_registered", agent=agent_name)
        return {"errors": [f"{agent_name} not available"]}

    output = await agent.run(agent_input)
    return {"agent_outputs": {agent_name: output["reply"]}}


async def _compose_reply(outputs: dict[str, str]) -> str:
    try:
        llm = create_llm_client()
        joined = "\n\n".join(f"[{name}] {reply}" for name, reply in outputs.items())
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Combine the following results from different assistant "
                        "modules into a single, natural reply to the user, in the "
                        "same language the results are written in. Do not mention "
                        "the module names."
                    ),
                },
                {"role": "user", "content": joined},
            ]
        )
        return response.content
    except Exception as exc:
        logger.warning("reply_composition_failed", error=str(exc))
        return "\n\n".join(outputs.values())


async def _format_response(state: AgentState) -> AgentState:
    outputs = state["agent_outputs"]

    if not outputs:
        reply = FALLBACK_REPLY
    elif len(outputs) == 1:
        reply = next(iter(outputs.values()))
    else:
        reply = await _compose_reply(outputs)

    if state.get("errors") and outputs:
        reply += PARTIAL_FAILURE_NOTE

    return {"messages": [AIMessage(content=reply)]}


async def _episodic_writer(state: AgentState) -> AgentState:
    """Stub — no eager Qdrant write implemented yet (see CLAUDE.md)."""
    logger.debug("episodic_writer_stub", user_id=state["user_id"])
    return {}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ingest", _ingest)
    graph.add_node("rehydrate_context", _rehydrate_context)
    graph.add_node("route", _route)
    graph.add_node("dispatch_agent", _dispatch_agent)
    graph.add_node("format_response", _format_response)
    graph.add_node("episodic_writer", _episodic_writer)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "rehydrate_context")
    graph.add_edge("rehydrate_context", "route")
    graph.add_conditional_edges("route", _fan_out, ["dispatch_agent", "format_response"])
    graph.add_edge("dispatch_agent", "format_response")
    graph.add_edge("format_response", "episodic_writer")
    graph.add_edge("episodic_writer", END)

    return graph


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled
```

- [ ] **Step 2: Manually verify all four dispatch paths (mocked registry + router, no live LLM)**

Run:

```bash
uv run python -c "
import asyncio
from unittest.mock import patch

from langchain_core.messages import HumanMessage

import app.orchestrator.graph as graph_mod


class FakeAgent:
    def __init__(self, reply):
        self._reply = reply

    async def run(self, input):
        return {'reply': self._reply}


fake_agents = {'search': FakeAgent('search result'), 'journal': FakeAgent('journal result')}


def fake_get(name):
    if name not in fake_agents:
        raise KeyError(name)
    return fake_agents[name]


def resolve(intents):
    known = {'search', 'journal', 'todo'}
    return [i for i in intents if i in known]


def base_state():
    return {
        'messages': [HumanMessage(content='hi')], 'user_id': 'u1', 'intents': [],
        'retrieved_memories': [], 'agent_outputs': {}, 'errors': [], 'metadata': {},
    }


async def run_case(classify_intents, compose_llm=None):
    graph = graph_mod.build_graph().compile()
    patches = [
        patch.object(graph_mod.registry, 'get', side_effect=fake_get),
        patch.object(graph_mod.router, 'classify_intents', side_effect=classify_intents),
        patch.object(graph_mod.router, 'resolve_agent_names', side_effect=resolve),
    ]
    if compose_llm is not None:
        patches.append(patch.object(graph_mod, 'create_llm_client', return_value=compose_llm))
    for p in patches:
        p.start()
    try:
        result = await graph.ainvoke(base_state())
    finally:
        for p in patches:
            p.stop()
    last_ai = [m for m in result['messages'] if getattr(m, 'type', None) == 'ai'][-1]
    return last_ai.content, result.get('errors', [])


class FakeComposeLLM:
    async def ainvoke(self, messages):
        class R:
            content = 'combined reply'
        return R()


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError('llm down')


async def main():
    async def single(state):
        return ['search']
    print(await run_case(single))

    async def unknown(state):
        return ['unknown']
    print(await run_case(unknown))

    async def partial(state):
        return ['search', 'todo']
    print(await run_case(partial))

    async def multi(state):
        return ['search', 'journal']
    print(await run_case(multi, compose_llm=FakeComposeLLM()))
    print(await run_case(multi, compose_llm=FailingLLM()))

asyncio.run(main())
"
```

Expected output (five lines, in order):
```
('search result', [])
('Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?', [])
('search result\n\n(Mình chưa xử lý được một phần yêu cầu của bạn.)', ['todo not available'])
('combined reply', [])
('search result\n\njournal result', [])
```

- [ ] **Step 3: Do not commit**

---

### Task 5: Shrink `bot/handlers.py`'s state construction to minimal input

**Files:**
- Modify: `app/bot/handlers.py:56-85` (`handle_text`'s `state` construction only — no other changes to this file)

**Interfaces:**
- Consumes: `AgentState` (Task 1), `get_compiled_graph()` (Task 4, unchanged signature).

- [ ] **Step 1: Update the `state` construction inside `handle_text`**

Current (`app/bot/handlers.py:62-69`):

```python
    try:
        state: AgentState = {
            "messages": [HumanMessage(content=message.text)],
            "user_id": user_id,
            "intent": "",
            "retrieved_memories": [],
            "metadata": {},
        }
```

Replace with:

```python
    try:
        state: AgentState = {
            "messages": [HumanMessage(content=message.text)],
            "user_id": user_id,
        }
```

No other line in `handle_text` changes — `_ingest` (Task 4) fills every other `AgentState` field, and the rest of the function (extracting the last AI message from `result["messages"]`, error handling) is untouched.

- [ ] **Step 2: Manually verify end-to-end with the real `search` agent registered**

Run:

```bash
uv run python -c "
import asyncio
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.orchestrator import registry
import app.orchestrator.graph as graph_mod

registry.discover_and_register()


async def fake_classify(state):
    return ['search']


async def main():
    state = {
        'messages': [HumanMessage(content='Search for the latest news on AI regulation')],
        'user_id': 'u1',
    }
    with patch.object(graph_mod.router, 'classify_intents', side_effect=fake_classify):
        graph = graph_mod.get_compiled_graph()
        result = await graph.ainvoke(state)
    last_ai = [m for m in result['messages'] if getattr(m, 'type', None) == 'ai'][-1]
    print(last_ai.content)

asyncio.run(main())
"
```

Expected output: one line containing `[stub] Placeholder summary for: 'Search for the latest news on AI regulation' — search agent not yet connected to a real search/LLM provider.` — this confirms the minimal two-key input dict, passed through `_ingest`'s defaults, `_rehydrate_context`'s stub, `_route`, the `Send`-based fan-out, `_dispatch_agent` calling the real registered `SearchAgent`, and `_format_response`'s single-output pass-through all work together end-to-end.

- [ ] **Step 3: Do not commit**

## Self-Review Notes

- **Spec coverage:** all 7 design sections in `docs/superpowers/specs/2026-07-14-multi-intent-workflow-design.md` map onto a task — state/base_agent → Task 1, search module → Task 2, router → Task 3, graph → Task 4, handlers → Task 5.
- **Placeholder scan:** none — every step has complete, final file contents or an exact before/after block.
- **Type/name consistency:** `AgentState`, `AgentInput`, `AgentOutput`, `classify_intents`, `resolve_agent_names`, `_ingest`/`_rehydrate_context`/`_route`/`_dispatch_agent`/`_format_response`/`_episodic_writer` node names, and `FALLBACK_REPLY`/`PARTIAL_FAILURE_NOTE` are used identically across every task that references them.
- **Dependency order:** Task 1 (core) → Task 2 (search, needs `AgentInput`/`AgentOutput`) and Task 3 (router, needs `AgentState`) can proceed in either order once Task 1 lands → Task 4 (graph) needs Tasks 1–3 → Task 5 (handlers) needs Task 4. Dispatch in this numeric order.
