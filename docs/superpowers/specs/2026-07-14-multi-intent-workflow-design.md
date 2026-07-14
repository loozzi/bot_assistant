# Multi-Intent Workflow: Isolated Modules + Parallel Dispatch

Date: 2026-07-14
Status: Approved

## Problem

The orchestrator forces exactly one intent per message (`orchestrator/graph.py`'s `_route` → `_dispatch`), so compound requests ("ghi journal + tạo todo mai") can only ever be routed to one module. Separately, `BaseAgent.run(state: AgentState) -> AgentState` hands every module the orchestrator's *entire* state and lets it return a full merged state back — meaning modules implicitly depend on and can mutate bookkeeping fields (`messages`, `metadata`) that belong to the orchestrator, not to them. That coupling is workable for one agent running at a time, but breaks down for parallel dispatch (concurrent branches can't each return a full merged state safely) and makes it awkward for a future module with a fundamentally different internal task (e.g. a `FinanceAgent` tracking amounts/categories/receipt OCR) to stay isolated from `search`'s or `journal`'s concerns.

`BaseState` (`app/core/state.py`, used today only by `search`'s internal subgraph via `SearchState`) also overlaps `AgentState`'s fields — CLAUDE.md documents this as two competing, incompatible schemas.

## Goals

- Support multiple intents per message, dispatched to their respective agents in parallel.
- Fully isolate module state and I/O: modules receive a minimal `AgentInput` and return a minimal `AgentOutput`. They never see the orchestrator's `intents`/`agent_outputs`/`errors` bookkeeping, and never touch the shared `messages`/`agent_outputs` channels directly — only the orchestrator's dispatch node does.
- New modules keep fanning out automatically via the existing `registry.discover_and_register()` — no `graph.py` edits per module, matching CLAUDE.md's "Adding a new module" checklist.
- Preserve graceful degradation at every layer: LLM classification failure → `["unknown"]` (unchanged behavior); one intent's agent missing/erroring → the others still run, a soft note is appended; zero intents resolved → today's exact fallback message, unchanged.

## Non-goals

- Real Redis/Qdrant retrieval or episodic writes — `rehydrate_context` and `episodic_writer` are honest no-op stub nodes this round. `infra/memory/working.py` and `infra/db/vector.py` only set up client connections today (confirmed by reading both files) — there's no chunking, embedding, or retrieval/write helper anywhere yet. Wiring real logic is a separate future spec against CLAUDE.md's already-documented memory-layer conventions.
- Sticky-session behavior (`config.yaml`'s `sticky`/`sticky_turns`) — still out of scope, unchanged from the earlier LLM-router spec.
- Conversation history passed to modules — `AgentInput.message` is just the latest human message text, matching current behavior (no multi-turn context fed to agents yet).
- No new `Settings`/env fields, no changes to `app/infra/providers/*`.

## Design

### 1. `app/core/state.py` — orchestrator-only state, `BaseState` removed

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

`BaseState` is deleted, not merged — it previously overlapped `AgentState` and was the only thing coupling `search`'s internal state to `core`. `agent_outputs`'s reducer changes from `operator.add` (broken — `dict` has no `__add__`; the first time two parallel branches actually needed to merge, this would raise `TypeError`) to `operator.or_` (`dict.__or__`, available since Python 3.9).

### 2. `app/core/base_agent.py` — isolated I/O contract

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

Modules never see `AgentState` at all now. This is the isolation the design targets: a future `FinanceAgent` can define whatever internal state it needs without touching or being constrained by the orchestrator's bookkeeping.

### 3. `app/modules/search/state.py` — standalone, no core inheritance

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

Drops `messages` (the `add_messages` machinery) and `chat_id` — confirmed by reading `search_node.py`/`crawl_node.py`/`summary_node.py` that none of the three internal nodes reference either field, only `user_query` and `documents`.

### 4. `app/modules/search/agent.py` — updated `run()`

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

No changes needed to `search_node.py`, `crawl_node.py`, or `summary_node.py` — they only ever read `user_query`/`documents`.

### 5. `app/orchestrator/router.py` — multi-intent classification

```python
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

`_build_system_prompt`'s copy needs two clauses updated: "choose exactly one module" → "choose every module that applies — most messages need only one, but choose more than one when the message clearly asks for multiple distinct things", and the JSON-shape instruction sentence → `Respond only with a JSON object of the form {"intents": ["<module_name>", ...]}.`. `_load_module_specs` and `_build_system_prompt`'s per-module rendering loop are otherwise unchanged. `classify_intent` (singular) and the old `_build_intent_schema`'s single-`Literal` shape are replaced, not kept alongside the new function.

### 6. `app/orchestrator/graph.py` — full rewrite

```python
from langchain_core.messages import AIMessage, HumanMessage
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

`errors` is only read in `_format_response`, never reset — since a fresh `AgentState` is built per incoming Telegram message (`bot/handlers.py` never reuses state across turns), this is safe; it is not a cross-turn accumulator.

### 7. `app/bot/handlers.py` — shrinks to minimal input

```python
state: AgentState = {
    "messages": [HumanMessage(content=message.text)],
    "user_id": user_id,
}
```

`_ingest` fills every other field (`intents`, `retrieved_memories`, `agent_outputs`, `errors`, `metadata`), so callers no longer need to know the full `AgentState` shape. The rest of `handle_text` (extracting the last AI message from `result["messages"]`) is unchanged.

## Testing

Manual verification only (no pytest configured):

1. Single-intent message ("Search for the latest news on AI regulation") → `intents == ["search"]` → one `Send` → `agent_outputs == {"search": "..."}` → `_format_response` passes it through with **no LLM compose call** (verify via mock/log — this is the single-intent fast path, matching today's latency/cost).
2. Multi-intent message combining a resolvable and unresolvable module (e.g. journal + todo, only `search` registered today — substitute two real module names once more agents exist) → both dispatch in parallel → `errors == ["<name> not available"]`, `agent_outputs` contains only the resolvable one's reply → final reply is that content plus the partial-failure note.
3. Fully unresolved message ("asdkfjhasdf") → `intents == ["unknown"]` → `_fan_out` returns `"format_response"` directly (no `Send`, `_dispatch_agent` never invoked) → reply is exactly `FALLBACK_REPLY`, unchanged from today.
4. Two resolvable intents both registered (once a second module has a real agent) → `_format_response` takes the `len(outputs) > 1` branch → confirm one extra LLM call, both `[name] reply` blocks present in the prompt sent to it.
5. `SearchAgent.run()` in isolation: `await agent.run({"user_id": "u1", "message": "test query", "retrieved_memories": []})` returns `{"reply": "..."}` — confirms the module never touches `AgentState` shape at all.
6. Simulate an LLM failure in `_compose_reply` (invalid/missing API key with 2+ agent_outputs) → falls back to `"\n\n".join(outputs.values())`, no crash.

## Follow-ups (not in this change)

- Wire real Redis/Qdrant logic into `_rehydrate_context` / `_episodic_writer` per CLAUDE.md's memory-layer conventions.
- Sticky-session behavior.
- `AgentInput` may need to grow (e.g. conversation history) once a module actually needs multi-turn context — add then, not now.
