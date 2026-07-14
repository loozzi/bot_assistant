# LLM-based intent routing

Date: 2026-07-14
Status: Approved

## Problem

`app/orchestrator/router.py:classify_intent()` is a hardcoded keyword-matching stub with its own inline keyword tuples, diverging from the rich routing metadata (`description`, `examples`, `keywords`) already declared in every module's `config.yaml` — nothing in the codebase reads those files today. The router's `_INTENT_MAP` is also hand-maintained and has drifted (missing `"todo"`).

Separately, `orchestrator/graph.py:_dispatch()` calls `registry.get(agent_name)` with no error handling. Only `search` is currently a registered agent, so any intent that resolves to `journal`/`finance`/`insight`/`todo` raises an unhandled `KeyError` and crashes the graph. This was a latent, rarely-hit bug under keyword matching (which only routes to those labels via a handful of specific words) — smarter LLM-based classification will hit it far more often, since it will confidently route emotional/spending/planning messages to modules that don't have real agents yet.

## Goals

- Replace `classify_intent()`'s keyword matching with an LLM call, using `app/infra/providers/llm_client.py:create_llm_client()` (already implemented — LangChain `BaseChatModel`, currently backed by `ChatOpenAI` regardless of `LLM_PROVIDER`).
- Feed the LLM each module's `config.yaml` `description`, `examples`, and `keywords` as classification context — this is the first code to actually consume that metadata.
- Constrain the LLM's output to a fixed enum of known module names (or `"unknown"`) via structured output — no free-text parsing.
- Collapse every failure mode — LLM error, malformed output, or the LLM legitimately choosing `"unknown"` — to `intent = "unknown"`. `orchestrator/graph.py`'s existing `_dispatch()` already turns `agent_name is None` into the friendly fallback reply ("Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?"), so no new user-facing text is needed for that path.
- Fix `_dispatch()` so a real-but-unregistered module (`journal`, `finance`, `insight`, `todo` today) degrades to that same friendly fallback instead of crashing with `KeyError`.

## Non-goals

- No sticky-session logic (`config.yaml`'s `sticky` / `sticky_turns`) — every turn is classified fresh, same as today's keyword matcher.
- No use of `config.yaml`'s `priority` tie-breaker — irrelevant once the LLM returns exactly one label.
- No changes to `app/infra/providers/*` (`openai.py`, `llm_client.py`) — already implemented, out of scope.
- Classification input stays the latest human message only — no full conversation history, matching current behavior.
- No new `Settings` fields — reuses `llm_model` / `llm_temperature` / `llm_max_tokens` / `llm_provider` / `llm_api_key` / `llm_base_url`, all already read by `create_llm_client()`.
- No changes to `app/modules/*/config.yaml` files — read-only input to the new router.

## Design

### 1. Module spec discovery — `app/orchestrator/router.py`

```python
import functools
from pathlib import Path

import yaml

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
```

Cached with `functools.lru_cache` — `config.yaml` files don't change at runtime, matching the existing `get_settings()` `@lru_cache` pattern in `app/config/settings.py`.

**Intentional asymmetry with `registry.discover_and_register()`:** that function uses `pkgutil.iter_modules`, which only sees a directory as a package if it has `__init__.py` — `insight` lacks one today and is invisible to it. `_load_module_specs()` here globs `config.yaml` files directly and has no such requirement, so it **does** include `insight`. This is correct for routing (classification doesn't need the module to be an importable Python package, just to have declared metadata) and is called out explicitly so it isn't mistaken for a bug when the two discovery mechanisms return different module sets.

### 2. System prompt construction

```python
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
```

### 3. Structured-output schema + `classify_intent()`

```python
from typing import Literal

from pydantic import BaseModel, create_model

from app.infra.providers.llm_client import create_llm_client


def _build_intent_schema(specs: dict[str, dict]) -> type[BaseModel]:
    labels = (*specs.keys(), "unknown")
    return create_model("IntentClassification", intent=(Literal[labels], ...))


async def classify_intent(state: AgentState) -> str:
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
```

`classify_intent` becomes `async` (it already was, per its call site in `graph.py:_route` — `intent = await router.classify_intent(state)` — so this is not a new requirement on callers, just confirming the existing signature is honored now that there's a real `await`able call inside).

The broad `except Exception` is intentional and is the one designated place in this module where "the LLM is unavailable" must degrade to `"unknown"` rather than propagate — matching `CLAUDE.md`'s "agents must never crash the bot" rule. This is the resilience boundary the earlier design conversation settled on: no separate keyword-matching fallback path, no second LLM attempt — one call, and any failure of it (including the LLM confidently returning `"unknown"`) means "route to the generic fallback."

`_build_intent_schema` builds a fresh dynamic `Literal` type per call (module set can't change within a process lifetime given the `lru_cache` on `_load_module_specs`, but rebuilding the schema is cheap — a few microseconds — so there's no reason to cache it separately).

### 4. `resolve_agent_name()` simplification

Replace the hardcoded `_INTENT_MAP` entirely:

```python
def resolve_agent_name(intent: str) -> str | None:
    if intent == "unknown":
        return None
    return intent if intent in _load_module_specs() else None
```

This relies on the convention already true of every existing `config.yaml`: `agent.name` equals the module's folder name equals the key `registry.register()` stores the `BaseAgent` under. No behavior change for `search` (still resolves to `"search"`); `journal`/`finance`/`insight`/`todo` now correctly resolve to themselves instead of being absent from a hand-maintained map.

### 5. `app/orchestrator/graph.py` — `_dispatch()` no longer crashes on a known-but-unregistered module

Current (`app/orchestrator/graph.py:16-26`):

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

New — wrap the `registry.get()` call:

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

Reuses the exact same fallback string already used for the `agent_name is None` branch — from the user's perspective, "the model doesn't know what you mean" and "the model knows what you mean but that module isn't built yet" look identical, which is the correct behavior for both cases today.

## Testing

No automated tests (no pytest configured in this repo — unchanged constraint from prior work). Manual verification:

1. A search-only message ("Search for the latest news on AI regulation") through the compiled graph → `intent == "search"` → dispatches to the real `SearchAgent`, returns its stub summary.
2. A journal-style message ("Hôm nay tôi cảm thấy khá mệt mỏi") → `intent == "journal"` → `resolve_agent_name("journal") == "journal"` → `registry.get("journal")` raises `KeyError` internally → `_dispatch` returns the friendly fallback message, no crash.
3. Simulate an LLM failure (invalid/missing `LLM_API_KEY`) → `classify_intent()` catches the exception, returns `"unknown"` → `_dispatch` returns the friendly fallback message.
4. `_load_module_specs()` discovers all 5 modules (journal, finance, search, insight, todo), confirming `insight` is included despite lacking `__init__.py`.

## Follow-ups (not in this change)

- Sticky-session behavior (`config.yaml`'s `sticky`/`sticky_turns`) is still unimplemented.
- Once journal/finance/insight/todo get real `BaseAgent` implementations (following the `search` template from the prior change) and are registered, the `KeyError`-catch branch in `_dispatch` simply stops being hit for those modules — no further change needed there.
- No caching of classification results — every user message pays one LLM round trip. Acceptable for now; revisit if latency/cost becomes a concern.
