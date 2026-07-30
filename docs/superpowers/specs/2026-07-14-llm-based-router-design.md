# LLM-based intent routing

Date: 2026-07-14
Status: Implemented and superseded by the multi-intent workflow

## Purpose

The orchestrator routes each Telegram turn to the feature modules that can
handle it. Routing is driven by enabled modules' declarative `config.yaml`
metadata rather than a hand-maintained keyword map. It supports one or more
intents per turn, isolates each module behind the `BaseAgent` contract, and
degrades safely when classification, memory, or an agent is unavailable.

The authoritative implementation is in:

- `app/orchestrator/router.py`
- `app/orchestrator/graph.py`
- `app/orchestrator/registry.py`
- `app/infra/memory/episodic.py`

Related designs cover the later extensions in more detail:
`2026-07-14-multi-intent-workflow-design.md`,
`2026-07-15-hitl-subgraph-design.md`, and
`2026-07-27-research-agent-design.md`.

## Current behavior

1. The graph ingests a turn and retrieves relevant episodic memories for the
   latest human message.
2. `router.classify_intents()` asks the LLM to return every applicable enabled
   module name as JSON.
3. The graph resolves valid, distinct module names and fans out one
   `dispatch_agent` branch per name through LangGraph `Send`.
4. Each branch receives only an `AgentInput` containing the user ID, latest
   message, and memories scoped to that module's `memory.source_type`.
5. Successful module replies are merged. One reply is passed through; several
   replies are composed by the LLM; no replies produce the generic fallback.
6. Successful replies are written to episodic memory on a best-effort basis.

```
ingest → rehydrate_context → route → dispatch_agent × N → format_response → episodic_writer → END
                                     │
                                     └─ no resolved module → format_response
```

`search` and `research` currently provide registered agents. `journal`,
`finance`, `insight`, and `todo` are still classifiable from their configs but
have no `agent.py`; they are therefore handled as per-intent soft failures.

## Module discovery and metadata

`_load_module_specs()` caches enabled `app/modules/*/config.yaml` files with
`functools.lru_cache(maxsize=1)`. It reads:

```python
{
    "description": agent.description,
    "examples": routing.examples,
    "keywords": routing.keywords,
    "source_type": memory.source_type,
    "default_importance": memory.default_importance,
}
```

The current enabled labels are `finance`, `insight`, `journal`, `research`,
`search`, and `todo`. Discovery globs configuration files directly, so it does
not require a module to be an importable Python package. In particular,
`insight` participates in classification even though registry discovery cannot
import an agent for it.

`registry.discover_and_register()` has a different responsibility: it imports
only package modules with an `agent.py` exporting a `BaseAgent`, then respects
`agent.enabled`. The router must not use registry membership to constrain the
classifier, because classifying a future module before its implementation is
valid and must fail gracefully at dispatch time.

## Classification

For the latest message whose LangChain type is `human`, the router builds a
system prompt from every discovered module's description, examples, and
Vietnamese/English keywords. It tells the model to select every module that
clearly applies, or `unknown` when none does.

The output schema is dynamically constrained to discovered labels plus
`unknown`:

```python
labels = (*specs.keys(), "unknown")
IntentClassification = create_model(
    "IntentClassification",
    intents=(list[Literal[labels]], ...),
)
```

The classifier uses JSON mode:

```python
classifier = llm.with_structured_output(IntentClassification, method="json_mode")
result = await classifier.ainvoke(messages)
```

The explicit prompt shape is:

```json
{"intents": ["<module_name>", "..."]}
```

JSON mode is required because the configured gateway/model may not honor
function/tool-calling semantics. It still leaves Pydantic responsible for
validating labels and the response shape.

`classify_intents()` returns `["unknown"]` when there is no human message, the
model returns an empty list, or any LLM/transport/structured-output failure is
raised. The broad exception boundary is deliberate: an unavailable LLM must
not crash the bot. There is no keyword-matching fallback and no retry.

`resolve_agent_names()` removes `unknown`, excludes labels no longer present in
the cached metadata, and preserves first-seen order while deduplicating. Thus,
an `unknown` item cannot prevent other valid intents in the same model response
from being dispatched.

## Dispatch, aggregation, and failures

Each resolved intent creates an independent `Send` payload:

```python
{
    "_agent_name": name,
    "_input": {
        "user_id": state["user_id"],
        "message": latest_human_message,
        "retrieved_memories": memories_for_that_source_type,
    },
}
```

Modules do not receive the orchestrator's `AgentState`; they implement
`BaseAgent.run(AgentInput) -> AgentOutput`. Parallel branch outputs merge into
the `agent_outputs` dictionary and branch failures append to `errors`.

If an intent is classifiable but absent from the runtime registry,
`registry.get()` raises `KeyError`. `_dispatch_agent()` catches it, logs
`resolved_agent_not_registered`, and returns an error instead of raising. A
registered agent's own exception is intentionally not caught here, so LangGraph
interrupts used by HITL-capable subgraphs can propagate and pause the outer
turn correctly.

`_format_response()` behaves as follows:

| Result | User-visible behavior |
|---|---|
| No successful outputs | `Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?` |
| One output | Return that reply unchanged |
| Multiple outputs | Ask the LLM to combine them without naming modules; newline-join them if that call fails |
| Any errors plus one or more outputs | Append `(Mình chưa xử lý được một phần yêu cầu của bạn.)` |

This means a turn routed solely to an unimplemented module shows the generic
fallback; a compound turn can still return the work completed by other modules
with a partial-failure note.

## Memory integration

Before routing, `_rehydrate_context()` calls
`episodic.retrieve_memories(user_id, latest_message, source_types)` for every
`source_type` declared by enabled modules. Retrieval:

- checks an exact-query Redis cache first;
- embeds the message and queries Qdrant independently per source type;
- filters every Qdrant query by `user_id` and `source_type`;
- returns up to three matches per source type and updates their
  `last_accessed` timestamps; and
- falls back to no context when the memory backend fails.

The fan-out node passes each agent only the payloads for its own source type.
After response formatting, `_episodic_writer()` writes each successful module
reply independently. It chunks and embeds replies, stores standard Qdrant
metadata, and uses the module's `default_importance`. Journal, search, and
finance writes add their respective mood, URL, and expense fields. A failed
write is logged and does not affect the reply.

The `research` agent also persists its own curated research records; this is
separate from the orchestrator's general reply-memory write.

## Deliberate limitations

- Routing is fresh on every non-resumed turn; `sticky` and `sticky_turns` in
  module configs are not implemented.
- `routing.priority` is not used because the LLM, rather than a keyword
  scorer, selects labels.
- The classifier sees only the latest human message, not full conversation
  history.
- Memory retrieval is dense-vector retrieval with an exact-query cache; it
  does not yet implement the planned sparse/BM25 hybrid search or reranking.
- Only `search` and `research` are dispatchable today. The other enabled
  modules need real `BaseAgent` implementations before their classifications
  yield a substantive reply.
- The LLM client currently delegates every configured provider branch to the
  OpenAI-compatible client; provider-specific clients remain future work.

## Verification

There is not yet an automated test suite. Manual verification should cover:

1. A search request classifies to `search`, reaches the search subgraph, and
   either returns a reply or pauses for its crawl-choice HITL prompt.
2. A research save/recall request classifies to `research` and dispatches to
   the research subgraph.
3. A compound request produces multiple distinct intents and invokes one
   dispatch branch per resolvable module.
4. A journal/finance/insight/todo-only request does not crash despite its
   missing agent; it produces the generic fallback.
5. A mixed request with one registered and one missing module retains the
   registered reply and appends the partial-failure note.
6. A missing/invalid LLM configuration returns `["unknown"]` and yields the
   generic fallback without propagating an exception.
7. Qdrant, Redis, or embedding failures during retrieval/write are logged but
   do not prevent routing or a successful agent reply.
