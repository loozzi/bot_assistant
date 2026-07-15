# Human-in-the-loop (HITL) for subgraph-style modules

Date: 2026-07-15
Status: Approved

## Problem

Neither the orchestrator graph (`orchestrator/graph.py`) nor any module subgraph (only `search/agent.py` exists today) is compiled with a checkpointer. `bot/handlers.py:handle_text` calls `graph.ainvoke(state)` once per Telegram message and always runs the graph to completion in a single shot — there is no way for a node to pause mid-turn, ask the user a question via Telegram (button or free-text reply), and resume exactly where it left off on the next message.

This spec adds a generic human-in-the-loop mechanism, using LangGraph's `interrupt()`/`Command(resume=...)` primitives, and a minimal proof-of-concept in the `search` module (a Yes/No confirmation before crawling full page content) to prove the mechanism end-to-end.

## Goals

- A reusable pattern any subgraph-style module (not just `search`) can use to pause for human input without losing work already done inside that subgraph.
- Telegram UX: the bot can ask a question via inline keyboard buttons **and** accept a plain-text reply as the answer.
- An explicit `/cancel` escape hatch when the user wants to abandon a pending question and start a new topic instead.
- A working PoC in `search`: pause after the `search` node, ask "read full pages or just summarize snippets?", and skip `crawl` entirely if the user declines.

## Non-goals

- Chaining multiple sequential interrupts within a single subgraph invocation (ask A, resume, ask B, resume). The helper introduced here supports exactly one pending pause per subgraph run; multi-question flows are future work.
- Persisting HITL state beyond the existing Redis session TTL (`redis_ttl_session`, 4h) — no new durability tier.
- Adding HITL to any module other than `search`.
- Formal test suite (`tests/` remains unset up per CLAUDE.md; this spec doesn't change that).

## Design

### 1. Checkpointer infra — `app/infra/memory/checkpointer.py` (new)

New dependency: `langgraph-checkpoint-redis`. Wraps `AsyncRedisSaver` with the same lifecycle shape as `infra/memory/working.py`:

```python
_saver: AsyncRedisSaver | None = None

async def init_checkpointer(settings: Settings | None = None) -> None: ...
async def close_checkpointer() -> None: ...
def get_checkpointer() -> AsyncRedisSaver: ...
```

`init_checkpointer()` calls `AsyncRedisSaver`'s one-time `.asetup()` (creates required Redis index structures) and configures TTL from `settings.redis_ttl_session` (seconds → minutes) so abandoned pauses expire automatically, consistent with existing session semantics.

**Wiring in `app/main.py`:** `init_checkpointer(settings)` runs in `on_startup`, immediately after `init_redis()` and **before** `registry.discover_and_register()` — module `agent.py` files compile their subgraph (which now needs the checkpointer) at import time, and `discover_and_register()` is what triggers that import. `close_checkpointer()` added to `on_shutdown` alongside `close_redis()`.

### 2. Shared contract + bridge helper — `app/core/hitl.py` (new)

Deliberate deviation from CLAUDE.md's "`app/core/` holds contracts only" rule: the bridge helper lives here too, colocated with the contracts it operates on, because every subgraph-style module needs both together. (CLAUDE.md gets a note about this exception once implemented.)

```python
class HumanReviewOption(TypedDict):
    label: str
    value: str

class HumanReviewRequest(TypedDict):
    question: str
    options: NotRequired[list[HumanReviewOption]]


async def run_interruptible_subgraph(
    graph: CompiledStateGraph,
    initial_state: dict,
    *,
    thread_id: str,
) -> dict:
    """Run a checkpointed subgraph, bridging any pause up to the parent graph.

    Must be awaited from inside a node of a checkpointed parent graph — the
    bridging `interrupt()` call relies on that parent's checkpoint context.
    Supports exactly one pending pause per invocation (see Non-goals).
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot.next:  # paused by a previous attempt at this same call site
        pending = snapshot.tasks[0].interrupts[0].value
        answer = interrupt(pending)  # returns instantly on replay, else re-pauses parent
        result = await graph.ainvoke(Command(resume=answer), config=config)
    else:
        result = await graph.ainvoke(initial_state, config=config)

    snapshot = await graph.aget_state(config)
    if snapshot.next:  # just paused for the first time
        interrupt(snapshot.tasks[0].interrupts[0].value)  # bridges the pause upward

    return result
```

**Why this works (the "bridge" mechanism):** the subgraph gets its *own* checkpointer (same Redis-backed `AsyncRedisSaver` instance as the outer graph, different `thread_id`). When a subgraph node calls `interrupt(payload)`, the subgraph pauses gracefully and `ainvoke()` returns normally (not raising) because it has a checkpointer. `run_interruptible_subgraph()` detects this and calls `interrupt(payload)` itself — this second call executes in the *parent* graph's checkpoint context (since the helper runs inside a parent node), so it raises `GraphInterrupt`, uncaught by anything in between, and is caught by the parent graph's own checkpointed Pregel loop, pausing the whole turn. On resume (`Command(resume=answer)` at the parent level), the parent node re-executes from scratch, the helper's `aget_state()` check finds the subgraph still paused, and its own `interrupt(pending)` call — at the same call position as before — returns `answer` immediately instead of pausing again. The helper then resumes the *subgraph* precisely from where it paused via `Command(resume=answer)`, so any subgraph work done before the pause (e.g. a finished `search` node) is not re-run.

**Thread ID scheme:** outer graph uses `thread_id = user_id`; each subgraph invocation uses `thread_id = f"{user_id}:{agent_name}"`. Stable and reusable across unrelated turns — when *not* paused, invoking with fresh `initial_state` simply restarts the subgraph from its entry node and overwrites prior fields (no accumulating reducers in `SearchState`), matching how the outer graph already reuses `thread_id=user_id` across every turn.

### 3. Outer graph — `app/orchestrator/graph.py`

`get_compiled_graph()`: `build_graph().compile(checkpointer=get_checkpointer())`. No other changes — `_dispatch_agent`'s `except KeyError` only wraps `registry.get(agent_name)`, not `await agent.run(...)`, so a bridged `GraphInterrupt` propagates out untouched.

### 4. Search module PoC

- `app/modules/search/state.py`: add `skip_crawl: NotRequired[bool]` to `SearchState`.
- `app/modules/search/node/confirm_node.py` (new): after `search` populates `documents`, call
  `interrupt(HumanReviewRequest(question="Tìm thấy N kết quả. Đọc chi tiết từng trang hay chỉ tóm tắt nhanh?", options=[{"label": "Đọc chi tiết", "value": "yes"}, {"label": "Tóm tắt nhanh", "value": "no"}]))` and set `skip_crawl` from the answer.
- `app/modules/search/agent.py`:
  - Add `confirm` node between `search` and `crawl`; conditional edge after `confirm` routes to `crawl` when `skip_crawl` is falsy, else straight to `summary`.
  - `graph = builder.compile(checkpointer=get_checkpointer())`.
  - `SearchAgent.run()` calls `run_interruptible_subgraph(graph, initial, thread_id=f"{input['user_id']}:{self.name}")` instead of `graph.ainvoke(initial)` directly. `AgentInput`/`AgentOutput` contracts stay unchanged — the interrupt bypasses the normal return-value path entirely.

### 5. Bot handlers — `app/bot/handlers.py`, `app/bot/keyboards.py`

- `keyboards.py`: add `hitl_keyboard(options: list[HumanReviewOption]) -> InlineKeyboardMarkup`, buttons with `callback_data=f"hitl:{value}"`.
- `handlers.py`: refactor into a shared `_run_turn(user_id, *, new_text=None, resume_answer=None) -> AgentState` used by all three entry points below, to avoid duplicating the "invoke → check `__interrupt__` → render" logic three times.
  - `handle_text`: build `config={"configurable": {"thread_id": user_id}}`; call `graph.aget_state(config)` first — if `.next` is non-empty, treat the incoming text as `resume_answer`; otherwise it's a fresh turn (`new_text=message.text`).
  - New `@router.callback_query(F.data.startswith("hitl:"))`: extract the value after `hitl:`, call `_run_turn(user_id, resume_answer=value)`.
  - After `_run_turn` returns, check `result.get("__interrupt__")`: if present, render the `HumanReviewRequest.question` as text with `hitl_keyboard(options)` attached (if `options` present) instead of the normal composed reply.
  - New `@router.message(Command("cancel"))`: delete the outer thread's checkpoint state and, for every `name in registry.all_agents()`, delete the `f"{user_id}:{name}"` subgraph checkpoint too, then reply confirming cancellation. Necessary because thread IDs are reused across turns (§2) — a stale pause left behind would make the next unrelated message misread as a resume.

## Known limitations

- **Single pause per subgraph run.** See Non-goals — a module needing two sequential questions in one subgraph invocation isn't supported by `run_interruptible_subgraph()` yet.
- **`Send` fan-out interaction untested.** If one message classifies to multiple intents and one branch (e.g. `search`) pauses, LangGraph is expected to cache the other already-completed branch's output and only replay the paused branch on resume — this is standard Pregel task-level checkpointing behavior, but should get an explicit test case when implemented, since it hasn't been exercised in this codebase before.
- **Stable subgraph thread IDs require `/cancel` to clean up both layers.** Documented in §5; forgetting the subgraph-side cleanup would leave a dangling pause that corrupts the next unrelated turn.
- **Redis TTL is the only expiry mechanism.** No scheduler job cleans up abandoned pauses early; they simply age out with the existing 4h session TTL, after which `aget_state()` returns an empty snapshot and the next message is treated as a fresh turn with no special-casing needed.
- **Startup ordering is load-bearing.** `init_checkpointer()` must run before `registry.discover_and_register()`, since `search/agent.py` now needs `get_checkpointer()` to already return a real instance at import time.
- **Intentional CLAUDE.md deviation.** `app/core/hitl.py` mixes contract types with an implementation (`run_interruptible_subgraph`), against the stated "core/ = contracts only" rule — accepted for simplicity per explicit user direction; CLAUDE.md should get a one-line callout for this exception when the code lands.
