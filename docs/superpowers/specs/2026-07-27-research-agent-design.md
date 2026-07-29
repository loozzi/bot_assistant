# Research agent design

Date: 2026-07-27
Status: approved, not yet implemented

## Problem

The bot needs a `research` module: a "second brain" capability where the user
can send a link or keywords, get a summary back, have that summary persisted,
and later ask to recall what they previously looked into. Today only
`search` is dispatchable (one-shot Tavily search → optional crawl → LLM
summary, nothing persisted). `research` reuses that subgraph pattern but adds
a third capability — semantic recall — backed by Qdrant, which no module
writes to today.

## Goals

- Accept a URL → crawl it → summarize → persist.
- Accept keywords → web search → summarize → persist (same confirm-before-crawl
  UX as `search`).
- Accept a recall query ("tôi từng đọc gì về X chưa?") → semantic search over
  previously persisted research → return the saved items directly.
- All within a single `research` agent/subgraph, matching the `BaseAgent`
  contract (one `run(AgentInput) -> AgentOutput)` call per dispatch).

## Non-goals

- No Postgres table for research items — Qdrant is the only persistence
  layer for this module (decided during brainstorming: semantic recall
  matters more than structured filtering for this use case).
- No automated tests — repo has no test suite/config yet (see CLAUDE.md);
  manual verification via the running bot only.
- No dedup of previously-saved URLs/topics.
- No cross-module reuse of `search`'s Tavily/Jina wrappers — duplicated
  intentionally to avoid touching the existing `search` module.

## Architecture

`app/modules/research/` follows the `search/` subpackage pattern: `agent.py`
wraps an internal `StateGraph` behind `BaseAgent`, using
`run_interruptible_subgraph` (`app/core/hitl.py`) for the same HITL confirm
step `search` already uses.

### Graph shape

```
classify ──┬─→ link_node ──────────────────────→ crawl ──┐
           ├─→ search_node → confirm ─(full)──→ crawl ──┤
           │                        └─(skip)───────────┤→ summarize_and_save → END
           └─→ recall_node ─────────────────────────────────────────────────→ END
```

- **classify_node** — first node. LLM structured-output call
  (`schema/mode_classification.py`, `mode: new_link | new_keyword | recall`)
  decides which of the three flows this message is. On LLM failure, default
  to `new_keyword` (matches `search`'s current one-flow behavior — the safest
  fallback).
- **link_node** — `new_link` mode. Regex-extracted URL from the message
  becomes a single `DocumentInfo` (`title=""`, `raw_text=""`). Always
  proceeds to `crawl` — no confirm step, since the user explicitly gave a
  link (unlike a keyword search that might return several results).
- **search_node** — `new_keyword` mode. Calls Tavily exactly like `search`'s
  `search_node`, populates `documents` from results.
- **confirm_node** — `new_keyword` branch only. Same `interrupt()` pattern as
  `search/node/confirm_node.py` (full crawl vs. snippets-only). Skipped
  entirely for `link_node` (nothing to ask — always crawl).
- **crawl_node** — identical logic to `search/node/crawl_node.py`: parallel
  Jina fetch per document, per-URL fallback to existing `raw_text` on
  failure.
- **summarize_and_save_node** — LLM structured-output summary (`answer`,
  `key_points`, `topics`) via `schema/research_summary.py`, then:
  1. Embed the summary text via `app/infra/providers/embedding.py`
     (`"passage: " + text` prefix, per e5 convention).
  2. Upsert a point into the shared Qdrant collection
     (`app/infra/db/vector.py`'s `secondbrain` collection) via
     `tools/qdrant_store.py`, payload: `user_id`, `source_type="research"`,
     `timestamp`, `importance` (from `config.yaml`
     `memory.default_importance`), `topics`, `url`, `title`, `summary`.
  3. If the Qdrant write fails, log a warning and still return the summary —
     saving is best-effort, the user's immediate answer must not be lost.
  Sets `state["reply"]` to the rendered summary either way.
- **recall_node** — `recall` mode. Embeds the user's query
  (`"query: " + text` prefix), searches Qdrant filtered by `user_id` AND
  `source_type="research"`, takes top-K (`config.yaml` `tuning.recall_top_k`,
  default 5) above a minimum score (`tuning.recall_min_score`, default 0.5).
  Formats results directly as a list (title + saved summary + url + saved
  date) — **no LLM call** for this path, per brainstorming decision. Empty
  result set → a canned "chưa có gì được lưu về chủ đề này" message. Sets
  `state["reply"]` directly; does not touch `summarize_and_save_node`.

### State schema (`state.py`)

```python
class ResearchState(TypedDict):
    user_id: str
    user_query: str
    mode: NotRequired[Literal["new_link", "new_keyword", "recall"]]
    url: NotRequired[str]
    documents: list[DocumentInfo]   # same shape as search's DocumentInfo
    skip_crawl: NotRequired[bool]
    reply: str
```

`reply` is the unified output field (unlike `search`'s `summary`) since three
different terminal nodes (`summarize_and_save_node`, `recall_node`) both need
to set the final answer.

### `agent.py`

```python
class ResearchAgent(BaseAgent):
    async def run(self, input: AgentInput) -> AgentOutput:
        initial: ResearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": [],
            "reply": "",
        }
        result = await run_interruptible_subgraph(
            graph, initial, thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}
```

## New infra: `app/infra/providers/embedding.py`

No embedding client exists anywhere in the codebase today, even though
`Settings.embedding_model`/`embedding_device` already anticipate one
(`multilingual-e5-large` / `cpu`) and Qdrant's collection is already sized
for it (1024-dim cosine, see `app/infra/db/vector.py`). This is genuinely
shared infrastructure — not research-specific — matching how `llm_client.py`
is a shared provider client today, and how the CLAUDE.md "Memory layer
conventions" section anticipates other modules (journal, insight) writing to
Qdrant later. Placed in `infra/` rather than `modules/research/tools/`
because `infra/ → modules/` never happens but `modules/ → infra/` is exactly
the sanctioned path, and duplicating a multi-hundred-MB model load per module
would be wasteful.

```python
def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the configured sentence-transformers model."""

async def embed_text(text: str, *, prefix: str) -> list[float]:
    """Run embedding in a thread (model call is sync/CPU-bound)."""
```

`embed_text` takes an explicit `prefix` argument (`"passage: "` or
`"query: "`) rather than hardcoding it, since the e5 asymmetric convention is
call-site-specific, not a model-level constant.

New dependency: `sentence-transformers` (pulls in `torch` CPU build) added
to `pyproject.toml`.

## `tools/qdrant_store.py` (research-specific)

Thin wrapper around `app.infra.db.vector.get_qdrant_client()` with two
functions:

```python
async def upsert_research_point(user_id, url, title, summary, topics, importance) -> None
async def search_research_points(user_id, query_vector, top_k, min_score) -> list[dict]
```

Both scope every query/write with a `user_id` filter plus
`source_type == "research"` (`FieldCondition`/`MatchValue`), since the
Qdrant collection is shared across future modules.

## `config.yaml`

```yaml
agent:
  name: research
  enabled: true
  description: >
    Saves web links or keyword research (crawled + summarized) for later
    recall, and answers questions about previously saved research.

routing:
  keywords:
    vi: [lưu lại, nghiên cứu, tra cứu tài liệu, đọc lại, tóm tắt bài, đường link]
    en: [research, save this, read later, summarize this article, remember this link]
  examples:
    - "Lưu link này lại giúp tôi: https://example.com/article"
    - "Nghiên cứu giúp tôi về xu hướng AI 2026 rồi lưu lại"
    - "Tôi từng đọc gì về chủ đề blockchain chưa nhỉ?"
    - "Summarize and save this article: https://..."
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

## Error handling summary

| Failure | Behavior |
|---|---|
| Tavily/Jina call fails | Empty/fallback documents, same graceful degradation as `search` |
| LLM classify fails | Default to `new_keyword` |
| LLM summarize fails | Fallback to plain title/URL list (same pattern as `search/prompts/summary.py`) |
| Qdrant not initialized / write fails | Log warning, still return the summary reply |
| Recall finds nothing above threshold | Canned "nothing saved yet" message, no LLM call |

## Testing

No test suite exists in this repo yet (CLAUDE.md: no pytest/ruff config).
Verification is manual: run the bot locally and exercise all three flows
(send a link, send a keyword query, ask a recall question) end to end.

## Open items deferred (explicitly out of scope, noted for future work)

- Hybrid dense+sparse search / cross-encoder re-rank (full "Memory layer
  conventions" spec in CLAUDE.md) — this design only does dense cosine
  search.
- Importance decay / weekly compression jobs — same stub state as the rest
  of the memory layer (`app/bot/scheduler.py` TODOs).
- Structured Postgres index for research items — deferred; Qdrant is the
  sole store for now.
