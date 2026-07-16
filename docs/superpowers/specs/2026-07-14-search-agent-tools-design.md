# Search agent: real tools, prompt, schema (reference module pattern)

Date: 2026-07-14
Status: Approved

## Problem

`search` is currently the only module wired end-to-end through the orchestrator (`SearchAgent(BaseAgent)` wrapping a 3-node `search → crawl → summary` subgraph, per `docs/superpowers/specs/2026-07-14-search-agent-and-registry-design.md`), but every node still returns hardcoded `[stub]` placeholder text — no real search, crawl, or LLM call. The module's `tools/`, `schema/`, and `prompts/` packages exist but are empty.

There is no working example in the codebase of a module that actually calls an external API, produces structured LLM output, or degrades gracefully when an external call fails. Future modules (journal, finance, insight, todo) need such a reference to copy.

## Goals

- Make `search` perform a real web search (Tavily) → crawl (Jina Reader, `r.jina.ai`) → structured LLM summary pipeline.
- Populate `tools/`, `schema/`, `prompts/` as real subpackages (not flat files — see Decision Log) that other modules can copy the shape of.
- Demonstrate layered graceful degradation: a failed search, a failed per-URL crawl, and a failed LLM summarization call must each degrade to a friendly result instead of crashing the graph or the whole request.
- Demonstrate config-driven tuning (`max_results`, `timeout_seconds`) via the module's own `config.yaml`, not global `Settings`.
- Demonstrate the existing `with_structured_output(..., method="json_mode")` pattern (already used in `orchestrator/router.py`) applied inside a module agent.

## Non-goals

- No caching of search results (e.g. in Redis/working memory) — explicitly deferred; the memory layer isn't wired into any request path yet (see CLAUDE.md).
- No pytest/test infrastructure. Verification is manual (`uv run python -c "..."`), matching the existing pattern from the search-agent-and-registry change. A real `TAVILY_API_KEY` is not available during implementation; the user will add one and verify the live happy-path themselves afterward.
- No changes to `AgentState`, `BaseAgent`, `SearchState`/`DocumentInfo`, the orchestrator graph, or the registry — this is scoped entirely inside `app/modules/search/`.
- No query rewriting/expansion, no multi-turn follow-up handling, no citation-format bikeshedding beyond a simple title+URL list.
- Not implementing journal/finance/insight/todo agents — only documenting the pattern they can copy.

## Decision Log (from brainstorming)

- **Subpackage layout, not flat files.** `tools/`, `prompts/`, `schema/` stay as packages (`tools/tavily.py`, `tools/jina.py`, `tools/_common.py`; `prompts/summary.py`; `schema/search_summary.py`) rather than flattening to `tools.py`/`prompts.py`/`schemas.py`. This means CLAUDE.md's "Adding a new module" section (which currently describes flat files) needs a doc update to match — included in this change's scope.
- **Raw `httpx` calls, no `tavily-python` SDK.** Keeps the dependency footprint minimal; `httpx` is promoted from a transitive to a direct dependency since app code now imports it.
- **Manual verification only**, no pytest — consistent with the rest of the repo today.
- **Missing/failed search and zero-results look the same to the user** (`NO_RESULTS_MESSAGE`) — a deliberate simplification. A misconfigured `TAVILY_API_KEY` will not surface a distinct error to the end user; it will be visible in logs (`logger.warning`) instead.

## Design

### 1. `tools/` — external tool wrappers

`tools/_common.py`:
- `SearchToolError` (Tavily failures), `CrawlToolError` (Jina failures) — plain `Exception` subclasses.
- `get_max_results() -> int` / `get_timeout_seconds() -> float`, reading a new `tuning:` block from `search/config.yaml` (`lru_cache`d, mirrors the existing `_load_module_specs` caching pattern in `orchestrator/router.py`).

`tools/tavily.py`:
- `async def tavily_search(query: str) -> list[dict]` — POSTs `{"api_key", "query", "max_results", "search_depth": "basic"}` to `https://api.tavily.com/search` via `httpx.AsyncClient`. Returns `[{"title", "url", "content"}, ...]` from the response's `results`. Raises `SearchToolError` if `Settings.tavily_api_key` is empty (checked before the request) or on any `httpx.HTTPError`/non-2xx response.

`tools/jina.py`:
- `async def jina_fetch(url: str) -> str` — GETs `https://r.jina.ai/<url>`, with `Authorization: Bearer <jina_api_key>` only if `Settings.jina_api_key` is set (Jina Reader works keyless, just rate-limited). Returns response text truncated to 4000 chars. Raises `CrawlToolError` on failure.

`tools/__init__.py`: re-exports `tavily_search`, `jina_fetch`, `SearchToolError`, `CrawlToolError` as the package's public surface — nodes import from `..tools`, never from `..tools.tavily` directly.

### 2. `schema/search_summary.py`

```python
class SearchSource(BaseModel):
    title: str
    url: str

class SearchSummary(BaseModel):
    answer: str                                  # concise direct answer, same language as the query
    key_points: list[str] = Field(default_factory=list)
    sources: list[SearchSource] = Field(default_factory=list)

    def render(self) -> str: ...                 # answer + bullet points + "Nguồn:" links, joined for the Telegram reply
```

This is the structured-output target for the summary LLM call — the same `with_structured_output(schema, method="json_mode")` call already used in `orchestrator/router.py`.

### 3. `prompts/summary.py`

- `SUMMARY_SYSTEM_PROMPT` — instructs the LLM to answer from the given documents only, in the query's language, and to leave `key_points`/`sources` empty if the documents don't support a confident answer.
- `NO_RESULTS_MESSAGE` — user-facing Vietnamese fallback when there are zero documents to summarize (covers both "Tavily returned nothing" and "Tavily call failed").
- `format_fallback_summary(documents: list[dict]) -> str` — non-LLM fallback (title + URL, top 3) used only if the structured LLM call itself fails after documents were successfully fetched.

### 4. Node rewrites

`node/search_node.py`:
```python
async def search_node(state: SearchState) -> dict:
    try:
        results = await tavily_search(state["user_query"])
    except SearchToolError as exc:
        logger.warning("tavily_search_failed", error=str(exc))
        return {"documents": []}
    return {"documents": [{"title": r["title"], "url": r["url"], "raw_text": r["content"]} for r in results]}
```

`node/crawl_node.py`: if `documents` is empty, pass through unchanged. Otherwise `asyncio.gather` one `jina_fetch` per doc; a per-doc `CrawlToolError` keeps that doc's existing Tavily-snippet `raw_text` instead of failing the whole crawl step.

`node/summary_node.py`: empty `documents` → return `{"summary": NO_RESULTS_MESSAGE}` (no LLM call, no network). Otherwise build a user message from `state["user_query"]` + each doc's title/url/content (content trimmed further to ~2000 chars per doc to bound prompt size), call the LLM for structured `SearchSummary`, return `{"summary": result.render()}`. On any exception from the LLM call, fall back to `{"summary": format_fallback_summary(documents)}`.

### 5. Config additions

`search/config.yaml` gains:
```yaml
tuning:
  max_results: 3
  timeout_seconds: 10
```

`app/config/settings.py` gains two optional fields (empty default, consistent with other optional secrets):
```python
tavily_api_key: str = ""
jina_api_key: str = ""
```

`.env.example` gains a new `SEARCH AGENT TOOLS` section documenting both.

`pyproject.toml`: add `"httpx>=0.28"` as an explicit dependency (currently only present transitively via other packages, per `uv.lock`).

## Error handling summary

| Failure point | Behavior |
|---|---|
| `TAVILY_API_KEY` unset or Tavily request fails | `search_node` catches `SearchToolError`, logs a warning, `documents = []` |
| Jina fetch fails for one URL | `crawl_node` keeps that doc's Tavily snippet, other docs unaffected |
| LLM structured-output call fails or times out | `summary_node` falls back to plain title/URL list, no crash |
| No documents at all (either cause above, or Tavily legitimately found nothing) | `summary_node` returns `NO_RESULTS_MESSAGE`, skips the LLM call entirely |

No exception should ever propagate out of `search_node`/`crawl_node`/`summary_node` — `SearchAgent.run()` and the orchestrator's existing top-level `try/except` (in `bot/handlers.py`) remain the last resort, unchanged by this work.

## Testing

Manual verification (no pytest, per Decision Log):
1. Without `TAVILY_API_KEY` set, run the compiled `search` subgraph directly and confirm it returns `NO_RESULTS_MESSAGE` with no exception and no network call attempted.
2. Directly unit-check `crawl_node`'s per-item fallback by feeding it a document list and patching `jina_fetch` with `unittest.mock.patch` (stdlib, not a pytest dependency) to fail for one URL, confirming the other proceeds and the failed one keeps its snippet.
3. Confirm `SearchSummary.render()` produces the expected text shape from a hand-built instance (no network).
4. Confirm `app.modules.search.agent` and `app.orchestrator.registry` still import and `discover_and_register()` + a full graph run still produce a reply with no `TAVILY_API_KEY` set (regression check against the existing search-agent-and-registry behavior).
5. Real happy-path (actual Tavily/Jina calls returning real content) is verified by the user after adding a real `TAVILY_API_KEY`/optionally `JINA_API_KEY` to `.env` — out of scope for automated verification in this change.

## Follow-ups (not in this change)

- Add `agent.py` (+ `tools/`/`prompts`/`schema/` following this same subpackage shape) for journal/finance/insight/todo.
- Redis-backed caching of search results, once the working-memory layer is wired into the request path.
- Surface a distinct "search is misconfigured" signal instead of folding it into `NO_RESULTS_MESSAGE`, if that ambiguity becomes a real problem.
- Set up pytest and backfill tests for this and prior changes (still globally deferred).

## Documentation updates required by this change

- CLAUDE.md "Adding a new module" section: replace the flat `tools.py`/`schemas.py`/`prompts.py` description with the subpackage pattern demonstrated here.
- CLAUDE.md "Current implementation gaps": remove/rewrite the "Search's own subgraph nodes are stubs" bullet to describe the real (if minimal) implementation and its fallback behavior instead.
