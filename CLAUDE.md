# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telegram bot with a long-term "second brain" memory system, orchestrated through specialized LangGraph agents (journal, finance, search, insight, todo). See README.md for the feature summary and memory-layer table.

The project is early-stage. Infra plumbing (DB/Redis/Qdrant clients, settings, logging, bot skeleton, LLM provider) and the orchestrator's routing/dispatch/compose graph are built and wired together. The `search` module has a real end-to-end implementation (Tavily search → Jina crawl → LLM summary pipeline with graceful error handling). Other modules (journal, finance, insight, todo) are routable but not yet dispatchable. Read "Current implementation gaps" below before assuming a described capability produces real output.

## Commands

Dependency management is `uv` (`uv.lock` present, Python 3.10 pinned in `.python-version`).

```bash
# Install deps
uv sync

# Run the bot locally (requires .env filled in + postgres/redis/qdrant reachable)
uv run python main.py

# Start infra only (docker-compose.yml has no `app` service — the bot itself always runs locally via uv, not in Docker)
docker compose up -d postgres redis qdrant

# Alembic migrations (uses postgres_sync_dsn — psycopg2, not the app's asyncpg pool)
uv run alembic revision --autogenerate -m "message"
uv run alembic upgrade head
```

There is **no test suite, linter, or formatter configured yet** — `tests/` only contains empty `__init__.py` files (mirroring `unit/{infra,modules,orchestrator}` and `integration/`), and `pyproject.toml` has no `[tool.pytest]`, `ruff`, or `mypy` sections. Don't assume `pytest`/`ruff` commands work until they're added; add the relevant config alongside the first real test/lint setup.

`scripts/` (referenced by README's Quick Start as `scripts/seed_db.py`) does not exist yet — the directory is empty.

## Architecture

### Vertical slice layout + dependency rules

`app/modules/<name>/` is a self-contained feature slice (journal, finance, search, insight, todo). Enforced by convention, not tooling:

```
modules/  → core/, infra/     OK
modules/  → modules/          NEVER (no cross-module imports)
infra/    → core/             OK
infra/    → modules/          NEVER (infra must not know modules exist)
bot/      → orchestrator/     OK (bot only talks to orchestrator)
orchestrator/ → modules/      OK (only place allowed to know all modules, via registry — not direct imports)
```

`app/core/` holds contracts only (`base_agent.py`, `state.py`) — no implementations.

### Request flow

`bot/handlers.py` (`handle_text`) builds an `AgentState` (`app/core/state.py` — single schema, TypedDict with `messages`/`user_id`/`intents`/`retrieved_memories`/`agent_outputs`/`errors`/`metadata`) and calls `orchestrator/graph.py:get_compiled_graph().ainvoke(state)`. The reply is the last AI message in the returned state.

The compiled LangGraph (`orchestrator/graph.py`) has six nodes:

1. `ingest` — fills default field values for anything the caller didn't supply.
2. `rehydrate_context` — **stub**, no working/episodic memory retrieval implemented yet.
3. `route` — calls `orchestrator/router.py:classify_intents`, an **LLM-based, multi-label** classifier (not a keyword stub). It builds a system prompt from every enabled module's `config.yaml` (description/examples/keywords), asks the LLM for structured JSON output (`{"intents": [...]}` via `with_structured_output(..., method="json_mode")`), and falls back to `["unknown"]` on any failure.
4. `dispatch_agent` — fanned out via `langgraph.types.Send`, one branch per resolved intent (`router.resolve_agent_names`), so a single message can hit multiple module agents in parallel. Each branch calls `registry.get(agent_name).run(agent_input)`; a `KeyError` (agent classified but not registered) is caught and recorded in `state["errors"]` rather than crashing the graph.
5. `format_response` — single output passes through as-is; multiple outputs are merged by an LLM call (`_compose_reply`) that combines module replies into one natural response without naming the modules; falls back to newline-joining raw outputs if that LLM call fails.
6. `episodic_writer` — **stub**, no eager Qdrant write implemented yet.

### Agent contract

`app/core/base_agent.py` defines `AgentInput` (`user_id`, `message`, `retrieved_memories`), `AgentOutput` (`reply`), and `BaseAgent` (ABC: `name` property + `async def run(input: AgentInput) -> AgentOutput`). Agents receive only this minimal input, not the full graph `AgentState`.

`app/modules/search/agent.py` is the only implemented agent, and is also the reference pattern for **subgraph-style modules**: it builds its own internal `StateGraph` (`search → crawl → summary`, using `SearchState` from `state.py`, distinct from `AgentState`) and wraps it behind `BaseAgent.run()`, translating `AgentInput` into the subgraph's initial state and its final state back into `AgentOutput`. All three of its nodes (`node/search_node.py`, `node/crawl_node.py`, `node/summary_node.py`) currently return hardcoded `[stub]` placeholder content — no real search/crawl/LLM-summarize logic yet. `tools/` and `schema/` under `search/` are empty (`__init__.py` only); `prompts/` is an empty directory.

### Registry and auto-discovery

`orchestrator/registry.py:discover_and_register()` (called once, from `app/main.py:on_startup`) walks `app/modules/*` via `pkgutil.iter_modules`, imports `app.modules.<name>.agent` for any module that has one, and registers the module-level `agent` instance — skipping modules without `agent.py` (expected today for journal/finance/insight/todo, which only have `config.yaml` + empty `__init__.py`) and respecting `config.yaml`'s `agent.enabled: false`. This closes the gap that used to exist (no agent was ever registered) — `search` is now live end-to-end; the other four modules will need `agent.py` before they can be dispatched to, even though the router can already classify intent toward them via their `config.yaml`.

### Known inconsistency: `app/modules/todo/config.yaml`

Its header comment and `memory.source_type` block say "insight" (copy-paste from `insight/config.yaml`) even though `agent.name: todo` and the `routing` section are todo-specific. Functionally harmless today (routing keys off `agent.name`) but fix the comment/memory block together when next touching that file, don't propagate the mismatch.

### Current implementation gaps (important — don't assume these work)

- **Only `search` has an `agent.py`.** journal/finance/insight/todo are routable (their `config.yaml` feeds the LLM classifier) but not dispatchable — the graph catches the resulting `KeyError` and reports a partial-failure note rather than crashing, but no real work happens for those intents.
- **Search's subgraph nodes are real but minimal.** The `search_node` calls Tavily's web search API (gracefully handles missing `TAVILY_API_KEY` by returning empty results), `crawl_node` fetches full page content via Jina Reader with per-URL fallback (keeps original Tavily snippet if fetch fails), and `summary_node` calls the LLM with structured output for a curated answer. All three nodes degrade gracefully on network/API failures — no exception propagates out. The implementation demonstrates the subpackage pattern (`tools/`, `schema/`, `prompts/`) that other modules should copy.
- **No working/episodic memory retrieval or write.** `_rehydrate_context` and `_episodic_writer` in `orchestrator/graph.py` are no-op stubs; Redis/Qdrant clients are initialized at startup (`app/infra/memory/working.py`, `app/infra/db/vector.py`) but nothing reads or writes through them yet from the request path.
- **LLM provider registry is minimal.** `app/infra/providers/llm_client.py:create_llm_client()` branches on `settings.llm_provider`, but `anthropic`/`ollama`/`gemini` all currently fall through to `create_openai_client()` (with a comment showing the intended real implementation) — only `openai.py` (via `langchain_openai.ChatOpenAI`) is real. Don't assume setting `LLM_PROVIDER=anthropic` actually calls Anthropic's API today.
- **Postgres models are minimal; only `users` is migrated.** `app/models.py` defines `Base` (`DeclarativeBase`) and a `User` model; Alembic is wired up (`alembic/env.py` builds its engine from `Settings.postgres_sync_dsn`, a dedicated sync/psycopg2 DSN — app runtime keeps using the asyncpg pool in `app/infra/db/session.py` unchanged). Per-module tables (journal entries, finance transactions, todos, insight summaries) are still unmodeled — add them to `app/models.py` and run `uv run alembic revision --autogenerate` as each module's schema is designed.
- **Scheduler jobs are TODO stubs.** `app/bot/scheduler.py` registers the memory-decay and compression cron jobs (03:00 daily / Sunday 04:00, `Asia/Ho_Chi_Minh`) but both job bodies are `# TODO: implement in Phase 2`.
- **`scripts/seed_db.py`** referenced in README's Quick Start does not exist yet.
- **No `app` service in `docker-compose.yml`.** README's "Full stack via Docker: `docker compose up app`" doesn't currently work — the compose file only defines `postgres`/`redis`/`qdrant`; the bot process itself is run with `uv run python main.py`, not containerized.

### Memory layer conventions (target design — mostly not yet implemented)

Three backends, each with a distinct role: Redis (`infra/memory/working.py`, session TTL), Qdrant (`infra/db/vector.py`, permanent vector search, collection created lazily on `init_vector_db()` with 1024-dim cosine vectors for `multilingual-e5-large`), Postgres (structured facts — not yet modeled beyond `users`).

Intended Qdrant payload/retrieval conventions to follow once ingestion is built:
- Semantic chunking only (200–400 tokens, 50-token overlap) — never fixed character-count splitting.
- Every point carries `user_id`, `source_type`, `timestamp`, `importance`, `mood_score` (journal only), `topics`, `language`.
- Hybrid search: dense (cosine) + sparse (BM25), weighted 0.7/0.3; always filter by `user_id` first; re-rank top-20 with a cross-encoder before context assembly.
- `importance` decays 10%/30 days if unretrieved; weekly job compresses chunks below 0.2 importance into summaries (this is what the scheduler stubs above are for).

### Adding a new module

1. `app/modules/<name>/` needs `agent.py` (subclass `BaseAgent`, implement `async def run(input: AgentInput) -> AgentOutput`). For modules that build internal subgraphs or have external tool calls, follow the subpackage pattern from `search/`:
   - `tools/` — external API wrappers (e.g., `tools/tavily.py`, `tools/jina.py`, `tools/_common.py` for shared error classes/config loaders).
   - `schema/` — Pydantic models for structured LLM output (e.g., `schema/search_summary.py`).
   - `prompts/` — system prompts and fallback messages (e.g., `prompts/summary.py`).
   - `node/` — subgraph node implementations (for subgraph-style modules).
   - `state.py` — subgraph-specific state schema (for subgraph-style modules).
   `config.yaml` already exists for all five modules — match its shape (`agent.name/enabled/description`, `routing.keywords/examples/sticky/sticky_turns/priority`, `memory.source_type/default_importance`) when adding new fields like `tuning:`.
2. No registry wiring needed — `discover_and_register()` in `app/main.py:on_startup` picks up any module with an `agent.py` automatically. Follow `search/agent.py` as the reference implementation for both simple agents and subgraph-style modules.
3. Routing is already LLM-driven via each module's `config.yaml` (`orchestrator/router.py`); a new module becomes classifiable as soon as its `config.yaml` exists — no router code changes needed unless you're changing classification behavior itself.
4. Never import another `modules/*` package — shared logic goes in `infra/` or `utils/`.

### Config/env

`app/config/settings.py` (`pydantic-settings`) is the single source of truth for env vars — see it directly rather than `.env.example` for defaults and which fields are required vs optional (only `telegram_bot_token`, `postgres_user`, `postgres_password` have no default).

### Planning docs

`docs/superpowers/{plans,specs}/` holds design specs and implementation plans (dated, e.g. `2026-07-14-llm-based-router.md`) written before larger features landed — useful for the *why* behind the router/registry/Postgres design, but treat them as historical design intent, not live status; cross-check against the code before relying on a claim there.
