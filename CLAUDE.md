# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telegram bot with a long-term "second brain" memory system, orchestrated through specialized LangGraph agents (journal, finance, search, insight, todo). See README.md for the feature summary and memory-layer table.

The project is early-stage: infra plumbing (DB/Redis/Qdrant clients, settings, logging, bot skeleton) is built, but the orchestrator routing and most module agents are stubs. Read "Current implementation gaps" below before assuming a described capability is wired up.

## Commands

Dependency management is `uv` (`uv.lock` present, Python 3.10 pinned in `.python-version`).

```bash
# Install deps
uv sync

# Run the bot locally (requires .env filled in + postgres/redis/qdrant reachable)
uv run python main.py

# Start infra only
docker compose up -d postgres redis qdrant

# Full stack via Docker
docker compose up
```

There is **no test suite, linter, or formatter configured yet** — `tests/` only contains empty `__init__.py` files, and `pyproject.toml` has no `[tool.pytest]`, `ruff`, or `mypy` sections. Don't assume `pytest`/`ruff` commands work until they're added; add the relevant config alongside the first real test/lint setup.

## Architecture

### Vertical slice layout + dependency rules

`app/modules/<name>/` is a self-contained feature slice (journal, finance, search, insight, todo). Enforced by convention, not tooling:

```
modules/  → core/, infra/     OK
modules/  → modules/          NEVER (no cross-module imports)
infra/    → core/             OK
infra/    → modules/          NEVER (infra must not know modules exist)
bot/      → orchestrator/     OK (bot only talks to orchestrator)
orchestrator/ → modules/      OK (only place allowed to know all modules)
```

`app/core/` holds contracts only (`base_agent.py`, `state.py`) — no implementations.

### Request flow

`bot/handlers.py` (`handle_text`) builds an `AgentState`, calls `orchestrator/graph.py:get_compiled_graph()`, and replies with the last AI message. The compiled LangGraph has exactly two nodes: `route` (calls `orchestrator/router.py:classify_intent`) → `dispatch` (calls `orchestrator/registry.py:get(agent_name).run(state)`).

### Current implementation gaps (important — don't assume these work)

- **Routing is a keyword stub, not config-driven.** Each module ships a `config.yaml` with rich routing metadata (keywords, few-shot examples, sticky-session behavior, priority) — see `app/modules/*/config.yaml`. Nothing loads these files yet. `orchestrator/router.py` has its own hardcoded `_INTENT_MAP` and keyword lists that don't match the yaml (e.g. `todo` isn't in `_INTENT_MAP` at all). Treat the yaml files as a routing spec to implement against, not as active config.
- **No agent is registered.** `orchestrator/registry.register()` is never called anywhere in the codebase. Any successfully classified intent will hit `registry.get(agent_name)` and raise `KeyError` inside `_dispatch`. Journal/finance/insight/todo modules currently have only `config.yaml` + empty `__init__.py` — no `agent.py` yet.
- **Two competing state schemas.** `app/core/state.py` defines both `AgentState` (used by the orchestrator graph and `bot/handlers.py`) and `BaseState` (used by `app/modules/search/state.py:SearchState`). They overlap but aren't the same shape (`AgentState` has `intent`/`metadata`; `BaseState` has `intends`/`agent_outputs`/`errors`). Check which one a given file actually imports before adding fields — don't assume they're interchangeable.
- **`search` doesn't implement `BaseAgent`.** Unlike the `BaseAgent.run(state) -> state` contract in `core/base_agent.py`, `app/modules/search/agent.py` builds its own standalone `StateGraph` (search → crawl → summary nodes, all currently `return {}` stubs) and is not wired into `registry`/`router` at all. If you finish the search module, decide whether it becomes a `BaseAgent` wrapping this subgraph, or whether the `BaseAgent` contract changes to accommodate subgraph-style modules — don't silently leave two patterns.
- **`search/agent.py` loads `config.yaml` via a relative path** (`open("./config.yaml")`), which only resolves if the process cwd happens to be `app/modules/search/`. This will break under normal execution from the repo root — fix to a path relative to `__file__` when touching this file.
- **No LLM provider implementation.** `.env`/`Settings` have `llm_provider`/`llm_api_key`/`llm_model` fields and `anthropic` is a dependency, but there's no `BaseLLMProvider`, no `app/infra/providers/*.py` beyond an empty `__init__.py`, and no registry. Router's intent classification is pure keyword matching (no LLM call yet).
- **Postgres models are minimal; only `users` is migrated.** `app/models.py` defines `Base` (`DeclarativeBase`) and a `User` model; Alembic is wired up (`alembic/env.py` builds its engine from `Settings.postgres_sync_dsn`, a dedicated sync/psycopg2 DSN — app runtime keeps using the asyncpg pool in `app/infra/db/session.py` unchanged). Per-module tables (journal entries, finance transactions, todos, insight summaries) are still unmodeled — add them to `app/models.py` and run `uv run alembic revision --autogenerate` as each module's schema is designed.
- **Scheduler jobs are TODO stubs.** `app/bot/scheduler.py` registers the memory-decay and compression cron jobs (03:00 daily / Sunday 04:00, `Asia/Ho_Chi_Minh`) but both job bodies are `# TODO: implement in Phase 2`.
- **`scripts/seed_db.py`** referenced in README's Quick Start does not exist yet.

### Memory layer conventions (target design — mostly not yet implemented)

Three backends, each with a distinct role: Redis (`infra/memory/working.py`, session TTL), Qdrant (`infra/db/vector.py`, permanent vector search, collection created lazily with 1024-dim cosine vectors for `multilingual-e5-large`), Postgres (structured facts — not yet modeled).

Intended Qdrant payload/retrieval conventions to follow once ingestion is built:
- Semantic chunking only (200–400 tokens, 50-token overlap) — never fixed character-count splitting.
- Every point carries `user_id`, `source_type`, `timestamp`, `importance`, `mood_score` (journal only), `topics`, `language`.
- Hybrid search: dense (cosine) + sparse (BM25), weighted 0.7/0.3; always filter by `user_id` first; re-rank top-20 with a cross-encoder before context assembly.
- `importance` decays 10%/30 days if unretrieved; weekly job compresses chunks below 0.2 importance into summaries (this is what the scheduler stubs above are for).

### Adding a new module

1. `app/modules/<name>/` with `agent.py` (subclass `BaseAgent`, implement `async def run(state) -> state`), `schemas.py`, `tools.py`, `prompts.py`, `config.yaml` (match the shape used by existing modules — `agent.name/enabled/description`, `routing.keywords/examples/sticky/priority`, `memory.source_type/default_importance`).
2. Call `orchestrator/registry.register()` somewhere it actually runs at startup — currently no module does this, so you're the first.
3. Wire actual routing in `orchestrator/router.py` (today's `_INTENT_MAP` + keyword lists are what you're extending/replacing).
4. Never import another `modules/*` package — shared logic goes in `infra/` or `utils/`.

### Config/env

`app/config/settings.py` (`pydantic-settings`) is the single source of truth for env vars — see it directly rather than `.env.example` for defaults and which fields are required vs optional (only `telegram_bot_token`, `postgres_user`, `postgres_password` have no default).
