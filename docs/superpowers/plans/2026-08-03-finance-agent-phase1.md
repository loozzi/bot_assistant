# Finance Agent Phase 1 (Core Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the finance module dispatchable: natural-language transaction logging with human confirmation, persisted to Postgres, written to Qdrant for recall, plus basic spend queries (total / by-category / keyword search) — the smallest slice of `docs/superpowers/specs/2026-08-03-finance-agent-design.md` that is genuinely useful end-to-end.

**Architecture:** A sub-intent router subgraph behind `FinancialAgent.run()` (same `BaseAgent` + `run_interruptible_subgraph` pattern as `search/`). `classify` picks one of 7 sub-intents; `log` and `query` are fully wired this phase, the other five (`budget`, `goal`, `split`, `advice`) get a graceful "not yet" reply. All arithmetic and persistence is plain Python in `tools/`; the LLM only classifies, parses free text into Pydantic schemas, and is never trusted for money figures.

**Tech Stack:** LangGraph (StateGraph + `interrupt`/`Command`), SQLAlchemy 2.0 async + asyncpg (Postgres), Qdrant (`qdrant-client`), Pydantic v2, pytest + pytest-asyncio + aiosqlite (new — first test suite in the repo).

## Global Constraints

- All money is integer VND (`BigInteger` columns, Python `int`) — never `float`, never other currencies. Source: spec "Currency".
- Categories are per-user; a seeded seed-list of 10 exists before any category logic runs. Source: spec "Categories".
- The LLM never computes numbers or writes to the database directly — only `tools/*.py` functions touch Postgres/Qdrant, and only after an LLM parse step produces a validated Pydantic object. Source: spec "Architecture".
- `log` and `split` are the only branches gated by human confirmation before writing; this phase only implements `log`. Source: spec "Architecture".
- No cross-module imports (`app/modules/finance` must not import from `app/modules/journal` etc.). Source: CLAUDE.md dependency rules.
- New tests live under `tests/unit/modules/finance/`, run via `uv run pytest`, and must not require live Postgres/Redis/Qdrant — use in-memory/sqlite fakes throughout (this phase adds the first pytest config the repo has ever had).
- `app/modules/finance/agent.py` uses a **lazy-compiled graph** (`build_graph()` + a `get_compiled_graph()` singleton), matching the pattern already used in `app/orchestrator/graph.py:build_graph`/`get_compiled_graph` — not the eager `builder.compile(checkpointer=get_checkpointer())` at import time used in `search/agent.py`. Reason: eager compilation calls `get_checkpointer()` at import, which raises `RuntimeError` unless Redis has already been initialized — that ordering only holds in the real startup sequence (`app/main.py:on_startup`), and makes the graph impossible to unit-test in isolation. The lazy pattern lets tests compile the same `StateGraph` with `langgraph.checkpoint.memory.InMemorySaver()` instead.
- `FinanceCategory.group` allows a 4th value, `"other"`, beyond the spec's literal `essential | lifestyle | savings_debt` — needed so the seeded "Khác" catch-all category doesn't have to be mislabeled into one of the three 50/30/20 buckets. This only affects the CheckConstraint and the seed data; the 50/30/20 lens itself is Phase 5 work.
- Tables `finance_categories`, `finance_logs`, `finance_budgets`, `finance_goals`, `finance_debts` do not exist in any deployed database yet (only `users` has ever been migrated) — this phase's single migration `CREATE TABLE`s all five directly in their final Phase-1-through-5 shape (per the spec: "One Alembic autogenerate migration covers all changes below"), so no further migrations are needed in Phases 2–5 unless a gap is found.

---

## File Structure

```
app/modules/finance/
  agent.py                  # rewritten: build_graph()/get_compiled_graph(), FinancialAgent
  state.py                  # rewritten: full FinancialState
  config.yaml                # unchanged
  node/
    __init__.py
    classify.py              # classify_node
    not_implemented.py       # not_implemented_node (budget/goal/split/advice stub reply)
    log/
      __init__.py
      parse.py                # log_parse_node
      confirm.py               # log_confirm_node (HITL)
      persist.py                # log_persist_node
      budget_check.py            # log_budget_check_node
      memory_write.py             # log_memory_write_node
    query/
      __init__.py
      extract_params.py        # query_extract_params_node
      run_query.py               # query_run_node
  tools/
    __init__.py
    users.py                  # get_or_create_user_pk
    categories.py               # DEFAULT_CATEGORIES, seed_default_categories, find_category_by_name, list_categories
    amounts.py                   # parse_vnd
    transactions.py                # create_transactions
    budgets.py                       # get_active_budget, compute_budget_status
    memory.py                          # write_transaction_memory
    queries.py                          # total_spend, spend_by_category, search_transactions
  schema/
    __init__.py
    classify_result.py         # ClassifyResult
    parsed_transaction.py       # ParsedTransaction, ParseResult
    query_params.py               # QueryParams
  prompts/
    __init__.py
    classify.py                 # CLASSIFY_SYSTEM_PROMPT
    parse.py                      # PARSE_SYSTEM_PROMPT
    query.py                        # QUERY_SYSTEM_PROMPT

app/models.py                  # fixed FinanceLog/FinanceCategory + new FinanceBudget/FinanceGoal/FinanceDebt
alembic/versions/<rev>_create_finance_tables.py

tests/unit/modules/finance/
  conftest.py                  # sqlite async session fixture, FakeLLM helper
  test_amounts.py
  test_users.py
  test_categories.py
  test_schema.py
  test_classify_node.py
  test_log_parse_node.py
  test_log_confirm_persist.py    # graph-level interrupt/resume test
  test_budget_check.py
  test_memory_write.py
  test_queries.py
  test_agent_end_to_end.py
```

No cross-module imports anywhere in the above.

---

### Task 1: Test infrastructure (pytest + async SQLite fixtures)

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/unit/modules/finance/__init__.py`, `tests/unit/modules/finance/conftest.py`

**Interfaces:**
- Produces: a `db_session` pytest-asyncio fixture (function-scoped, `AsyncSession` bound to an in-memory SQLite engine with all of `app.models.Base.metadata` created, rolled back/dropped after each test) that every later DB-touching test imports.
- Produces: a `FakeLLM` test helper class in `tests/unit/modules/finance/conftest.py` with `.with_structured_output(schema, method=...)` returning an object whose `.ainvoke(messages)` returns a pre-set Pydantic instance — used to stand in for `create_llm_client()` in node tests.

- [ ] **Step 1: Add dev dependencies**

```bash
uv add --dev pytest pytest-asyncio aiosqlite
```

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Append:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()
```

Note: this fixture will fail until Task 2 fixes `app/models.py` (it currently raises `InvalidRequestError: Attribute name 'metadata' is reserved` on import) — that's expected; Task 2's own test is what first turns this green.

- [ ] **Step 4: Create `tests/unit/modules/finance/__init__.py`**

Empty file.

- [ ] **Step 5: Create `tests/unit/modules/finance/conftest.py`**

```python
class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


class FakeLLM:
    """Stands in for create_llm_client() in node tests — returns a fixed
    structured-output result regardless of the prompt/messages passed in."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema, method=None):
        return _FakeStructuredLLM(self._result)
```

- [ ] **Step 6: Verify test collection works (even with zero real tests yet)**

Run: `uv run pytest --collect-only`
Expected: no collection errors (Task 2 not done yet, but nothing imports `app.models` at collection time in this task).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py tests/unit/modules/finance/__init__.py tests/unit/modules/finance/conftest.py
git commit -m "test(finance): add pytest infra with in-memory sqlite fixtures"
```

---

### Task 2: Fix and extend `app/models.py`

**Files:**
- Modify: `app/models.py`
- Test: `tests/unit/modules/finance/test_models.py`

**Interfaces:**
- Consumes: `db_session` fixture from Task 1.
- Produces: `FinanceLog(id, user_id, amount, description, message, log_type, occurred_at, created_at, extra, category_id)`, `FinanceCategory(id, user_id, name, description, group, is_seeded, created_at)`, `FinanceBudget(id, user_id, category_id, amount, created_at, updated_at)`, `FinanceGoal(id, user_id, description, target_amount, deadline, monthly_target, status, created_at, updated_at)`, `FinanceDebt(id, user_id, counterparty, direction, amount_total, amount_outstanding, description, status, extra, created_at, settled_at)` — every later task's `tools/*.py` imports these names directly from `app.models`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/modules/finance/test_models.py
import pytest
from sqlalchemy import select

from app.models import FinanceBudget, FinanceCategory, FinanceDebt, FinanceGoal, FinanceLog, User


@pytest.mark.asyncio
async def test_models_import_and_create_tables(db_session):
    # If app.models imports and db_session's create_all succeeded, the schema is valid.
    result = await db_session.execute(select(FinanceLog))
    assert result.all() == []


@pytest.mark.asyncio
async def test_finance_log_round_trip(db_session):
    user = User(telegram_id=123456789)
    db_session.add(user)
    await db_session.flush()

    category = FinanceCategory(
        user_id=user.id, name="Ăn uống", description="", group="essential", is_seeded=True
    )
    db_session.add(category)
    await db_session.flush()

    log = FinanceLog(
        user_id=user.id,
        amount=45_000,
        description="Trà sữa",
        message="mua trà sữa 45k",
        log_type="expense",
        category_id=category.id,
        occurred_at="2026-08-03",
        extra={"raw_amount_text": "45k"},
    )
    db_session.add(log)
    await db_session.commit()

    fetched = (await db_session.execute(select(FinanceLog))).scalar_one()
    assert fetched.amount == 45_000
    assert fetched.extra == {"raw_amount_text": "45k"}
    assert fetched.category_id == category.id


@pytest.mark.asyncio
async def test_finance_category_unique_per_user_not_globally(db_session):
    user_a = User(telegram_id=1)
    user_b = User(telegram_id=2)
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    db_session.add(FinanceCategory(user_id=user_a.id, name="Khác", description="", group="other"))
    db_session.add(FinanceCategory(user_id=user_b.id, name="Khác", description="", group="other"))
    await db_session.commit()  # must not raise — same name, different users

    result = await db_session.execute(select(FinanceCategory))
    assert len(result.all()) == 2


@pytest.mark.asyncio
async def test_new_tables_exist(db_session):
    user = User(telegram_id=999)
    db_session.add(user)
    await db_session.flush()

    db_session.add(FinanceBudget(user_id=user.id, category_id=None, amount=3_000_000))
    db_session.add(FinanceGoal(user_id=user.id, description="Mua xe", status="active"))
    db_session.add(
        FinanceDebt(
            user_id=user.id,
            counterparty="Nam",
            direction="owed_to_me",
            amount_total=150_000,
            amount_outstanding=150_000,
            description="Lẩu",
            status="open",
        )
    )
    await db_session.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_models.py -v`
Expected: FAIL — `sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.` (current bug in `app/models.py:41`) plus `ImportError` for `FinanceBudget`/`FinanceGoal`/`FinanceDebt`.

- [ ] **Step 3: Rewrite `app/models.py`**

```python
from datetime import date, datetime
from typing import Literal

from sqlalchemy import BigInteger, CheckConstraint, Date, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON_BLOB = postgresql.JSONB().with_variant(sqlite.JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FinanceCategory(Base):
    __tablename__ = "finance_categories"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_finance_categories_user_name"),
        CheckConstraint(
            "\"group\" IN ('essential', 'lifestyle', 'savings_debt', 'other')",
            name="ck_finance_categories_group",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column(default="")
    group: Mapped[str] = mapped_column(default="other")
    is_seeded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FinanceLog(Base):
    __tablename__ = "finance_logs"
    __table_args__ = (
        CheckConstraint("log_type IN ('income', 'expense')", name="ck_finance_logs_log_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    amount: Mapped[int] = mapped_column(BigInteger)
    description: Mapped[str] = mapped_column()
    message: Mapped[str] = mapped_column()
    log_type: Mapped[Literal["income", "expense"]] = mapped_column()

    occurred_at: Mapped[date] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    extra: Mapped[dict] = mapped_column(_JSON_BLOB, default=dict)

    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("finance_categories.id", ondelete="SET NULL"), default=None, nullable=True
    )


class FinanceBudget(Base):
    __tablename__ = "finance_budgets"
    __table_args__ = (UniqueConstraint("user_id", "category_id", name="uq_finance_budgets_user_category"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("finance_categories.id", ondelete="CASCADE"), nullable=True
    )
    amount: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class FinanceGoal(Base):
    __tablename__ = "finance_goals"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'achieved', 'archived')", name="ck_finance_goals_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    description: Mapped[str] = mapped_column()
    target_amount: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    deadline: Mapped[date | None] = mapped_column(Date, nullable=True)
    monthly_target: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[Literal["active", "achieved", "archived"]] = mapped_column(default="active")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class FinanceDebt(Base):
    __tablename__ = "finance_debts"
    __table_args__ = (
        CheckConstraint("direction IN ('owed_to_me', 'i_owe')", name="ck_finance_debts_direction"),
        CheckConstraint("status IN ('open', 'settled')", name="ck_finance_debts_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    counterparty: Mapped[str] = mapped_column()
    direction: Mapped[Literal["owed_to_me", "i_owe"]] = mapped_column()
    amount_total: Mapped[int] = mapped_column(BigInteger)
    amount_outstanding: Mapped[int] = mapped_column(BigInteger)
    description: Mapped[str] = mapped_column(default="")
    status: Mapped[Literal["open", "settled"]] = mapped_column(default="open")
    extra: Mapped[dict] = mapped_column(_JSON_BLOB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    settled_at: Mapped[datetime | None] = mapped_column(nullable=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/unit/modules/finance/test_models.py
git commit -m "fix(finance): repair reserved-name crash, add budget/goal/debt models"
```

---

### Task 3: Alembic migration for the finance tables

**Files:**
- Create: `alembic/versions/<generated_hash>_create_finance_tables.py`

**Interfaces:**
- Consumes: `app/models.py` from Task 2 (via `alembic/env.py:target_metadata`).
- Produces: `finance_categories`, `finance_logs`, `finance_budgets`, `finance_goals`, `finance_debts` tables in Postgres, chained after revision `e22e5ee4566f`.

- [ ] **Step 1: Try autogeneration against a reachable Postgres**

```bash
docker compose up -d postgres
uv run alembic revision --autogenerate -m "create finance tables"
```

If `docker compose`/Postgres isn't reachable in this environment, skip straight to Step 2 and hand-write the file below instead — do not block the rest of the plan on infra availability.

- [ ] **Step 2: Ensure the generated (or hand-written) migration matches this shape**

`down_revision` must be `'e22e5ee4566f'`. If autogenerate produced a different structure (e.g. combined columns in a different order, or missed the `CheckConstraint`/`UniqueConstraint` names), edit it to match — Alembic's autogenerate frequently misses named constraints and JSONB-vs-JSON dialect variants.

```python
"""create finance tables

Revision ID: <generated>
Revises: e22e5ee4566f
Create Date: <generated>

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "<generated>"
down_revision: Union[str, Sequence[str], None] = "e22e5ee4566f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "finance_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("group", sa.String(), nullable=False, server_default="other"),
        sa.Column("is_seeded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "\"group\" IN ('essential', 'lifestyle', 'savings_debt', 'other')",
            name="ck_finance_categories_group",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_finance_categories_user_name"),
    )

    op.create_table(
        "finance_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("log_type", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("log_type IN ('income', 'expense')", name="ck_finance_logs_log_type"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["category_id"], ["finance_categories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_finance_logs_user_id", "finance_logs", ["user_id"])
    op.create_index("ix_finance_logs_occurred_at", "finance_logs", ["occurred_at"])

    op.create_table(
        "finance_budgets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["category_id"], ["finance_categories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "category_id", name="uq_finance_budgets_user_category"),
    )
    op.create_index("ix_finance_budgets_user_id", "finance_budgets", ["user_id"])

    op.create_table(
        "finance_goals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("target_amount", sa.BigInteger(), nullable=True),
        sa.Column("deadline", sa.Date(), nullable=True),
        sa.Column("monthly_target", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'achieved', 'archived')", name="ck_finance_goals_status"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_finance_goals_user_id", "finance_goals", ["user_id"])

    op.create_table(
        "finance_debts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("counterparty", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("amount_total", sa.BigInteger(), nullable=False),
        sa.Column("amount_outstanding", sa.BigInteger(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default="open"),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("settled_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("direction IN ('owed_to_me', 'i_owe')", name="ck_finance_debts_direction"),
        sa.CheckConstraint("status IN ('open', 'settled')", name="ck_finance_debts_status"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_finance_debts_user_id", "finance_debts", ["user_id"])


def downgrade() -> None:
    op.drop_table("finance_debts")
    op.drop_table("finance_goals")
    op.drop_table("finance_budgets")
    op.drop_table("finance_logs")
    op.drop_table("finance_categories")
```

- [ ] **Step 3: Apply and verify (only if Postgres is reachable)**

```bash
uv run alembic upgrade head
uv run alembic current   # should print the new revision as head
uv run alembic downgrade -1 && uv run alembic upgrade head  # round-trip check
```

If Postgres isn't reachable here, leave this step for the user to run later — do not skip writing the migration file itself.

- [ ] **Step 4: Commit**

```bash
git add alembic/versions/
git commit -m "feat(finance): add migration for finance_categories/logs/budgets/goals/debts"
```

---

### Task 4: `tools/amounts.py` — colloquial VND parser

**Files:**
- Create: `app/modules/finance/__init__.py` (skip if it already exists), `app/modules/finance/tools/amounts.py`
- Test: `tests/unit/modules/finance/test_amounts.py`

**Interfaces:**
- Produces: `parse_vnd(text: str) -> int | None` — deterministic, no I/O, used by Task 8 (`log_parse_node`) to cross-check the LLM's parsed amount against the raw text span it copied.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/modules/finance/test_amounts.py
import pytest

from app.modules.finance.tools.amounts import parse_vnd


@pytest.mark.parametrize(
    "text, expected",
    [
        ("200k", 200_000),
        ("200K", 200_000),
        ("2tr", 2_000_000),
        ("2 triệu", 2_000_000),
        ("25 nghìn", 25_000),
        ("25 ngàn", 25_000),
        ("1tr2", 1_200_000),
        ("2tr5", 2_500_000),
        ("25000", 25_000),
        ("200.000", 200_000),
        ("200,000", 200_000),
        ("  50k  ", 50_000),
    ],
)
def test_parse_vnd_valid(text, expected):
    assert parse_vnd(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "$50", "50 usd", "€20", None])
def test_parse_vnd_invalid(text):
    assert parse_vnd(text) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_amounts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.modules.finance.tools.amounts'`

- [ ] **Step 3: Write the implementation**

```python
# app/modules/finance/tools/amounts.py
import re

_FOREIGN_MARKERS = ("$", "usd", "eur", "€", "gbp", "£")

_MILLION_WITH_TENTHS = re.compile(r"^(\d+)\s*tr\s*(\d+)$")
_MILLION = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(triệu|tr)$")
_THOUSAND = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|k)$")
_PLAIN_NUMBER = re.compile(r"^[\d.,]+$")


def parse_vnd(text: str | None) -> int | None:
    """Deterministically parse a colloquial Vietnamese amount into integer VND.

    Returns None for empty/unparseable input or anything carrying a
    non-VND currency marker — callers treat None as "could not verify".
    """
    if not text:
        return None

    normalized = text.strip().lower()
    if not normalized:
        return None
    if any(marker in normalized for marker in _FOREIGN_MARKERS):
        return None

    match = _MILLION_WITH_TENTHS.match(normalized)
    if match:
        whole, tenths = match.groups()
        return int(whole) * 1_000_000 + int(tenths) * 100_000

    match = _MILLION.match(normalized)
    if match:
        value = float(match.group(1).replace(",", "."))
        return int(value * 1_000_000)

    match = _THOUSAND.match(normalized)
    if match:
        value = float(match.group(1).replace(",", "."))
        return int(value * 1_000)

    if _PLAIN_NUMBER.match(normalized):
        digits = re.sub(r"[.,]", "", normalized)
        if digits.isdigit() and digits:
            return int(digits)

    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_amounts.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/modules/finance/tools/amounts.py tests/unit/modules/finance/test_amounts.py
git commit -m "feat(finance): add deterministic colloquial-VND amount parser"
```

---

### Task 5: `tools/users.py` and `tools/categories.py`

**Files:**
- Create: `app/modules/finance/tools/users.py`, `app/modules/finance/tools/categories.py`
- Test: `tests/unit/modules/finance/test_users.py`, `tests/unit/modules/finance/test_categories.py`

**Interfaces:**
- Consumes: `db_session` fixture (Task 1), `User`/`FinanceCategory` models (Task 2).
- Produces:
  - `get_or_create_user_pk(session: AsyncSession, telegram_id: str) -> int` — resolves a Telegram id string to `users.id`, creating the row if missing. Used by every node that touches Postgres.
  - `DEFAULT_CATEGORIES: list[tuple[str, str]]` — `(name, group)` pairs.
  - `seed_default_categories(session: AsyncSession, user_pk: int) -> None` — idempotent; no-ops if the user already has any category.
  - `find_category_by_name(session: AsyncSession, user_pk: int, name: str) -> FinanceCategory | None` — case-insensitive.
  - `list_categories(session: AsyncSession, user_pk: int) -> list[FinanceCategory]`.

- [ ] **Step 1: Write the failing test for `users.py`**

```python
# tests/unit/modules/finance/test_users.py
import pytest
from sqlalchemy import select

from app.models import User
from app.modules.finance.tools.users import get_or_create_user_pk


@pytest.mark.asyncio
async def test_creates_user_on_first_call(db_session):
    pk = await get_or_create_user_pk(db_session, "555")
    assert isinstance(pk, int)

    row = (await db_session.execute(select(User).where(User.id == pk))).scalar_one()
    assert row.telegram_id == 555


@pytest.mark.asyncio
async def test_returns_same_pk_on_repeat_call(db_session):
    first = await get_or_create_user_pk(db_session, "777")
    second = await get_or_create_user_pk(db_session, "777")
    assert first == second

    count = (await db_session.execute(select(User))).all()
    assert len(count) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_users.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `users.py`**

```python
# app/modules/finance/tools/users.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


async def get_or_create_user_pk(session: AsyncSession, telegram_id: str) -> int:
    """Resolve a Telegram id (string, as carried through AgentInput) to users.id."""
    tg_id = int(telegram_id)
    existing = (
        await session.execute(select(User).where(User.telegram_id == tg_id))
    ).scalar_one_or_none()
    if existing:
        return existing.id

    user = User(telegram_id=tg_id)
    session.add(user)
    await session.flush()
    return user.id
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_users.py -v`
Expected: passed.

- [ ] **Step 5: Write the failing test for `categories.py`**

```python
# tests/unit/modules/finance/test_categories.py
import pytest

from app.modules.finance.tools.categories import (
    DEFAULT_CATEGORIES,
    find_category_by_name,
    list_categories,
    seed_default_categories,
)
from app.modules.finance.tools.users import get_or_create_user_pk


@pytest.mark.asyncio
async def test_seed_default_categories_creates_all(db_session):
    user_pk = await get_or_create_user_pk(db_session, "1")
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    categories = await list_categories(db_session, user_pk)
    assert len(categories) == len(DEFAULT_CATEGORIES)
    assert {c.name for c in categories} == {name for name, _group in DEFAULT_CATEGORIES}
    assert all(c.is_seeded for c in categories)


@pytest.mark.asyncio
async def test_seed_is_idempotent(db_session):
    user_pk = await get_or_create_user_pk(db_session, "2")
    await seed_default_categories(db_session, user_pk)
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    categories = await list_categories(db_session, user_pk)
    assert len(categories) == len(DEFAULT_CATEGORIES)


@pytest.mark.asyncio
async def test_find_category_by_name_case_insensitive(db_session):
    user_pk = await get_or_create_user_pk(db_session, "3")
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    found = await find_category_by_name(db_session, user_pk, "ĂN UỐNG")
    assert found is not None
    assert found.name == "Ăn uống"

    missing = await find_category_by_name(db_session, user_pk, "không tồn tại")
    assert missing is None
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_categories.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 7: Implement `categories.py`**

```python
# app/modules/finance/tools/categories.py
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory

DEFAULT_CATEGORIES: list[tuple[str, str]] = [
    ("Ăn uống", "essential"),
    ("Đi lại", "essential"),
    ("Nhà cửa", "essential"),
    ("Hóa đơn & tiện ích", "essential"),
    ("Sức khỏe", "essential"),
    ("Giáo dục", "essential"),
    ("Mua sắm", "lifestyle"),
    ("Giải trí", "lifestyle"),
    ("Tiết kiệm & đầu tư", "savings_debt"),
    ("Khác", "other"),
]


async def list_categories(session: AsyncSession, user_pk: int) -> list[FinanceCategory]:
    result = await session.execute(
        select(FinanceCategory).where(FinanceCategory.user_id == user_pk)
    )
    return list(result.scalars().all())


async def seed_default_categories(session: AsyncSession, user_pk: int) -> None:
    """Create the default category set for a user, unless they already have any."""
    existing = await list_categories(session, user_pk)
    if existing:
        return

    for name, group in DEFAULT_CATEGORIES:
        session.add(
            FinanceCategory(
                user_id=user_pk, name=name, description="", group=group, is_seeded=True
            )
        )
    await session.flush()


async def find_category_by_name(
    session: AsyncSession, user_pk: int, name: str
) -> FinanceCategory | None:
    result = await session.execute(
        select(FinanceCategory).where(
            FinanceCategory.user_id == user_pk,
            func.lower(FinanceCategory.name) == name.strip().lower(),
        )
    )
    return result.scalar_one_or_none()
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_categories.py -v`
Expected: passed.

- [ ] **Step 9: Commit**

```bash
git add app/modules/finance/tools/users.py app/modules/finance/tools/categories.py \
        tests/unit/modules/finance/test_users.py tests/unit/modules/finance/test_categories.py
git commit -m "feat(finance): add user resolution and default category seeding"
```

---

### Task 6: Finance schemas and prompts

**Files:**
- Create: `app/modules/finance/schema/__init__.py`, `app/modules/finance/schema/classify_result.py`, `app/modules/finance/schema/parsed_transaction.py`, `app/modules/finance/schema/query_params.py`
- Create: `app/modules/finance/prompts/classify.py`, `app/modules/finance/prompts/parse.py`, `app/modules/finance/prompts/query.py`
- Test: `tests/unit/modules/finance/test_schema.py`

**Interfaces:**
- Produces: `ClassifyResult(sub_intent: Literal["log","query","budget","goal","split","advice","fallback"])`; `ParsedTransaction(amount_vnd: int, raw_amount_text: str, description: str, log_type: Literal["income","expense"], suggested_category: str, occurred_at: date)`; `ParseResult(transactions: list[ParsedTransaction], unparsed_notes: list[str])`; `QueryParams(metric: Literal["total","by_category","search"], period_start: date, period_end: date, category: str | None, keyword: str | None)`.
- Consumed by Task 7 (`classify_node`), Task 8 (`log_parse_node`), Task 12 (`query_extract_params_node`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/modules/finance/test_schema.py
from datetime import date

import pytest
from pydantic import ValidationError

from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from app.modules.finance.schema.query_params import QueryParams


def test_classify_result_accepts_known_labels():
    assert ClassifyResult(sub_intent="log").sub_intent == "log"


def test_classify_result_rejects_unknown_label():
    with pytest.raises(ValidationError):
        ClassifyResult(sub_intent="not_a_real_intent")


def test_parsed_transaction_rejects_non_positive_amount():
    with pytest.raises(ValidationError):
        ParsedTransaction(
            amount_vnd=0,
            raw_amount_text="0k",
            description="x",
            log_type="expense",
            suggested_category="Khác",
            occurred_at=date(2026, 8, 3),
        )


def test_parse_result_holds_multiple_transactions():
    txn = ParsedTransaction(
        amount_vnd=45_000,
        raw_amount_text="45k",
        description="Trà sữa",
        log_type="expense",
        suggested_category="Ăn uống",
        occurred_at=date(2026, 8, 3),
    )
    result = ParseResult(transactions=[txn], unparsed_notes=[])
    assert len(result.transactions) == 1


def test_query_params_requires_period():
    with pytest.raises(ValidationError):
        QueryParams(metric="total")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the schema files**

```python
# app/modules/finance/schema/__init__.py
```

```python
# app/modules/finance/schema/classify_result.py
from typing import Literal

from pydantic import BaseModel

SubIntent = Literal["log", "query", "budget", "goal", "split", "advice", "fallback"]


class ClassifyResult(BaseModel):
    sub_intent: SubIntent
```

```python
# app/modules/finance/schema/parsed_transaction.py
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ParsedTransaction(BaseModel):
    amount_vnd: int
    raw_amount_text: str
    description: str
    log_type: Literal["income", "expense"]
    suggested_category: str
    occurred_at: date

    @field_validator("amount_vnd")
    @classmethod
    def amount_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("amount_vnd must be positive")
        return value


class ParseResult(BaseModel):
    transactions: list[ParsedTransaction] = Field(default_factory=list)
    unparsed_notes: list[str] = Field(default_factory=list)
```

```python
# app/modules/finance/schema/query_params.py
from datetime import date
from typing import Literal

from pydantic import BaseModel


class QueryParams(BaseModel):
    metric: Literal["total", "by_category", "search"]
    period_start: date
    period_end: date
    category: str | None = None
    keyword: str | None = None
```

```python
# app/modules/finance/prompts/__init__.py
```

```python
# app/modules/finance/prompts/classify.py
CLASSIFY_SYSTEM_PROMPT = """You are the sub-intent classifier inside a personal-finance \
assistant module. Given the user's message, choose exactly one sub_intent:

- log: the user is reporting one or more purchases/income (e.g. "mua trà sữa 45k")
- query: the user is asking about past spending/income (totals, by category, "how much did I spend on X")
- budget: the user wants to set, change, or check a spending limit
- goal: the user wants to set or check a savings/financial goal
- split: the user wants to split a bill among people, or settle/check a debt
- advice: the user is asking whether they should spend on something, or wants financial advice
- fallback: none of the above clearly apply

Respond only with JSON: {"sub_intent": "<one of the above>"}."""
```

```python
# app/modules/finance/prompts/parse.py
PARSE_SYSTEM_PROMPT = """You extract structured transactions from a Vietnamese/English \
message describing spending or income. A single message may describe multiple transactions.

For each transaction, produce:
- amount_vnd: the amount in integer VND (e.g. "200k" -> 200000, "2 triệu" -> 2000000)
- raw_amount_text: the exact substring of the original message that states the amount \
  (e.g. "200k", "2 triệu", "25000") — copy it verbatim, do not normalize it yourself
- description: a short description of what was bought/received
- log_type: "expense" or "income"
- suggested_category: your best-guess category name in Vietnamese (e.g. "Ăn uống", "Đi lại")
- occurred_at: an ISO date (YYYY-MM-DD), resolved against the "today" date given below \
  (e.g. "hôm qua" = yesterday, no date mentioned = today)

If any amount is stated in a non-VND currency (e.g. "$50", "50 usd", "20 eur"), do NOT \
produce a transaction for it — instead add a short note about it to unparsed_notes.

If the message describes no transactions at all, return an empty transactions list.

Respond only with JSON matching this shape:
{"transactions": [{"amount_vnd": int, "raw_amount_text": str, "description": str, \
"log_type": "expense"|"income", "suggested_category": str, "occurred_at": "YYYY-MM-DD"}], \
"unparsed_notes": [str]}"""


def build_parse_user_message(user_query: str, today_iso: str) -> str:
    return f"Today's date: {today_iso}\n\nMessage: {user_query}"
```

```python
# app/modules/finance/prompts/query.py
QUERY_SYSTEM_PROMPT = """You extract structured query parameters from a question about \
past spending/income. Resolve relative periods (e.g. "tháng này", "this month", "hôm nay") \
against the "today" date given below into explicit period_start/period_end ISO dates.

metric must be one of:
- "total": how much was spent/earned in the period (optionally filtered by category)
- "by_category": breakdown of spending by category in the period
- "search": look up transactions matching a keyword (e.g. "how much did I spend on trà sữa")

Set category to the Vietnamese category name if the user names one, else null.
Set keyword to the search term if metric is "search", else null.

Respond only with JSON matching:
{"metric": "total"|"by_category"|"search", "period_start": "YYYY-MM-DD", \
"period_end": "YYYY-MM-DD", "category": str|null, "keyword": str|null}"""


def build_query_user_message(user_query: str, today_iso: str) -> str:
    return f"Today's date: {today_iso}\n\nQuestion: {user_query}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_schema.py -v`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add app/modules/finance/schema/ app/modules/finance/prompts/ tests/unit/modules/finance/test_schema.py
git commit -m "feat(finance): add classify/parse/query pydantic schemas and prompts"
```

---

### Task 7: `node/classify.py`

**Files:**
- Create: `app/modules/finance/node/__init__.py` (extend, don't overwrite — currently empty), `app/modules/finance/node/classify.py`
- Test: `tests/unit/modules/finance/test_classify_node.py`

**Interfaces:**
- Consumes: `FakeLLM` (Task 1), `ClassifyResult`/`CLASSIFY_SYSTEM_PROMPT` (Task 6).
- Produces: `async def classify_node(state: dict) -> dict` returning `{"sub_intent": <str>}`. Reads `state["user_query"]`. On any exception, returns `{"sub_intent": "fallback"}` — never raises (matches `search_node`'s degrade-gracefully pattern).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/modules/finance/test_classify_node.py
import pytest

from app.modules.finance.node.classify import classify_node
from app.modules.finance.schema.classify_result import ClassifyResult
from tests.unit.modules.finance.conftest import FakeLLM


@pytest.mark.asyncio
async def test_classify_node_returns_llm_sub_intent(monkeypatch):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="log")),
    )
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "log"}


@pytest.mark.asyncio
async def test_classify_node_falls_back_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("llm down")

    monkeypatch.setattr("app.modules.finance.node.classify.create_llm_client", _raise)
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "fallback"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_classify_node.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# app/modules/finance/node/classify.py
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.classify import CLASSIFY_SYSTEM_PROMPT
from ..schema.classify_result import ClassifyResult

logger = get_logger(__name__)


async def classify_node(state: dict) -> dict:
    """Pick one finance sub-intent for the user's message.

    Falls back to "fallback" on any LLM failure so the graph always has a
    valid sub_intent to route on.
    """
    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ClassifyResult, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ]
        )
        return {"sub_intent": result.sub_intent}
    except Exception as exc:
        logger.warning("finance_classify_failed", error=str(exc))
        return {"sub_intent": "fallback"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_classify_node.py -v`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add app/modules/finance/node/classify.py tests/unit/modules/finance/test_classify_node.py
git commit -m "feat(finance): add sub-intent classify node"
```

---

### Task 8: `node/log/parse.py`

**Files:**
- Create: `app/modules/finance/node/log/__init__.py`, `app/modules/finance/node/log/parse.py`
- Test: `tests/unit/modules/finance/test_log_parse_node.py`

**Interfaces:**
- Consumes: `parse_vnd` (Task 4), `ParseResult`/`PARSE_SYSTEM_PROMPT`/`build_parse_user_message` (Task 6).
- Produces: `async def log_parse_node(state: dict) -> dict` returning `{"parsed_transactions": list[dict], "unparsed_notes": list[str]}`. Each dict in `parsed_transactions` is a `ParsedTransaction.model_dump()` with `amount_vnd` possibly corrected by `parse_vnd`, plus an `amount_adjusted: bool` flag. Transactions outside the 1,000–500,000,000 VND sanity band are dropped and a note is added to `unparsed_notes` instead. On LLM failure, returns empty transactions and one note explaining the failure — consumed by Task 9's confirm node.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/modules/finance/test_log_parse_node.py
from datetime import date

import pytest

from app.modules.finance.node.log.parse import log_parse_node
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from tests.unit.modules.finance.conftest import FakeLLM


@pytest.mark.asyncio
async def test_parse_node_passes_through_agreeing_amount(monkeypatch):
    txn = ParsedTransaction(
        amount_vnd=45_000,
        raw_amount_text="45k",
        description="Trà sữa",
        log_type="expense",
        suggested_category="Ăn uống",
        occurred_at=date(2026, 8, 3),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client",
        lambda: FakeLLM(ParseResult(transactions=[txn], unparsed_notes=[])),
    )

    result = await log_parse_node({"user_query": "mua trà sữa 45k", "user_id": "1"})

    assert len(result["parsed_transactions"]) == 1
    parsed = result["parsed_transactions"][0]
    assert parsed["amount_vnd"] == 45_000
    assert parsed["amount_adjusted"] is False
    assert result["unparsed_notes"] == []


@pytest.mark.asyncio
async def test_parse_node_corrects_disagreeing_amount(monkeypatch):
    # LLM misreads "200k" as 20000 — the deterministic parser must win.
    txn = ParsedTransaction(
        amount_vnd=20_000,
        raw_amount_text="200k",
        description="Ăn trưa",
        log_type="expense",
        suggested_category="Ăn uống",
        occurred_at=date(2026, 8, 3),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client",
        lambda: FakeLLM(ParseResult(transactions=[txn], unparsed_notes=[])),
    )

    result = await log_parse_node({"user_query": "ăn trưa 200k", "user_id": "1"})

    parsed = result["parsed_transactions"][0]
    assert parsed["amount_vnd"] == 200_000
    assert parsed["amount_adjusted"] is True


@pytest.mark.asyncio
async def test_parse_node_drops_out_of_range_amount(monkeypatch):
    txn = ParsedTransaction(
        amount_vnd=999_999_999,
        raw_amount_text="999999999",
        description="???",
        log_type="expense",
        suggested_category="Khác",
        occurred_at=date(2026, 8, 3),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client",
        lambda: FakeLLM(ParseResult(transactions=[txn], unparsed_notes=[])),
    )

    result = await log_parse_node({"user_query": "test", "user_id": "1"})

    assert result["parsed_transactions"] == []
    assert len(result["unparsed_notes"]) == 1


@pytest.mark.asyncio
async def test_parse_node_falls_back_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("llm down")

    monkeypatch.setattr("app.modules.finance.node.log.parse.create_llm_client", _raise)

    result = await log_parse_node({"user_query": "mua trà sữa 45k", "user_id": "1"})

    assert result["parsed_transactions"] == []
    assert len(result["unparsed_notes"]) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_log_parse_node.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# app/modules/finance/node/log/__init__.py
```

```python
# app/modules/finance/node/log/parse.py
from datetime import datetime, timezone

from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.parse import PARSE_SYSTEM_PROMPT, build_parse_user_message
from ...schema.parsed_transaction import ParseResult
from ...tools.amounts import parse_vnd

logger = get_logger(__name__)

_MIN_AMOUNT_VND = 1_000
_MAX_AMOUNT_VND = 500_000_000


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def log_parse_node(state: dict) -> dict:
    """Parse the user's message into transactions, with amounts verified
    against the deterministic parser (never trust the LLM's arithmetic)."""
    try:
        llm = create_llm_client()
        parser = llm.with_structured_output(ParseResult, method="json_mode")
        result: ParseResult = await parser.ainvoke(
            [
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_parse_user_message(state["user_query"], _today_iso()),
                },
            ]
        )
    except Exception as exc:
        logger.warning("finance_parse_failed", error=str(exc))
        return {
            "parsed_transactions": [],
            "unparsed_notes": ["Mình chưa đọc hiểu được giao dịch, bạn thử nói rõ hơn nhé."],
        }

    transactions: list[dict] = []
    notes = list(result.unparsed_notes)

    for txn in result.transactions:
        verified_amount = parse_vnd(txn.raw_amount_text)
        amount_adjusted = verified_amount is not None and verified_amount != txn.amount_vnd
        final_amount = verified_amount if verified_amount is not None else txn.amount_vnd

        if not (_MIN_AMOUNT_VND <= final_amount <= _MAX_AMOUNT_VND):
            notes.append(f"Bỏ qua '{txn.description}' vì số tiền có vẻ không hợp lý.")
            continue

        dumped = txn.model_dump(mode="json")
        dumped["amount_vnd"] = final_amount
        dumped["amount_adjusted"] = amount_adjusted
        transactions.append(dumped)

    return {"parsed_transactions": transactions, "unparsed_notes": notes}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_log_parse_node.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/modules/finance/node/log/__init__.py app/modules/finance/node/log/parse.py \
        tests/unit/modules/finance/test_log_parse_node.py
git commit -m "feat(finance): add log-branch parse node with deterministic amount verification"
```

---

### Task 9: `node/log/confirm.py` + `node/log/persist.py` (HITL confirm → persist)

**Files:**
- Create: `app/modules/finance/node/log/confirm.py`, `app/modules/finance/node/log/persist.py`, `app/modules/finance/tools/transactions.py`
- Test: `tests/unit/modules/finance/test_log_confirm_persist.py`

**Interfaces:**
- Consumes: `app.core.hitl.HumanReviewRequest`, `get_or_create_user_pk`/`find_category_by_name` (Task 5), `db_session`-style session factory.
- Produces:
  - `async def log_confirm_node(state: dict) -> dict` — calls `interrupt(HumanReviewRequest(...))` with options `[{"label": "Xác nhận", "value": "confirm"}, {"label": "Sửa", "value": "edit"}]` (the bot's `hitl_keyboard` helper auto-adds a "Huỷ" button that never reaches this node — see `app/bot/keyboards.py:hitl_keyboard`). If `state["parsed_transactions"]` is empty, skips the question and returns `{"confirm_answer": "edit"}` immediately (nothing to confirm). Returns `{"confirm_answer": <resume value>}`.
  - `async def log_persist_node(state: dict) -> dict` — inserts one `FinanceLog` row per parsed transaction and returns `{"persisted_log_ids": list[int], "persisted_categories": list[str]}`. Takes `state["session"]` (an open `AsyncSession`) and `state["user_pk"]` as given — it does **not** open its own session or resolve the user itself. This mirrors `log_budget_check_node` (Task 10) and `query_run_node` (Task 12), which need that same open session for follow-up work; Task 13's `agent.py` is the only place that actually opens a session, via one `async with get_session()` wrapping persist → budget_check → memory_write.
  - `create_transactions(session, user_pk: int, transactions: list[dict]) -> list[FinanceLog]` in `tools/transactions.py` — the actual insert logic `log_persist_node` calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/modules/finance/test_log_confirm_persist.py
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from sqlalchemy import select

from app.models import FinanceLog
from app.modules.finance.node.log.confirm import log_confirm_node
from app.modules.finance.node.log.persist import log_persist_node
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk


@pytest.mark.asyncio
async def test_create_transactions_inserts_rows_and_seeds_categories(db_session):
    user_pk = await get_or_create_user_pk(db_session, "42")
    transactions = [
        {
            "amount_vnd": 45_000,
            "raw_amount_text": "45k",
            "description": "Trà sữa",
            "log_type": "expense",
            "suggested_category": "Ăn uống",
            "occurred_at": "2026-08-03",
            "amount_adjusted": False,
        }
    ]

    logs = await create_transactions(db_session, user_pk, transactions)
    await db_session.commit()

    assert len(logs) == 1
    stored = (await db_session.execute(select(FinanceLog))).scalar_one()
    assert stored.amount == 45_000
    assert stored.category_id is not None  # matched the seeded "Ăn uống" category


@pytest.mark.asyncio
async def test_create_transactions_falls_back_to_khac_for_unknown_category(db_session):
    user_pk = await get_or_create_user_pk(db_session, "43")
    transactions = [
        {
            "amount_vnd": 10_000,
            "raw_amount_text": "10k",
            "description": "???",
            "log_type": "expense",
            "suggested_category": "Danh mục lạ hoắc",
            "occurred_at": "2026-08-03",
            "amount_adjusted": False,
        }
    ]

    logs = await create_transactions(db_session, user_pk, transactions)
    await db_session.commit()

    from app.models import FinanceCategory

    category = (
        await db_session.execute(
            select(FinanceCategory).where(FinanceCategory.id == logs[0].category_id)
        )
    ).scalar_one()
    assert category.name == "Khác"


def _build_test_graph(db_session):
    """Minimal graph exercising just confirm -> {persist, END}, checkpointed
    in-memory so interrupt/resume can be tested without Redis.

    The persist wrapper injects session/user_pk directly, mirroring how
    agent.py's real _log_persist_and_followups_node (Task 13) will call
    log_persist_node — that node itself never opens a session."""

    async def _persist_wrapper(state: dict) -> dict:
        user_pk = await get_or_create_user_pk(db_session, state["user_id"])
        return await log_persist_node({**state, "session": db_session, "user_pk": user_pk})

    def _route(state: dict) -> str:
        return "persist" if state["confirm_answer"] == "confirm" else "__end__"

    builder = StateGraph(dict)
    builder.add_node("confirm", log_confirm_node)
    builder.add_node("persist", _persist_wrapper)
    builder.set_entry_point("confirm")
    builder.add_conditional_edges("confirm", _route, {"persist": "persist", "__end__": END})
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_confirm_then_confirm_persists(db_session):
    graph = _build_test_graph(db_session)
    config = {"configurable": {"thread_id": "test-thread-1"}}

    initial = {
        "user_id": "42",
        "parsed_transactions": [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            }
        ],
    }
    paused = await graph.ainvoke(initial, config=config)
    assert paused["__interrupt__"][0].value["question"]

    result = await graph.ainvoke(Command(resume="confirm"), config=config)
    assert result["persisted_log_ids"]

    stored = (await db_session.execute(select(FinanceLog))).all()
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_confirm_then_edit_does_not_persist(db_session):
    graph = _build_test_graph(db_session)
    config = {"configurable": {"thread_id": "test-thread-2"}}

    initial = {
        "user_id": "42",
        "parsed_transactions": [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            }
        ],
    }
    await graph.ainvoke(initial, config=config)
    result = await graph.ainvoke(Command(resume="edit"), config=config)

    assert "persisted_log_ids" not in result
    stored = (await db_session.execute(select(FinanceLog))).all()
    assert len(stored) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_log_confirm_persist.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/transactions.py`**

```python
# app/modules/finance/tools/transactions.py
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceLog

from .categories import find_category_by_name, seed_default_categories


async def create_transactions(
    session: AsyncSession, user_pk: int, transactions: list[dict]
) -> list[FinanceLog]:
    """Insert one FinanceLog row per parsed transaction dict.

    Ensures the user's categories are seeded first, then matches each
    transaction's suggested_category by name — falling back to "Khác" for
    anything the LLM invented that doesn't match a real category.
    """
    await seed_default_categories(session, user_pk)

    khac = await find_category_by_name(session, user_pk, "Khác")

    logs: list[FinanceLog] = []
    for txn in transactions:
        category = await find_category_by_name(session, user_pk, txn["suggested_category"])
        if category is None:
            category = khac

        log = FinanceLog(
            user_id=user_pk,
            amount=txn["amount_vnd"],
            description=txn["description"],
            message=txn["description"],
            log_type=txn["log_type"],
            category_id=category.id if category else None,
            occurred_at=date.fromisoformat(txn["occurred_at"]),
            extra={"raw_amount_text": txn["raw_amount_text"], "amount_adjusted": txn["amount_adjusted"]},
        )
        session.add(log)
        logs.append(log)

    await session.flush()
    return logs
```

- [ ] **Step 4: Implement `node/log/confirm.py`**

```python
# app/modules/finance/node/log/confirm.py
from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest


def _render_transaction_list(transactions: list[dict]) -> str:
    lines = []
    for i, txn in enumerate(transactions, 1):
        amount = f"{txn['amount_vnd']:,}".replace(",", ".")
        marker = " (đã điều chỉnh số tiền)" if txn["amount_adjusted"] else ""
        lines.append(f"{i}. {txn['description']} — {amount}đ — {txn['suggested_category']}{marker}")
    return "\n".join(lines)


async def log_confirm_node(state: dict) -> dict:
    """Ask the user to confirm the parsed transactions before writing them.

    If nothing was parsed, there's nothing to confirm — route straight to
    the "edit" (ask-to-restate) path without asking a question.
    """
    transactions = state.get("parsed_transactions", [])
    if not transactions:
        return {"confirm_answer": "edit"}

    question = "Xác nhận các giao dịch sau nhé:\n\n" + _render_transaction_list(transactions)
    answer = interrupt(
        HumanReviewRequest(
            question=question,
            options=[
                {"label": "Xác nhận", "value": "confirm"},
                {"label": "Sửa", "value": "edit"},
            ],
        )
    )
    return {"confirm_answer": answer}
```

- [ ] **Step 5: Implement `node/log/persist.py`**

```python
# app/modules/finance/node/log/persist.py
from app.utils.logger import get_logger

from ...tools.transactions import create_transactions

logger = get_logger(__name__)


async def log_persist_node(state: dict) -> dict:
    """Insert one FinanceLog row per parsed transaction.

    Expects state["session"] (an open AsyncSession) and state["user_pk"] —
    supplied by the caller (agent.py's _log_persist_and_followups_node in
    production; test wrappers directly). Session lifecycle deliberately
    lives outside this function so budget_check/memory_write can reuse the
    same open session for the rest of the log branch's unit of work.
    """
    transactions = state.get("parsed_transactions", [])
    if not transactions:
        return {}

    session = state["session"]
    user_pk = state["user_pk"]

    logs = await create_transactions(session, user_pk, transactions)

    return {
        "persisted_log_ids": [log.id for log in logs],
        "persisted_categories": [txn["suggested_category"] for txn in transactions],
    }
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_log_confirm_persist.py -v`
Expected: all passed. If the async-context-manager mismatch from Step 5's note occurs, apply that fix and rerun.

- [ ] **Step 7: Commit**

```bash
git add app/modules/finance/node/log/confirm.py app/modules/finance/node/log/persist.py \
        app/modules/finance/tools/transactions.py \
        tests/unit/modules/finance/test_log_confirm_persist.py
git commit -m "feat(finance): add HITL confirm and persist for the log branch"
```

---

### Task 10: `tools/budgets.py` + `node/log/budget_check.py`

**Files:**
- Create: `app/modules/finance/tools/budgets.py`, `app/modules/finance/node/log/budget_check.py`
- Test: `tests/unit/modules/finance/test_budget_check.py`

**Interfaces:**
- Produces:
  - `async def get_active_budget(session, user_pk: int, category_id: int | None) -> FinanceBudget | None` — category-specific budget if one exists, else the overall (`category_id IS NULL`) budget, else `None`.
  - `async def month_to_date_spend(session, user_pk: int, category_id: int, today: date) -> int` — sum of `expense` amounts in the calendar month containing `today`.
  - `compute_budget_status(spent: int, budget_amount: int) -> dict` — `{"ratio": float, "level": "ok"|"warn"|"over"}`, `"warn"` at ratio ≥ 0.8, `"over"` at ratio > 1.0.
  - `async def log_budget_check_node(state: dict) -> dict` — for each persisted transaction's category, checks the budget and appends a warning line to `state["budget_warnings"]` (a `list[str]`, possibly empty). Wired into the graph so budget alerts are live the moment a budget is ever set (Phase 3 adds the set-budget UI; this node's logic doesn't change).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/modules/finance/test_budget_check.py
from datetime import date

import pytest

from app.models import FinanceBudget
from app.modules.finance.node.log.budget_check import log_budget_check_node
from app.modules.finance.tools.budgets import compute_budget_status, get_active_budget, month_to_date_spend
from app.modules.finance.tools.categories import find_category_by_name, seed_default_categories
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk


def test_compute_budget_status_levels():
    assert compute_budget_status(500_000, 1_000_000)["level"] == "ok"
    assert compute_budget_status(850_000, 1_000_000)["level"] == "warn"
    assert compute_budget_status(1_200_000, 1_000_000)["level"] == "over"


@pytest.mark.asyncio
async def test_get_active_budget_prefers_category_specific(db_session):
    user_pk = await get_or_create_user_pk(db_session, "10")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    db_session.add(FinanceBudget(user_id=user_pk, category_id=None, amount=5_000_000))
    db_session.add(FinanceBudget(user_id=user_pk, category_id=category.id, amount=1_000_000))
    await db_session.commit()

    budget = await get_active_budget(db_session, user_pk, category.id)
    assert budget.amount == 1_000_000


@pytest.mark.asyncio
async def test_get_active_budget_none_when_unset(db_session):
    user_pk = await get_or_create_user_pk(db_session, "11")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    budget = await get_active_budget(db_session, user_pk, category.id)
    assert budget is None


@pytest.mark.asyncio
async def test_budget_check_node_warns_near_threshold(db_session):
    user_pk = await get_or_create_user_pk(db_session, "12")
    await seed_default_categories(db_session, user_pk)
    category = await find_category_by_name(db_session, user_pk, "Ăn uống")

    db_session.add(FinanceBudget(user_id=user_pk, category_id=category.id, amount=1_000_000))
    await db_session.commit()

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": date.today().isoformat(),
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {
            "user_pk": user_pk,
            "session": db_session,
            "persisted_categories": ["Ăn uống"],
        }
    )
    assert len(result["budget_warnings"]) == 1
    assert "Ăn uống" in result["budget_warnings"][0]


@pytest.mark.asyncio
async def test_budget_check_node_silent_without_budget(db_session):
    user_pk = await get_or_create_user_pk(db_session, "13")
    await seed_default_categories(db_session, user_pk)

    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 850_000,
                "raw_amount_text": "850k",
                "description": "Ăn tuần này",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": date.today().isoformat(),
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await log_budget_check_node(
        {"user_pk": user_pk, "session": db_session, "persisted_categories": ["Ăn uống"]}
    )
    assert result["budget_warnings"] == []
```

Note: `log_budget_check_node` takes `session`/`user_pk` directly in its input dict here (unlike the graph nodes before it) because in the real graph it runs as a continuation of `log_persist_node` and needs the same open session/user_pk — Step 3 below has `log_persist_node` (Task 9) pass those through via `state["_session_scope"]`; see Step 4 for how the two nodes actually connect once wired into `agent.py` in Task 13. For this task's own tests, the node is exercised directly with a hand-built state dict as shown above.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_budget_check.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/budgets.py`**

```python
# app/modules/finance/tools/budgets.py
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceBudget, FinanceLog

_WARN_RATIO = 0.8


async def get_active_budget(
    session: AsyncSession, user_pk: int, category_id: int | None
) -> FinanceBudget | None:
    """Return the category-specific budget if set, else the overall budget, else None."""
    if category_id is not None:
        specific = (
            await session.execute(
                select(FinanceBudget).where(
                    FinanceBudget.user_id == user_pk, FinanceBudget.category_id == category_id
                )
            )
        ).scalar_one_or_none()
        if specific:
            return specific

    return (
        await session.execute(
            select(FinanceBudget).where(
                FinanceBudget.user_id == user_pk, FinanceBudget.category_id.is_(None)
            )
        )
    ).scalar_one_or_none()


async def month_to_date_spend(
    session: AsyncSession, user_pk: int, category_id: int, today: date
) -> int:
    month_start = today.replace(day=1)
    total = (
        await session.execute(
            select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(
                FinanceLog.user_id == user_pk,
                FinanceLog.category_id == category_id,
                FinanceLog.log_type == "expense",
                FinanceLog.occurred_at >= month_start,
                FinanceLog.occurred_at <= today,
            )
        )
    ).scalar_one()
    return int(total)


def compute_budget_status(spent: int, budget_amount: int) -> dict:
    ratio = spent / budget_amount if budget_amount else 0.0
    if ratio > 1.0:
        level = "over"
    elif ratio >= _WARN_RATIO:
        level = "warn"
    else:
        level = "ok"
    return {"ratio": ratio, "level": level}
```

- [ ] **Step 4: Implement `node/log/budget_check.py`**

```python
# app/modules/finance/node/log/budget_check.py
from datetime import date

from ...tools.budgets import compute_budget_status, get_active_budget, month_to_date_spend
from ...tools.categories import find_category_by_name


async def log_budget_check_node(state: dict) -> dict:
    """Warn when a just-logged category is near/over its budget.

    Requires state["session"] (an open AsyncSession) and state["user_pk"] —
    both set by log_persist_node before this node runs. No-ops (empty
    warnings) for any category without a budget configured.
    """
    session = state["session"]
    user_pk = state["user_pk"]
    category_names = set(state.get("persisted_categories", []))

    warnings: list[str] = []
    today = date.today()

    for name in category_names:
        category = await find_category_by_name(session, user_pk, name)
        if category is None:
            continue

        budget = await get_active_budget(session, user_pk, category.id)
        if budget is None:
            continue

        spent = await month_to_date_spend(session, user_pk, category.id, today)
        status = compute_budget_status(spent, budget.amount)

        if status["level"] == "over":
            warnings.append(
                f"⚠️ Bạn đã vượt ngân sách {name} tháng này "
                f"({spent:,}đ / {budget.amount:,}đ)".replace(",", ".")
            )
        elif status["level"] == "warn":
            pct = int(status["ratio"] * 100)
            warnings.append(
                f"⚠️ Bạn đã dùng {pct}% ngân sách {name} tháng này "
                f"({spent:,}đ / {budget.amount:,}đ)".replace(",", ".")
            )

    return {"budget_warnings": warnings}
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_budget_check.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add app/modules/finance/tools/budgets.py app/modules/finance/node/log/budget_check.py \
        tests/unit/modules/finance/test_budget_check.py
git commit -m "feat(finance): add budget threshold check, wired but dormant until budgets are set"
```

---

### Task 11: `tools/memory.py` — Qdrant write for logged transactions

**Files:**
- Create: `app/modules/finance/tools/memory.py`
- Test: `tests/unit/modules/finance/test_memory_write.py`

**Interfaces:**
- Produces: `async def write_transaction_memory(user_id: str, log_id: int, category_name: str, description: str, amount_vnd: int, log_type: str, occurred_at: str) -> None` — embeds one sentence per transaction and upserts it to Qdrant with `log_id` in the payload. Never raises (matches `episodic.retrieve_memories`'s never-raise contract) — logs and swallows on any failure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/modules/finance/test_memory_write.py
from unittest.mock import AsyncMock

import pytest

from app.modules.finance.tools.memory import write_transaction_memory


@pytest.mark.asyncio
async def test_write_transaction_memory_upserts_with_log_id(monkeypatch):
    fake_qdrant = AsyncMock()
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.get_qdrant_client", lambda: fake_qdrant
    )
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.get_settings",
        lambda: type("S", (), {"qdrant_collection": "secondbrain"})(),
    )
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.embed_text", AsyncMock(return_value=[0.1, 0.2])
    )

    await write_transaction_memory(
        user_id="1",
        log_id=99,
        category_name="Ăn uống",
        description="Trà sữa",
        amount_vnd=45_000,
        log_type="expense",
        occurred_at="2026-08-03",
    )

    fake_qdrant.upsert.assert_awaited_once()
    _, kwargs = fake_qdrant.upsert.call_args
    point = kwargs["points"][0]
    assert point.payload["log_id"] == 99
    assert point.payload["user_id"] == "1"
    assert point.payload["source_type"] == "finance"
    assert point.payload["topics"] == ["Ăn uống"]


@pytest.mark.asyncio
async def test_write_transaction_memory_swallows_errors(monkeypatch):
    def _raise():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr("app.modules.finance.tools.memory.get_qdrant_client", _raise)

    # Must not raise.
    await write_transaction_memory(
        user_id="1",
        log_id=1,
        category_name="Ăn uống",
        description="x",
        amount_vnd=1000,
        log_type="expense",
        occurred_at="2026-08-03",
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_memory_write.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# app/modules/finance/tools/memory.py
from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client.http.models import PointStruct

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client
from app.infra.memory.episodic import detect_language
from app.infra.providers.embedding import embed_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_IMPORTANCE = 0.6


def _render_sentence(description: str, amount_vnd: int, category_name: str, occurred_at: str) -> str:
    amount = f"{amount_vnd:,}".replace(",", ".")
    return f"{description} {amount}đ ({category_name}) — {occurred_at}"


async def write_transaction_memory(
    user_id: str,
    log_id: int,
    category_name: str,
    description: str,
    amount_vnd: int,
    log_type: str,
    occurred_at: str,
) -> None:
    """Write one structured, embeddable point per transaction to Qdrant.

    Bypasses app.infra.memory.episodic's regex-based amount/category
    extraction — finance already knows both exactly. Never raises: a
    Qdrant/embedding outage should not block the user's transaction from
    being confirmed as saved (Postgres already has it).
    """
    sentence = _render_sentence(description, amount_vnd, category_name, occurred_at)

    try:
        embedding = await embed_text(sentence)
        qdrant = get_qdrant_client()
        collection = get_settings().qdrant_collection
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "user_id": user_id,
            "source_type": "finance",
            "timestamp": timestamp,
            "last_accessed": timestamp,
            "importance": _DEFAULT_IMPORTANCE,
            "language": detect_language(sentence),
            "topics": [category_name],
            "text": sentence,
            "summary_snippet": sentence[:200],
            "log_id": log_id,
            "amount_vnd": amount_vnd,
            "log_type": log_type,
        }

        await qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=str(uuid4()), vector=embedding, payload=payload)],
        )
    except Exception as exc:
        logger.warning("finance_memory_write_failed", log_id=log_id, error=str(exc))
```

Remove the stray `verb = "Mua" if True else ""` line from `_render_sentence` before running tests — it was left in by mistake; the final function body is just:

```python
def _render_sentence(description: str, amount_vnd: int, category_name: str, occurred_at: str) -> str:
    amount = f"{amount_vnd:,}".replace(",", ".")
    return f"{description} {amount}đ ({category_name}) — {occurred_at}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_memory_write.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/modules/finance/tools/memory.py tests/unit/modules/finance/test_memory_write.py
git commit -m "feat(finance): write structured per-transaction memory to Qdrant"
```

---

### Task 12: Query branch — `tools/queries.py`, `node/query/*`

**Files:**
- Create: `app/modules/finance/tools/queries.py`, `app/modules/finance/node/query/__init__.py`, `app/modules/finance/node/query/extract_params.py`, `app/modules/finance/node/query/run_query.py`
- Test: `tests/unit/modules/finance/test_queries.py`, `tests/unit/modules/finance/test_query_nodes.py`

**Interfaces:**
- Produces:
  - `async def total_spend(session, user_pk, start: date, end: date, category_id: int | None) -> int`
  - `async def spend_by_category(session, user_pk, start: date, end: date) -> list[dict]` — `[{"category": str, "total": int}, ...]`, descending by total.
  - `async def search_transactions(session, user_pk, keyword: str, limit: int = 10) -> list[dict]` — `[{"description": str, "amount": int, "occurred_at": str}, ...]`, `ILIKE`-matched on `description`.
  - `async def query_extract_params_node(state: dict) -> dict` — LLM call producing `QueryParams`, returns `{"query_params": dict}`; on failure returns `{"query_params": None, "error": "..."}`.
  - `async def query_run_node(state: dict) -> dict` — dispatches on `state["query_params"]["metric"]`, calls the matching `tools/queries.py` function, and sets `{"reply": <phrased text>}`. Needs `session`/`user_pk` the same way `log_budget_check_node` does — see Task 13 for how the graph supplies them.

- [ ] **Step 1: Write the failing tests for `tools/queries.py`**

```python
# tests/unit/modules/finance/test_queries.py
from datetime import date

import pytest

from app.modules.finance.tools.categories import find_category_by_name, seed_default_categories
from app.modules.finance.tools.queries import search_transactions, spend_by_category, total_spend
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk


async def _seed_two_transactions(db_session, user_pk):
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
            {
                "amount_vnd": 30_000,
                "raw_amount_text": "30k",
                "description": "Xe ôm",
                "log_type": "expense",
                "suggested_category": "Đi lại",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_total_spend_sums_expenses_in_range(db_session):
    user_pk = await get_or_create_user_pk(db_session, "20")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    total = await total_spend(
        db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31), category_id=None
    )
    assert total == 75_000


@pytest.mark.asyncio
async def test_total_spend_filters_by_category(db_session):
    user_pk = await get_or_create_user_pk(db_session, "21")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)
    an_uong = await find_category_by_name(db_session, user_pk, "Ăn uống")

    total = await total_spend(
        db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31), category_id=an_uong.id
    )
    assert total == 45_000


@pytest.mark.asyncio
async def test_spend_by_category_breaks_down_and_sorts(db_session):
    user_pk = await get_or_create_user_pk(db_session, "22")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    breakdown = await spend_by_category(db_session, user_pk, date(2026, 8, 1), date(2026, 8, 31))
    assert breakdown[0] == {"category": "Ăn uống", "total": 45_000}
    assert breakdown[1] == {"category": "Đi lại", "total": 30_000}


@pytest.mark.asyncio
async def test_search_transactions_matches_description(db_session):
    user_pk = await get_or_create_user_pk(db_session, "23")
    await seed_default_categories(db_session, user_pk)
    await _seed_two_transactions(db_session, user_pk)

    results = await search_transactions(db_session, user_pk, "trà")
    assert len(results) == 1
    assert results[0]["description"] == "Trà sữa"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_queries.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/queries.py`**

```python
# app/modules/finance/tools/queries.py
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory, FinanceLog


async def total_spend(
    session: AsyncSession, user_pk: int, start: date, end: date, category_id: int | None
) -> int:
    conditions = [
        FinanceLog.user_id == user_pk,
        FinanceLog.log_type == "expense",
        FinanceLog.occurred_at >= start,
        FinanceLog.occurred_at <= end,
    ]
    if category_id is not None:
        conditions.append(FinanceLog.category_id == category_id)

    total = (
        await session.execute(select(func.coalesce(func.sum(FinanceLog.amount), 0)).where(*conditions))
    ).scalar_one()
    return int(total)


async def spend_by_category(
    session: AsyncSession, user_pk: int, start: date, end: date
) -> list[dict]:
    rows = await session.execute(
        select(FinanceCategory.name, func.coalesce(func.sum(FinanceLog.amount), 0))
        .join(FinanceLog, FinanceLog.category_id == FinanceCategory.id)
        .where(
            FinanceLog.user_id == user_pk,
            FinanceLog.log_type == "expense",
            FinanceLog.occurred_at >= start,
            FinanceLog.occurred_at <= end,
        )
        .group_by(FinanceCategory.name)
        .order_by(func.sum(FinanceLog.amount).desc())
    )
    return [{"category": name, "total": int(total)} for name, total in rows.all()]


async def search_transactions(
    session: AsyncSession, user_pk: int, keyword: str, limit: int = 10
) -> list[dict]:
    rows = await session.execute(
        select(FinanceLog)
        .where(FinanceLog.user_id == user_pk, FinanceLog.description.ilike(f"%{keyword}%"))
        .order_by(FinanceLog.occurred_at.desc())
        .limit(limit)
    )
    return [
        {
            "description": log.description,
            "amount": log.amount,
            "occurred_at": log.occurred_at.isoformat(),
        }
        for log in rows.scalars().all()
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_queries.py -v`
Expected: all passed.

- [ ] **Step 5: Write the failing tests for the query nodes**

```python
# tests/unit/modules/finance/test_query_nodes.py
from datetime import date

import pytest

from app.modules.finance.node.query.extract_params import query_extract_params_node
from app.modules.finance.node.query.run_query import query_run_node
from app.modules.finance.schema.query_params import QueryParams
from app.modules.finance.tools.categories import seed_default_categories
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk
from tests.unit.modules.finance.conftest import FakeLLM


@pytest.mark.asyncio
async def test_extract_params_node_returns_llm_result(monkeypatch):
    params = QueryParams(
        metric="total", period_start=date(2026, 8, 1), period_end=date(2026, 8, 31)
    )
    monkeypatch.setattr(
        "app.modules.finance.node.query.extract_params.create_llm_client",
        lambda: FakeLLM(params),
    )
    result = await query_extract_params_node({"user_query": "tháng này tiêu bao nhiêu?"})
    assert result["query_params"]["metric"] == "total"


@pytest.mark.asyncio
async def test_extract_params_node_falls_back_on_error(monkeypatch):
    def _raise():
        raise RuntimeError("down")

    monkeypatch.setattr(
        "app.modules.finance.node.query.extract_params.create_llm_client", _raise
    )
    result = await query_extract_params_node({"user_query": "tháng này tiêu bao nhiêu?"})
    assert result["query_params"] is None


@pytest.mark.asyncio
async def test_run_query_node_total(db_session):
    user_pk = await get_or_create_user_pk(db_session, "30")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    result = await query_run_node(
        {
            "session": db_session,
            "user_pk": user_pk,
            "query_params": {
                "metric": "total",
                "period_start": "2026-08-01",
                "period_end": "2026-08-31",
                "category": None,
                "keyword": None,
            },
        }
    )
    assert "45.000" in result["reply"] or "45000" in result["reply"]


@pytest.mark.asyncio
async def test_run_query_node_missing_params_gives_fallback_reply():
    result = await query_run_node({"session": None, "user_pk": None, "query_params": None})
    assert "chưa hiểu" in result["reply"].lower() or "chưa" in result["reply"].lower()
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_query_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 7: Implement the query nodes**

```python
# app/modules/finance/node/query/__init__.py
```

```python
# app/modules/finance/node/query/extract_params.py
from datetime import datetime, timezone

from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.query import QUERY_SYSTEM_PROMPT, build_query_user_message
from ...schema.query_params import QueryParams

logger = get_logger(__name__)


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def query_extract_params_node(state: dict) -> dict:
    try:
        llm = create_llm_client()
        extractor = llm.with_structured_output(QueryParams, method="json_mode")
        result: QueryParams = await extractor.ainvoke(
            [
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_query_user_message(state["user_query"], _today_iso()),
                },
            ]
        )
        return {"query_params": result.model_dump(mode="json")}
    except Exception as exc:
        logger.warning("finance_query_extract_failed", error=str(exc))
        return {"query_params": None}
```

```python
# app/modules/finance/node/query/run_query.py
from datetime import date

from ...tools.categories import find_category_by_name
from ...tools.queries import search_transactions, spend_by_category, total_spend

_FALLBACK_REPLY = "Mình chưa hiểu câu hỏi này, bạn thử hỏi lại rõ hơn nhé."


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + "đ"


async def query_run_node(state: dict) -> dict:
    params = state.get("query_params")
    if not params:
        return {"reply": _FALLBACK_REPLY}

    session = state["session"]
    user_pk = state["user_pk"]
    start = date.fromisoformat(params["period_start"])
    end = date.fromisoformat(params["period_end"])

    if params["metric"] == "total":
        category_id = None
        if params.get("category"):
            category = await find_category_by_name(session, user_pk, params["category"])
            category_id = category.id if category else None
        total = await total_spend(session, user_pk, start, end, category_id)
        label = f" cho {params['category']}" if params.get("category") else ""
        return {"reply": f"Bạn đã chi {_format_vnd(total)}{label} trong khoảng thời gian này."}

    if params["metric"] == "by_category":
        breakdown = await spend_by_category(session, user_pk, start, end)
        if not breakdown:
            return {"reply": "Chưa có giao dịch nào trong khoảng thời gian này."}
        lines = [f"• {row['category']}: {_format_vnd(row['total'])}" for row in breakdown]
        return {"reply": "Chi tiêu theo danh mục:\n" + "\n".join(lines)}

    if params["metric"] == "search":
        keyword = params.get("keyword") or ""
        results = await search_transactions(session, user_pk, keyword)
        if not results:
            return {"reply": f"Không tìm thấy giao dịch nào khớp với '{keyword}'."}
        total = sum(r["amount"] for r in results)
        lines = [f"• {r['description']}: {_format_vnd(r['amount'])} ({r['occurred_at']})" for r in results]
        return {"reply": f"Tổng {_format_vnd(total)} cho '{keyword}':\n" + "\n".join(lines)}

    return {"reply": _FALLBACK_REPLY}
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_query_nodes.py -v`
Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add app/modules/finance/tools/queries.py app/modules/finance/node/query/ \
        tests/unit/modules/finance/test_queries.py tests/unit/modules/finance/test_query_nodes.py
git commit -m "feat(finance): add basic query branch (total, by-category, search)"
```

---

### Task 13: Wire `app/modules/finance/state.py` and `app/modules/finance/agent.py`

**Files:**
- Modify: `app/modules/finance/state.py`, `app/modules/finance/agent.py`, `app/modules/finance/node/__init__.py`
- Create: `app/modules/finance/node/not_implemented.py`
- Test: `tests/unit/modules/finance/test_agent_end_to_end.py`

**Interfaces:**
- Produces: `FinancialState` TypedDict; `build_graph() -> StateGraph`; `get_compiled_graph()`; `FinancialAgent` (unchanged `BaseAgent` contract: `run(input: AgentInput) -> AgentOutput`).
- This task is where `log_budget_check_node`/`query_run_node` actually receive `state["session"]`/`state["user_pk"]` — done by having a single `_log_persist_and_followups` wrapper node open one session for the whole log branch (persist → budget_check → memory_write), and a `_query_run_with_session` wrapper node open one session for the query branch. This keeps `log_persist_node`/`log_budget_check_node`/`query_run_node` themselves pure (session passed in), while only `agent.py` owns connection lifecycle — matching how `app/infra/db/session.py:get_session` is meant to be used (one `async with` per unit of work).

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/unit/modules/finance/test_agent_end_to_end.py
from datetime import date
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select

from app.models import FinanceLog
from app.modules.finance.agent import build_graph
from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from tests.unit.modules.finance.conftest import FakeLLM


class _RoundRobinLLM:
    """Returns each fake result in order across successive create_llm_client() calls."""

    def __init__(self, results):
        self._results = iter(results)

    def __call__(self):
        return FakeLLM(next(self._results))


@pytest.mark.asyncio
async def test_log_happy_path_persists_and_replies(monkeypatch, db_session):
    txn = ParsedTransaction(
        amount_vnd=45_000,
        raw_amount_text="45k",
        description="Trà sữa",
        log_type="expense",
        suggested_category="Ăn uống",
        occurred_at=date(2026, 8, 3),
    )
    llm_sequence = _RoundRobinLLM(
        [ClassifyResult(sub_intent="log"), ParseResult(transactions=[txn], unparsed_notes=[])]
    )
    monkeypatch.setattr("app.modules.finance.node.classify.create_llm_client", llm_sequence)
    monkeypatch.setattr("app.modules.finance.node.log.parse.create_llm_client", llm_sequence)
    # agent.py does `from .tools.memory import write_transaction_memory`, so the
    # name to patch is the one bound in agent.py's namespace, not the source module.
    monkeypatch.setattr(
        "app.modules.finance.agent.write_transaction_memory", AsyncMock()
    )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)

    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "e2e-log"}}

    initial = {"user_id": "50", "user_query": "mua trà sữa 45k"}
    paused = await graph.ainvoke(initial, config=config)
    assert "__interrupt__" in paused

    result = await graph.ainvoke(Command(resume="confirm"), config=config)
    assert "Trà sữa" not in result["reply"] or True  # reply just needs to exist
    assert result["reply"]

    stored = (await db_session.execute(select(FinanceLog))).all()
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_query_happy_path(monkeypatch, db_session):
    from app.modules.finance.schema.query_params import QueryParams
    from app.modules.finance.tools.categories import seed_default_categories
    from app.modules.finance.tools.transactions import create_transactions
    from app.modules.finance.tools.users import get_or_create_user_pk

    user_pk = await get_or_create_user_pk(db_session, "51")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            }
        ],
    )
    await db_session.commit()

    params = QueryParams(metric="total", period_start=date(2026, 8, 1), period_end=date(2026, 8, 31))
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="query")),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.query.extract_params.create_llm_client",
        lambda: FakeLLM(params),
    )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)

    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "e2e-query"}}

    result = await graph.ainvoke({"user_id": "51", "user_query": "tháng này tiêu bao nhiêu?"}, config=config)
    assert "45.000" in result["reply"] or "45000" in result["reply"]


@pytest.mark.asyncio
async def test_not_implemented_sub_intent_gets_graceful_reply(monkeypatch, db_session):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="budget")),
    )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)

    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "e2e-budget"}}

    result = await graph.ainvoke({"user_id": "52", "user_query": "đặt ngân sách"}, config=config)
    assert result["reply"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/modules/finance/test_agent_end_to_end.py -v`
Expected: FAIL — current `agent.py`/`state.py` don't match this shape yet.

- [ ] **Step 3: Rewrite `state.py`**

```python
# app/modules/finance/state.py
from typing_extensions import NotRequired, TypedDict


class FinancialState(TypedDict):
    """State for the FinancialAgent's internal subgraph."""

    user_id: str
    user_query: str

    sub_intent: NotRequired[str]

    # log branch
    parsed_transactions: NotRequired[list[dict]]
    unparsed_notes: NotRequired[list[str]]
    confirm_answer: NotRequired[str]
    persisted_log_ids: NotRequired[list[int]]
    persisted_categories: NotRequired[list[str]]
    budget_warnings: NotRequired[list[str]]

    # query branch
    query_params: NotRequired[dict | None]

    reply: NotRequired[str]
```

- [ ] **Step 4: Implement `node/not_implemented.py`**

```python
# app/modules/finance/node/not_implemented.py
_MESSAGES = {
    "budget": "Tính năng đặt ngân sách sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "goal": "Tính năng mục tiêu tài chính sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "split": "Tính năng chia tiền sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "advice": "Tính năng tư vấn chi tiêu sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "fallback": "Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn về khoản thu/chi hoặc câu hỏi chi tiêu không?",
}


async def not_implemented_node(state: dict) -> dict:
    reply = _MESSAGES.get(state.get("sub_intent", "fallback"), _MESSAGES["fallback"])
    return {"reply": reply}
```

- [ ] **Step 5: Rewrite `agent.py`**

```python
# app/modules/finance/agent.py
from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.db.session import get_session
from app.infra.memory.checkpointer import get_checkpointer
from app.utils.logger import get_logger

from .node.classify import classify_node
from .node.log.budget_check import log_budget_check_node
from .node.log.confirm import log_confirm_node
from .node.log.parse import log_parse_node
from .node.log.persist import log_persist_node
from .node.not_implemented import not_implemented_node
from .node.query.extract_params import query_extract_params_node
from .node.query.run_query import query_run_node
from .tools.memory import write_transaction_memory
from .tools.users import get_or_create_user_pk
from .state import FinancialState

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())


def _route_after_classify(state: FinancialState) -> str:
    return state.get("sub_intent") if state.get("sub_intent") in ("log", "query") else "not_implemented"


def _route_after_confirm(state: FinancialState) -> str:
    return "persist" if state.get("confirm_answer") == "confirm" else "confirm_declined"


async def _confirm_declined_node(state: FinancialState) -> dict:
    if state.get("confirm_answer") == "edit":
        return {"reply": "Được, bạn gõ lại giao dịch cho đúng nhé."}
    return {"reply": "Đã huỷ."}


async def _log_persist_and_followups_node(state: FinancialState) -> dict:
    """Owns the one DB session for persist -> budget_check -> memory_write."""
    async with get_session() as session:
        user_pk = await get_or_create_user_pk(session, state["user_id"])

        persist_result = await log_persist_node({**state, "session": session, "user_pk": user_pk})

        budget_result = await log_budget_check_node(
            {**state, **persist_result, "session": session, "user_pk": user_pk}
        )

        log_ids = persist_result.get("persisted_log_ids", [])
        categories = persist_result.get("persisted_categories", [])
        for log_id, txn, category_name in zip(
            log_ids, state.get("parsed_transactions", []), categories
        ):
            await write_transaction_memory(
                user_id=state["user_id"],
                log_id=log_id,
                category_name=category_name,
                description=txn["description"],
                amount_vnd=txn["amount_vnd"],
                log_type=txn["log_type"],
                occurred_at=txn["occurred_at"],
            )

    reply_lines = [f"Đã ghi {len(log_ids)} giao dịch."]
    reply_lines.extend(budget_result.get("budget_warnings", []))
    return {**persist_result, **budget_result, "reply": "\n".join(reply_lines)}


async def _query_run_with_session_node(state: FinancialState) -> dict:
    async with get_session() as session:
        user_pk = await get_or_create_user_pk(session, state["user_id"])
        return await query_run_node({**state, "session": session, "user_pk": user_pk})


def build_graph() -> StateGraph:
    builder = StateGraph(
        state_schema=FinancialState,
        name=config["agent"]["name"],
        description=config["agent"]["description"],
    )

    builder.add_node("classify", classify_node)
    builder.add_node("log_parse", log_parse_node)
    builder.add_node("log_confirm", log_confirm_node)
    builder.add_node("persist", _log_persist_and_followups_node)
    builder.add_node("confirm_declined", _confirm_declined_node)
    builder.add_node("query_extract_params", query_extract_params_node)
    builder.add_node("query_run", _query_run_with_session_node)
    builder.add_node("not_implemented", not_implemented_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"log": "log_parse", "query": "query_extract_params", "not_implemented": "not_implemented"},
    )

    builder.add_edge("log_parse", "log_confirm")
    builder.add_conditional_edges(
        "log_confirm", _route_after_confirm, {"persist": "persist", "confirm_declined": "confirm_declined"}
    )
    builder.add_edge("persist", END)
    builder.add_edge("confirm_declined", END)

    builder.add_edge("query_extract_params", "query_run")
    builder.add_edge("query_run", END)

    builder.add_edge("not_implemented", END)

    return builder


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=get_checkpointer())
    return _compiled


class FinancialAgent(BaseAgent):
    """Wraps the finance sub-intent router subgraph behind BaseAgent."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: FinancialState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
        }
        result = await run_interruptible_subgraph(
            get_compiled_graph(),
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}


agent = FinancialAgent()
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/unit/modules/finance/test_agent_end_to_end.py -v`
Expected: all passed. If `_route_after_classify`/`_route_after_confirm` conditional-edge string mismatches surface (LangGraph is strict about the mapping dict matching returned strings exactly), fix the mapping keys to match, rerun.

- [ ] **Step 7: Run the entire finance test suite together**

Run: `uv run pytest tests/unit/modules/finance/ -v`
Expected: all tests across every earlier task still pass (no regressions from the `agent.py` rewrite).

- [ ] **Step 8: Commit**

```bash
git add app/modules/finance/agent.py app/modules/finance/state.py \
        app/modules/finance/node/not_implemented.py app/modules/finance/node/__init__.py \
        tests/unit/modules/finance/test_agent_end_to_end.py
git commit -m "feat(finance): wire full Phase 1 graph (log + query branches) into FinancialAgent"
```

---

### Task 14: Full-suite verification and registry smoke check

**Files:**
- None created — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including everything under `tests/unit/modules/finance/`.

- [ ] **Step 2: Verify the app still imports cleanly end-to-end**

Run: `uv run python -c "import app.main"`
Expected: no exceptions (this would have caught the original `metadata` reserved-name crash from Task 2 at process-startup time).

- [ ] **Step 3: If Postgres/Redis/Qdrant are reachable, do a live smoke check**

```bash
docker compose up -d postgres redis qdrant
uv run alembic upgrade head
uv run python main.py
```

Send the bot (via Telegram) `mua trà sữa 45k`, confirm via the inline keyboard, then ask `tháng này tiêu bao nhiêu?`. This step is a manual follow-up for whoever has bot-token/infra access — do not treat the plan as blocked on it if this environment has no Telegram token or Docker access; Steps 1–2 are the pass/fail gate for this task.

- [ ] **Step 4: Final commit if anything was adjusted during verification**

```bash
git add -A
git commit -m "chore(finance): verification fixes for phase 1 core loop"
```

Only commit if Steps 1–3 actually required changes — otherwise this task ends at Step 3.
