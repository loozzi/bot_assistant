# Finance Agent — Design Spec

**Date:** 2026-08-03
**Status:** Approved design, pre-implementation
**Branch:** `feature/financial-agent`

## Goal

A personal-finance module (`app/modules/finance/`) for the Telegram bot covering:
natural-language transaction logging with user confirmation, categorized spend
analysis with charts and month comparisons, per-category budgets with log-time
alerts, financial goals, spend-intent advice, and a bill-splitting debt ledger.
Vietnamese-first, English tolerated.

Non-goals (future work, out of this spec's phases): proactive scheduled push
notifications, multi-currency, shared expenses between bot users,
bank-statement import.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Scope | One spec covering all features; five phased implementation plans |
| Categories | Per-user seeded Vietnamese taxonomy; LLM assigns during parse; user can correct/add/rename via chat |
| Currency | VND only, stored as integer VND; colloquial parsing ("200k", "2tr", "25 nghìn"); foreign amounts politely rejected |
| Charts | Server-rendered matplotlib PNG sent as Telegram photo |
| Bill split | Persistent debt ledger with partial settlement, free-text counterparty names |
| Budget alerts | Checked at log time, appended to the confirmation reply (≥80% warn, >100% over) |
| Query engine | Fixed typed query tools + LLM-extracted parameters; never LLM-generated SQL |
| Goals | Free-text statement + optional numeric fields (target amount, deadline, monthly target) |
| Architecture | Sub-intent router subgraph (approach A); deterministic math in `tools/`, LLM only classifies/parses/phrases |

## Architecture

The module follows the `search/` reference pattern: an internal LangGraph
`StateGraph` behind `BaseAgent.run()`, already scaffolded in
`app/modules/finance/agent.py` with a checkpointer and the
`app/core/hitl.py:run_interruptible_subgraph` bridge (one pause per run).

### State

`FinancialState` (TypedDict, `state.py`):

```python
user_id: str            # telegram id as passed by AgentInput
user_query: str
sub_intent: str         # log | query | budget | goal | split | advice | fallback
parsed_transactions: list[dict]   # ParsedTransaction dumps (log branch)
split_event: dict | None          # SplitEvent dump (split branch)
query_result: dict | None         # structured numbers for reply phrasing
chart_path: str | None
reply: str
error: str | None
```

### Graph shape

```
classify ──► log:    parse ► confirm(HITL) ► persist ► budget_check ► memory_write ► respond
         ──► query:  extract_params ► run_query_tools ► [render_chart] ► respond
         ──► budget: parse ► apply ► respond
         ──► goal:   parse ► apply ► respond
         ──► split:  parse ► confirm(HITL) ► persist_ledger ► respond
         ──► advice: gather_context ► advise ► log_intent ► respond
         ──► fallback ► respond
```

- `classify`: one LLM structured-output call returning a single primary
  sub-intent. **v1 limitation (documented behavior):** a message mixing a log
  and a question is handled as its primary intent only; cross-module
  multi-intent remains the orchestrator's job.
- Division of labor: the LLM classifies, parses into Pydantic schemas
  (`schema/`), and phrases replies from precomputed numbers. All arithmetic —
  sums, comparisons, split shares, budget percentages, projections — is
  deterministic Python in `tools/`. Money figures in replies always come from
  code, never from the LLM.
- HITL confirmation gates only the two branches that create financial records
  from fuzzy parsing: `log` and `split`. Budget/goal setting applies
  immediately and echoes the result (single low-risk value, correctable by
  restating).
- `user_id` arrives as the Telegram id string (`AgentInput.user_id`); a helper
  (`tools/users.py`) resolves it to the `users.id` PK, creating the row and
  seeding categories on first finance interaction.

### Cross-cutting change: attachments

Charts must reach Telegram. `AgentOutput` (`app/core/base_agent.py`) gains an
optional key:

```python
attachments: NotRequired[list[dict]]  # {"type": "photo", "path": str}
```

`orchestrator/graph.py:format_response` passes attachments through (union of
all module outputs' attachments when merging), and `bot/handlers.py` sends
each photo attachment after the text reply. Backward-compatible — `search`
untouched.

### Module layout

```
app/modules/finance/
  agent.py            # exists; graph gets real nodes/edges
  state.py            # FinancialState (extended)
  config.yaml         # exists; routing already live
  node/               # classify.py, log.py, query.py, budget.py, goal.py,
                      # split.py, advice.py, respond.py
  tools/              # amounts.py, users.py, categories.py, transactions.py,
                      # queries.py, budgets.py, goals.py, ledger.py,
                      # analytics.py, charts.py
  schema/             # parsed_transaction.py, query_params.py, split_event.py,
                      # budget_command.py, goal_command.py, classify_result.py
  prompts/            # classify.py, parse.py, reply.py, advice.py
```

No cross-module imports; shared logic goes to `infra/` or `utils/`.

## Data model (Postgres)

Postgres is the **source of truth for every number**. One Alembic
autogenerate migration covers all changes below.

### Fixes to existing models (`app/models.py`)

- `FinanceLog.metadata` → **rename to `extra`** (JSONB). `metadata` is a
  reserved Declarative attribute and crashes at import.
- `amount: float` → `BigInteger` integer VND (everywhere money appears).
- `log_type: Mapped[Literal[...]]` (no SQL type) → `Mapped[str]` +
  `CheckConstraint("log_type IN ('income','expense')")`.
- Real FKs: `FinanceLog.user_id → users.id`;
  `FinanceLog.category_id → finance_categories.id` (nullable, `ondelete="SET NULL"`).
- Add `FinanceLog.occurred_at: date` (indexed with `user_id`) — when the
  spend happened ("hôm qua ăn phở") vs `created_at` (when logged). All
  analytics group by `occurred_at`.
- `FinanceCategory`: `name` globally-unique → `UniqueConstraint(user_id, name)`;
  FK `user_id → users.id`; add `group: str`
  (`essential | lifestyle | savings_debt`, CheckConstraint) and
  `is_seeded: bool`.

### Category seeding

On a user's first finance interaction, seed: ăn uống, đi lại, nhà cửa,
hóa đơn & tiện ích, mua sắm, giải trí, sức khỏe, giáo dục,
tiết kiệm & đầu tư, khác — each with a `group` assignment. Users add/rename
categories via chat: `classify` maps category-management requests
("thêm danh mục thú cưng", "đổi tên X thành Y") to the `budget` branch, which
owns all settings-type commands (budgets + categories). Renames update the
category row in place; transaction history follows via FK.

### New tables

```
finance_budgets
  id, user_id FK, category_id FK nullable,  -- NULL = overall monthly budget
  amount BigInteger (VND per calendar month),
  created_at, updated_at
  UNIQUE (user_id, category_id)             -- v1: monthly period only

finance_goals
  id, user_id FK, description Text,
  target_amount BigInteger nullable, deadline Date nullable,
  monthly_target BigInteger nullable,
  status ('active'|'achieved'|'archived'), created_at, updated_at

finance_debts
  id, user_id FK,                            -- owner
  counterparty Text,                         -- free-text name
  direction ('owed_to_me'|'i_owe'),
  amount_total BigInteger, amount_outstanding BigInteger,
  description Text,                          -- event, e.g. "lẩu 4 người 600k"
  status ('open'|'settled'), extra JSONB,    -- settlement history [{date, amount}]
  created_at, settled_at nullable
```

## Memory (Qdrant)

Qdrant serves **semantic recall only, never aggregation**.

- `memory_write` (log branch, after persist): per transaction, embed a natural
  sentence — "Mua trà sữa 45k (ăn uống) — 03/08/2026" — via
  `app/infra/providers/embedding.py:embed_text`, upsert a point with the
  standard payload (`user_id`, `source_type: "finance"`, `timestamp`,
  `importance: 0.6` from config.yaml, `topics: [category]`, `language`) plus
  **`log_id`** referencing the Postgres row.
- Free-text lookups ("món A hết bao nhiêu?"): SQL `ILIKE` on
  `FinanceLog.description` first; if thin, Qdrant semantic search →
  collect `log_id`s → fetch exact rows → reply quotes Postgres numbers.
- Advice-branch purchase intents are written as points with topic
  `purchase_intent` so past "user muốn mua X" moments are recallable.
- The keyword-regex `_extract_amount`/`_extract_category` helpers in
  `app/infra/memory/episodic.py` are bypassed — finance writes structured
  payloads itself.

## Feature flows

### Log (parse → confirm → persist → budget_check → memory_write)

- `parse`: LLM structured output → `list[ParsedTransaction]`:
  `amount_vnd: int`, `description: str`, `log_type`, `suggested_category`,
  `occurred_at: date` (relative dates resolved against Asia/Ho_Chi_Minh
  today). Multiple transactions per message supported.
- `tools/amounts.py`: deterministic colloquial-VND normalizer
  ("200k"→200_000, "2tr"/"2 triệu"→2_000_000, "25 nghìn"→25_000,
  "1tr2"→1_200_000). It validates/corrects the LLM's parsed number from the
  raw text span; disagreement resurfaces in the confirm message. Heavily
  unit-tested — a silent ×1000 error poisons all downstream analytics.
- `confirm` (HITL): numbered list ("1. Ăn trưa — 50.000đ — ăn uống …") with
  options **Xác nhận / Sửa / Huỷ**. "Sửa" resumes with free-text correction,
  which is re-parsed and re-confirmed on a fresh run (the HITL bridge allows
  one pause per invocation).
- `persist`: insert rows in one transaction. `budget_check`: month-to-date vs
  budget per affected category; append "⚠️ đã dùng 85% ngân sách ăn uống"
  at ≥80%, over-budget notice at >100%. Anomaly note: amount far above the
  category's rolling average adds a gentle remark (never blocks).
- Foreign-currency amounts ("$50") → reply asks to restate in VND.

### Query (extract_params → run_query_tools → [render_chart])

- `extract_params`: LLM → `QueryParams`
  `{metric, period, category?, keyword?, compare_to?, wants_chart}`.
- `tools/queries.py` (parameterized SQLAlchemy, always filtered by user):
  `total_spend`, `spend_by_category`, `month_comparison`, `top_expenses`,
  `search_transactions`, `income_vs_expense`.
- `tools/analytics.py` (derived, deterministic):
  - **Burn rate & projection** — avg daily spend → projected month-end.
  - **Recurring detection** — similar description+amount across months;
    flags missing or price-jumped recurring items.
  - **Savings rate** — (income − expense)/income monthly, trended.
  - **50/30/20 lens** — spend by category `group` vs the heuristic
    (framing device, not a rule).
  - **Monthly review** ("tổng kết tháng 7") — composes totals, top
    categories, vs previous month, budget adherence, goal progress,
    recurring items, one advice paragraph.
- `render_chart` (`tools/charts.py`, matplotlib `Agg`, DejaVu Sans for
  Vietnamese): pie (category breakdown), grouped bar (month comparison),
  line (daily cumulative). PNG written to the app's temp dir; path returned
  via `attachments`. Rendered when `wants_chart` or for comparisons.
- Reply: LLM phrases the precomputed structured numbers; it never computes.

### Budget & goal

- Parse command (set/update/delete/list) → upsert → echo with live context:
  "Đã đặt ngân sách ăn uống 3.000.000đ/tháng — tháng này đã dùng 1.2tr (40%)".
- Goal progress: monthly net savings (income − expense) or user-stated
  deposits into tiết kiệm & đầu tư, vs `target_amount`/`deadline`/
  `monthly_target` when present.

### Split (parse → confirm → persist_ledger)

- `parse` → `SplitEvent`: `total`, `payer` ("tôi" or a name),
  `participants`, equal or custom shares.
- Deterministic share math: round to 1.000đ, remainder assigned to payer.
- Confirm shows per-person amounts; on confirm, one `finance_debts` row per
  counterparty (direction depends on who paid).
- Settlement ("Nam trả tôi 150k"): match open debts by counterparty
  (case-insensitive); ambiguity → ask which debt; partial payments decrement
  `amount_outstanding`, history appended to `extra`; zero → `settled`.
- Queries: "ai còn nợ tôi?", "tôi nợ ai bao nhiêu?" from open rows.

### Advice (gather_context → advise → log_intent)

- Context: current-month stats, budget states, active goals (free text is the
  advisory "lens"), Qdrant recall of similar purchases/intents.
- LLM answers "có nên tiêu không?" through the goal lens with the numbers
  provided; `log_intent` writes the purchase intent to Qdrant.

## Error handling

Philosophy (as `search/`): degrade gracefully, never crash the graph.

- Classify/parse failure or schema violation → "mình chưa hiểu, bạn nói rõ
  hơn được không?"; nothing written.
- **No persistence without confirm** on log/split. Cancel/abandon leaves no
  rows; the HITL checkpoint thread prevents double-writes on re-sent
  confirmations.
- Amount sanity guard: outside 1.000đ–500.000.000đ → flagged in confirm
  message rather than silently saved.
- DB errors → `state["error"]`, apologetic reply. Qdrant write failures are
  logged and swallowed (Postgres has the truth; matches `retrieve_memories`'
  never-raise contract).
- Chart failure → text summary still sent, chart failure noted.

## Testing

First real test setup in the repo: pytest + pytest-asyncio config added to
`pyproject.toml`, tests under `tests/unit/modules/finance/`. Priority order:

1. `tools/amounts.py` — table-driven colloquial-VND cases (highest-risk pure
   function).
2. Split share math incl. rounding/remainder.
3. `tools/queries.py` + `tools/analytics.py` against async-SQLAlchemy SQLite
   fixtures (seeded transactions).
4. Budget threshold logic; goal progress.
5. Graph wiring/state transitions with the LLM structured-output call mocked
   (no live-LLM tests).

## Implementation phasing

One spec, five plans; each phase independently shippable:

1. **Core loop** — model fixes + migration, category seeding, log branch
   end-to-end, basic queries (totals, by-category, search), memory write.
2. **Analytics** — month comparison, top expenses, charts +
   `AgentOutput.attachments` plumbing, burn-rate projection.
3. **Budgets & goals** — tables in use, set/view/delete flows, log-time
   threshold warnings, goal progress.
4. **Bill split** — debts ledger, split/settle/query flows.
5. **Advice & review** — advice branch with goal lens, purchase-intent
   memory, monthly review, recurring + anomaly detection.
