from datetime import date
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select

from app.models import FinanceLog
from app.modules.finance.agent import (
    _confirm_declined_node,
    _log_persist_and_followups_node,
    _query_run_with_session_node,
    build_graph,
)
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
    mock_write_memory = AsyncMock()
    monkeypatch.setattr(
        "app.modules.finance.agent.write_transaction_memory", mock_write_memory
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
    assert result["reply"]
    assert "Đã ghi 1 giao dịch." in result["reply"]

    stored = (await db_session.execute(select(FinanceLog))).all()
    assert len(stored) == 1

    mock_write_memory.assert_awaited_once_with(
        user_id="50",
        log_id=stored[0][0].id,
        category_name="Ăn uống",
        description="Trà sữa",
        amount_vnd=45_000,
        log_type="expense",
        occurred_at="2026-08-03",
    )


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


@pytest.mark.asyncio
async def test_confirm_declined_edit_surfaces_unparsed_notes():
    """Issue 1: when log_parse_node drops something (foreign currency,
    out-of-range amount, etc.), the reason should reach the user, not just
    the generic 'type it again' message."""
    state = {
        "user_id": "60",
        "confirm_answer": "edit",
        "unparsed_notes": ["Bỏ qua 'vé máy bay' vì số tiền có vẻ không hợp lý."],
    }

    result = await _confirm_declined_node(state)

    assert "Được, bạn gõ lại giao dịch cho đúng nhé." in result["reply"]
    assert "Bỏ qua 'vé máy bay' vì số tiền có vẻ không hợp lý." in result["reply"]


@pytest.mark.asyncio
async def test_confirm_declined_without_notes_keeps_generic_message():
    result = await _confirm_declined_node({"user_id": "60", "confirm_answer": "edit"})
    assert result["reply"] == "Được, bạn gõ lại giao dịch cho đúng nhé."


@pytest.mark.asyncio
async def test_confirm_declined_free_text_answer_is_treated_as_restate_not_cancel():
    """Finding 8: bot/handlers.py resumes the interrupt with whatever raw
    text the user typed if they don't tap a button — that free text lands in
    confirm_answer. Investigation confirmed the Huỷ button routes through a
    completely separate handler (cb_hitl_cancel) that clears the whole
    checkpoint thread before ever resuming this graph, so confirm_answer in
    THIS graph is only ever "confirm", "edit", or arbitrary free text — never
    a genuine cancel signal. Any non-"confirm" value must be treated as an
    invitation to restate, not silently discarded as "Đã huỷ."."""
    result = await _confirm_declined_node(
        {"user_id": "60", "confirm_answer": "à không, phải là 50k chứ không phải 45k"}
    )
    assert "gõ lại" in result["reply"]
    assert result["reply"] != "Đã huỷ."


@pytest.mark.asyncio
async def test_log_persist_reply_includes_unparsed_notes(monkeypatch, db_session):
    """Issue 1: a message with one valid transaction and one dropped note
    (e.g. an absurdly large amount) should mention the dropped note in the
    final reply, not just report the count of logged transactions."""
    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", AsyncMock())

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)

    state = {
        "user_id": "61",
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
        "unparsed_notes": ["Bỏ qua 'cái kia' vì số tiền có vẻ không hợp lý."],
    }

    result = await _log_persist_and_followups_node(state)

    assert "Đã ghi 1 giao dịch." in result["reply"]
    assert "Bỏ qua 'cái kia' vì số tiền có vẻ không hợp lý." in result["reply"]


@pytest.mark.asyncio
async def test_log_persist_db_error_returns_safe_reply_not_exception(monkeypatch):
    """Finding 1: a DB error inside the persist/budget/memory session must not
    propagate — it should be caught and turned into a safe reply, otherwise
    the graph checkpoint is left mid-turn and the user's next message gets
    silently consumed as a resume answer."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _raising_session():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("app.modules.finance.agent.get_session", _raising_session)

    state = {
        "user_id": "70",
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
        "unparsed_notes": [],
    }

    result = await _log_persist_and_followups_node(state)

    assert "reply" in result
    assert result["reply"]


@pytest.mark.asyncio
async def test_query_run_db_error_returns_safe_reply_not_exception(monkeypatch):
    """Finding 1: same never-raise contract for the query branch's DB node."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _raising_session():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("app.modules.finance.agent.get_session", _raising_session)

    state = {
        "user_id": "71",
        "query_params": {
            "metric": "total",
            "period_start": "2026-08-01",
            "period_end": "2026-08-31",
            "category": None,
            "keyword": None,
        },
    }

    result = await _query_run_with_session_node(state)

    assert "reply" in result
    assert result["reply"]


@pytest.mark.asyncio
async def test_log_persist_memory_write_happens_after_session_closes(monkeypatch, db_session):
    """Finding 7: the Qdrant/embedding memory writes must run after the
    Postgres session/transaction has exited (not inside the `async with`
    block), so a slow embedding call doesn't hold the DB transaction open.
    We verify this by asserting write_transaction_memory is called with the
    session already committed/closed (db_session is a live session here, so
    we instead assert call ordering via a marker on the fake session)."""
    calls = []

    async def _mock_write_memory(**kwargs):
        calls.append(("write_memory", kwargs["log_id"]))

    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", _mock_write_memory)

    from contextlib import asynccontextmanager

    session_open = {"value": False}

    @asynccontextmanager
    async def _tracking_session():
        session_open["value"] = True
        try:
            yield db_session
        finally:
            session_open["value"] = False

    monkeypatch.setattr("app.modules.finance.agent.get_session", _tracking_session)

    state = {
        "user_id": "72",
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
        "unparsed_notes": [],
    }

    # Patch write_transaction_memory to record whether the session was still
    # "open" (per our tracking context manager) at call time.
    session_open_during_write = []

    async def _mock_write_memory_2(**kwargs):
        session_open_during_write.append(session_open["value"])

    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", _mock_write_memory_2)

    await _log_persist_and_followups_node(state)

    assert session_open_during_write == [False]
