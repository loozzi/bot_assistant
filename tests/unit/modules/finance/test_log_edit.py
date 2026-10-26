from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from sqlalchemy import select
from typing_extensions import TypedDict

from app.models import FinanceLog
from app.modules.finance.agent import FinancialAgent, build_graph
from app.modules.finance.node.log.edit import MAX_EDIT_ROUNDS, log_apply_edit_node
from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from tests.unit.modules.finance.conftest import FakeLLM


def _txn(description: str, amount: int, raw: str) -> ParsedTransaction:
    return ParsedTransaction(
        amount_vnd=amount,
        raw_amount_text=raw,
        description=description,
        log_type="expense",
        suggested_category="Ăn uống",
        occurred_at=date(2026, 8, 3),
    )


def _pending(description: str, amount: int, raw: str) -> dict:
    return {**_txn(description, amount, raw).model_dump(mode="json"), "amount_adjusted": False}


def _raise():
    raise RuntimeError("llm down")


@pytest.fixture
def patched_io(monkeypatch, db_session):
    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)
    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", AsyncMock())


class _EditLLMs:
    def __init__(self, results):
        self._results = iter(results)

    def __call__(self):
        nxt = next(self._results)
        if nxt is _raise:
            _raise()
        return FakeLLM(nxt)


# --- log_apply_edit_node ----------------------------------------------------


async def test_apply_edit_replaces_list_and_reverifies_amounts(monkeypatch):
    # The LLM claims 5_000 but its raw text says 50k — the deterministic parser wins.
    edited = ParseResult(transactions=[_txn("Trà sữa", 5_000, "50k")])
    monkeypatch.setattr("app.modules.finance.node.log.edit.create_llm_client", lambda: FakeLLM(edited))

    result = await log_apply_edit_node(
        {"parsed_transactions": [_pending("Trà sữa", 45_000, "45k")], "edit_instruction": "là 50k"}
    )

    assert result["parsed_transactions"][0]["amount_vnd"] == 50_000
    assert result["parsed_transactions"][0]["amount_adjusted"] is True
    assert result["edit_rounds"] == 1
    assert "reply" not in result


async def test_apply_edit_llm_failure_keeps_list_and_explains(monkeypatch):
    monkeypatch.setattr("app.modules.finance.node.log.edit.create_llm_client", _raise)

    result = await log_apply_edit_node(
        {"parsed_transactions": [_pending("Trà sữa", 45_000, "45k")], "edit_rounds": 2}
    )

    assert "parsed_transactions" not in result
    assert result["edit_rounds"] == 3
    assert result["edit_notice"]


async def test_apply_edit_removing_everything_ends_with_reply(monkeypatch):
    monkeypatch.setattr(
        "app.modules.finance.node.log.edit.create_llm_client", lambda: FakeLLM(ParseResult())
    )

    result = await log_apply_edit_node(
        {"parsed_transactions": [_pending("Trà sữa", 45_000, "45k")], "edit_instruction": "bỏ hết"}
    )

    assert result["parsed_transactions"] == []
    assert "không ghi" in result["reply"]


# --- full finance graph, standalone ----------------------------------------


def _setup(monkeypatch, edits: list):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="log")),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client",
        lambda: FakeLLM(ParseResult(transactions=[_txn("Trà sữa", 45_000, "45k"), _txn("Bánh mì", 20_000, "20k")])),
    )
    monkeypatch.setattr("app.modules.finance.node.log.edit.create_llm_client", _EditLLMs(edits))


async def _amounts(db_session) -> list[int]:
    return sorted(log.amount for log in (await db_session.execute(select(FinanceLog))).scalars())


async def test_typed_correction_at_confirm_reconfirms_then_persists(monkeypatch, db_session, patched_io):
    _setup(monkeypatch, [ParseResult(transactions=[_txn("Trà sữa", 45_000, "45k"), _txn("Bánh mì", 25_000, "25k")])])
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-typed"}}

    await graph.ainvoke({"user_id": "70", "user_query": "trà sữa 45k, bánh mì 20k"}, config=config)
    paused = await graph.ainvoke(Command(resume="món 2 là 25k"), config=config)

    question = paused["__interrupt__"][0].value["question"]
    assert "25.000đ" in question
    assert await _amounts(db_session) == []

    result = await graph.ainvoke(Command(resume="confirm"), config=config)
    assert "Đã ghi 2 giao dịch." in result["reply"]
    assert await _amounts(db_session) == [25_000, 45_000]


async def test_edit_button_asks_for_correction_text(monkeypatch, db_session, patched_io):
    _setup(monkeypatch, [ParseResult(transactions=[_txn("Bánh mì", 20_000, "20k")])])
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-button"}}

    await graph.ainvoke({"user_id": "71", "user_query": "trà sữa 45k, bánh mì 20k"}, config=config)
    asked = await graph.ainvoke(Command(resume="edit"), config=config)
    assert "Bạn muốn sửa gì?" in asked["__interrupt__"][0].value["question"]
    assert "options" not in asked["__interrupt__"][0].value

    reconfirm = await graph.ainvoke(Command(resume="bỏ món 1"), config=config)
    assert "Trà sữa" not in reconfirm["__interrupt__"][0].value["question"]

    await graph.ainvoke(Command(resume="confirm"), config=config)
    assert await _amounts(db_session) == [20_000]


async def test_failed_edit_shows_notice_and_keeps_list(monkeypatch, db_session, patched_io):
    _setup(monkeypatch, [_raise])
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-fail"}}

    await graph.ainvoke({"user_id": "72", "user_query": "trà sữa 45k, bánh mì 20k"}, config=config)
    paused = await graph.ainvoke(Command(resume="ờm cái kia ấy"), config=config)

    question = paused["__interrupt__"][0].value["question"]
    assert "chưa áp dụng được" in question
    assert "45.000đ" in question and "20.000đ" in question


async def test_edit_rounds_are_capped(monkeypatch, db_session, patched_io):
    unchanged = ParseResult(transactions=[_txn("Trà sữa", 45_000, "45k"), _txn("Bánh mì", 20_000, "20k")])
    _setup(monkeypatch, [unchanged] * MAX_EDIT_ROUNDS)
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-cap"}}

    await graph.ainvoke({"user_id": "73", "user_query": "trà sữa 45k, bánh mì 20k"}, config=config)
    for _ in range(MAX_EDIT_ROUNDS):
        await graph.ainvoke(Command(resume="sửa đi"), config=config)
    result = await graph.ainvoke(Command(resume="sửa nữa"), config=config)

    assert "__interrupt__" not in result
    assert "gõ lại" in result["reply"]
    assert await _amounts(db_session) == []


# --- native nesting inside a checkpointed parent ---------------------------


class _ParentState(TypedDict, total=False):
    reply: str


def _first_pending_interrupt(snapshot):
    # Same rule as app/bot/handlers.py uses to pick the live interrupt.
    for task in snapshot.tasks:
        if task.result is None and task.interrupts:
            return task.interrupts[0]
    return None


async def test_agent_pauses_repeatedly_inside_parent_graph(monkeypatch, db_session, patched_io):
    """FinancialAgent.run runs as a native subgraph of the caller's graph,
    so confirm -> correction -> confirm pauses twice in one parent node, and
    each pause is resumable by id the way bot/handlers.py does it."""
    _setup(monkeypatch, [ParseResult(transactions=[_txn("Trà sữa", 45_000, "45k"), _txn("Bánh mì", 25_000, "25k")])])

    async def _dispatch(state):
        output = await FinancialAgent().run(
            {"user_id": "74", "message": "trà sữa 45k, bánh mì 20k", "retrieved_memories": []}
        )
        return {"reply": output["reply"]}

    builder = StateGraph(_ParentState)
    builder.add_node("dispatch", _dispatch)
    builder.set_entry_point("dispatch")
    builder.add_edge("dispatch", END)
    parent = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "74"}}

    await parent.ainvoke({}, config=config)
    first = _first_pending_interrupt(await parent.aget_state(config))
    assert "Xác nhận" in first.value["question"]

    await parent.ainvoke(Command(resume={first.id: "món 2 là 25k"}), config=config)
    second = _first_pending_interrupt(await parent.aget_state(config))
    assert second is not None and second.id != first.id
    assert "25.000đ" in second.value["question"]

    result = await parent.ainvoke(Command(resume={second.id: "confirm"}), config=config)
    assert "Đã ghi 2 giao dịch." in result["reply"]
    assert _first_pending_interrupt(await parent.aget_state(config)) is None
    assert await _amounts(db_session) == [25_000, 45_000]


async def test_typed_ok_at_confirm_persists(monkeypatch, db_session, patched_io):
    _setup(monkeypatch, [])
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-ok"}}

    await graph.ainvoke({"user_id": "75", "user_query": "trà sữa 45k, bánh mì 20k"}, config=config)
    result = await graph.ainvoke(Command(resume="ok"), config=config)

    assert "Đã ghi 2 giao dịch." in result["reply"]
