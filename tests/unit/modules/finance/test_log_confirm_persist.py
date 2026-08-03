from datetime import date
from typing import TypedDict

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


@pytest.mark.asyncio
async def test_log_persist_node_reports_actual_category_not_suggested(db_session):
    user_pk = await get_or_create_user_pk(db_session, "44")
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

    result = await log_persist_node(
        {"session": db_session, "user_pk": user_pk, "parsed_transactions": transactions}
    )
    await db_session.commit()

    assert result["persisted_categories"] == ["Khác"]


class _TestGraphState(TypedDict, total=False):
    """LangGraph state schema for _build_test_graph.

    Must be a TypedDict (not bare `dict`) so each key gets its own
    LastValue channel and node returns merge key-by-key across supersteps.
    A bare `dict` has no __annotations__, so LangGraph collapses the whole
    state into a single "__root__" channel — each node's partial-dict
    return then *replaces* the entire state instead of merging into it,
    which silently drops user_id/parsed_transactions after resume. This
    mirrors why the real orchestrator state (app/core/state.py:AgentState)
    is a TypedDict rather than a plain dict.
    """

    user_id: str
    parsed_transactions: list[dict]
    confirm_answer: str
    persisted_log_ids: list[int]
    persisted_categories: list[str]


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

    builder = StateGraph(_TestGraphState)
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
