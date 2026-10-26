from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.modules.finance.agent import build_graph
from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from app.modules.finance.schema.query_params import QueryParams
from app.modules.finance.tools.dates import today_ict
from tests.unit.modules.finance.conftest import FakeLLM


def _patch(monkeypatch, db_session, intents: list[str]):
    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)
    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", AsyncMock())
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intents=intents)),
    )


async def test_log_then_query_counts_the_new_transaction(monkeypatch, db_session):
    today = today_ict()
    _patch(monkeypatch, db_session, ["query", "log"])
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client",
        lambda: FakeLLM(
            ParseResult(
                transactions=[
                    ParsedTransaction(
                        amount_vnd=45_000,
                        raw_amount_text="45k",
                        description="Trà sữa",
                        log_type="expense",
                        suggested_category="Ăn uống",
                        occurred_at=today,
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        "app.modules.finance.node.query.extract_params.create_llm_client",
        lambda: FakeLLM(QueryParams(metric="total", period_start=today.replace(day=1), period_end=today)),
    )
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "multi-log-query"}}

    paused = await graph.ainvoke(
        {"user_id": "110", "user_query": "trà sữa 45k, tháng này tiêu bao nhiêu rồi?"}, config=config
    )
    assert "__interrupt__" in paused

    result = await graph.ainvoke(Command(resume="confirm"), config=config)

    first, second = result["reply"].split("\n\n", 1)
    assert first.startswith("Đã ghi 1 giao dịch.")
    assert "45.000" in second


async def test_single_intent_reply_is_unchanged(monkeypatch, db_session):
    _patch(monkeypatch, db_session, ["budget"])
    graph = build_graph().compile(checkpointer=InMemorySaver())

    result = await graph.ainvoke(
        {"user_id": "111", "user_query": "đặt ngân sách"},
        config={"configurable": {"thread_id": "multi-single"}},
    )

    assert result["reply"].startswith("Tính năng đặt ngân sách")
    assert "\n\n" not in result["reply"]


async def test_second_intent_runs_even_if_first_ends_early(monkeypatch, db_session):
    _patch(monkeypatch, db_session, ["log", "budget"])
    monkeypatch.setattr(
        "app.modules.finance.node.log.parse.create_llm_client", lambda: FakeLLM(ParseResult())
    )
    graph = build_graph().compile(checkpointer=InMemorySaver())

    result = await graph.ainvoke(
        {"user_id": "112", "user_query": "ờ thì... với lại đặt ngân sách"},
        config={"configurable": {"thread_id": "multi-early"}},
    )

    assert "gõ lại" in result["reply"]
    assert "ngân sách" in result["reply"].split("\n\n")[1]
