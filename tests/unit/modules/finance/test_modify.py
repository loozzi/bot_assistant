from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select

from app.models import FinanceLog
from app.modules.finance.agent import build_graph
from app.modules.finance.node.answers import is_affirmative
from app.modules.finance.node.modify.extract_params import modify_extract_params_node
from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.modify_params import ModifyChanges, ModifyParams, ModifyTarget
from app.modules.finance.tools.memory import delete_transaction_memory
from app.modules.finance.tools.transactions import (
    category_name_of,
    create_transactions,
    find_logs,
    get_log,
    update_log,
)
from app.modules.finance.tools.users import get_or_create_user_pk
from tests.unit.modules.finance.conftest import FakeLLM


def _pending(description: str, amount: int, occurred_at: str = "2026-08-03", category: str = "Ăn uống") -> dict:
    return {
        "amount_vnd": amount,
        "raw_amount_text": str(amount),
        "description": description,
        "log_type": "expense",
        "suggested_category": category,
        "occurred_at": occurred_at,
        "amount_adjusted": False,
    }


async def _seed(db_session, user_id: str, txns: list[dict]) -> int:
    user_pk = await get_or_create_user_pk(db_session, user_id)
    await create_transactions(db_session, user_pk, txns)
    await db_session.commit()
    return user_pk


# --- tools ------------------------------------------------------------------


async def test_find_logs_filters_and_orders_newest_first(db_session):
    user_pk = await _seed(
        db_session,
        "80",
        [
            _pending("Trà sữa", 45_000, "2026-08-02"),
            _pending("Bánh mì", 20_000, "2026-08-03"),
            _pending("Trà sữa trân châu", 50_000, "2026-08-03"),
        ],
    )

    by_keyword = await find_logs(db_session, user_pk, keyword="trà sữa")
    assert [log.amount for log in by_keyword] == [50_000, 45_000]

    by_date = await find_logs(db_session, user_pk, occurred_at=date(2026, 8, 3))
    assert [log.description for log in by_date] == ["Trà sữa trân châu", "Bánh mì"]

    by_amount = await find_logs(db_session, user_pk, keyword="trà", amount_vnd=45_000)
    assert [log.description for log in by_amount] == ["Trà sữa"]

    latest = await find_logs(db_session, user_pk)
    assert latest[0].description == "Trà sữa trân châu"


async def test_get_log_is_scoped_to_user(db_session):
    owner_pk = await _seed(db_session, "81", [_pending("Trà sữa", 45_000)])
    other_pk = await get_or_create_user_pk(db_session, "82")
    log_id = (await find_logs(db_session, owner_pk))[0].id

    assert await get_log(db_session, owner_pk, log_id) is not None
    assert await get_log(db_session, other_pk, log_id) is None


async def test_update_log_changes_amount_and_category(db_session):
    user_pk = await _seed(db_session, "83", [_pending("Taxi", 45_000)])
    log = (await find_logs(db_session, user_pk))[0]

    await update_log(db_session, user_pk, log, {"amount_vnd": 60_000, "raw_amount_text": "60k", "category": "đi lại"})

    assert log.amount == 60_000
    assert log.extra["raw_amount_text"] == "60k"
    assert await category_name_of(db_session, log) == "Đi lại"


async def test_update_log_unknown_category_falls_back_to_khac(db_session):
    user_pk = await _seed(db_session, "84", [_pending("Taxi", 45_000)])
    log = (await find_logs(db_session, user_pk))[0]

    await update_log(db_session, user_pk, log, {"category": "Không tồn tại"})

    assert await category_name_of(db_session, log) == "Khác"


async def test_delete_transaction_memory_filters_by_user_and_log(monkeypatch):
    fake_qdrant = AsyncMock()
    monkeypatch.setattr("app.modules.finance.tools.memory.get_qdrant_client", lambda: fake_qdrant)
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.get_settings",
        lambda: type("S", (), {"qdrant_collection": "secondbrain"})(),
    )

    await delete_transaction_memory("1", 99)

    _, kwargs = fake_qdrant.delete.call_args
    conditions = {c.key: c.match.value for c in kwargs["points_selector"].filter.must}
    assert conditions == {"user_id": "1", "source_type": "finance", "log_id": 99}


async def test_delete_transaction_memory_swallows_errors(monkeypatch):
    def _raise():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr("app.modules.finance.tools.memory.get_qdrant_client", _raise)
    await delete_transaction_memory("1", 99)


@pytest.mark.parametrize("answer", ["confirm", "ok", "Ừ", " đồng ý ", "Xác nhận."])
def test_is_affirmative_accepts_typed_yes(answer):
    assert is_affirmative(answer)


@pytest.mark.parametrize("answer", ["edit", "thôi", "món 2 là 50k", None, {"x": 1}])
def test_is_affirmative_rejects_everything_else(answer):
    assert not is_affirmative(answer)


# --- extract_params ---------------------------------------------------------


def _patch_extract(monkeypatch, params):
    monkeypatch.setattr(
        "app.modules.finance.node.modify.extract_params.create_llm_client", lambda: FakeLLM(params)
    )


async def test_extract_update_rederives_amount_from_raw_text(monkeypatch):
    _patch_extract(
        monkeypatch,
        ModifyParams(action="update", target=ModifyTarget(keyword="ăn trưa"), changes=ModifyChanges(raw_amount_text="60k")),
    )
    result = await modify_extract_params_node({"user_query": "ăn trưa là 60k"})
    assert result["modify_changes"] == {"amount_vnd": 60_000, "raw_amount_text": "60k"}
    assert result["modify_target"]["keyword"] == "ăn trưa"


async def test_extract_update_without_changes_asks_what_to_change(monkeypatch):
    _patch_extract(monkeypatch, ModifyParams(action="update", target=ModifyTarget(keyword="trà sữa")))
    result = await modify_extract_params_node({"user_query": "sửa trà sữa"})
    assert "thành gì" in result["reply"]


async def test_extract_update_rejects_implausible_amount(monkeypatch):
    _patch_extract(
        monkeypatch, ModifyParams(action="update", changes=ModifyChanges(raw_amount_text="5 tỷ tỷ"))
    )
    result = await modify_extract_params_node({"user_query": "sửa thành 5 tỷ tỷ"})
    assert "không hợp lý" in result["reply"]


async def test_extract_llm_failure_replies(monkeypatch):
    def _raise():
        raise RuntimeError("llm down")

    monkeypatch.setattr("app.modules.finance.node.modify.extract_params.create_llm_client", _raise)
    result = await modify_extract_params_node({"user_query": "xoá"})
    assert result["reply"]


# --- full graph -------------------------------------------------------------


@pytest.fixture
def memory_mocks(monkeypatch, db_session):
    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)
    mocks = {"write": AsyncMock(), "delete": AsyncMock()}
    monkeypatch.setattr("app.modules.finance.agent.write_transaction_memory", mocks["write"])
    monkeypatch.setattr("app.modules.finance.agent.delete_transaction_memory", mocks["delete"])
    return mocks


def _setup_graph(monkeypatch, params: ModifyParams):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="modify")),
    )
    _patch_extract(monkeypatch, params)
    return build_graph().compile(checkpointer=InMemorySaver())


async def _all_logs(db_session) -> list[FinanceLog]:
    db_session.expire_all()
    return list((await db_session.execute(select(FinanceLog).order_by(FinanceLog.id))).scalars())


async def test_delete_latest_needs_no_selection(monkeypatch, db_session, memory_mocks):
    await _seed(db_session, "85", [_pending("Trà sữa", 45_000), _pending("Bánh mì", 20_000)])
    graph = _setup_graph(monkeypatch, ModifyParams(action="delete"))
    config = {"configurable": {"thread_id": "modify-delete"}}

    paused = await graph.ainvoke({"user_id": "85", "user_query": "xoá cái vừa rồi"}, config=config)
    assert "Bánh mì" in paused["__interrupt__"][0].value["question"]

    result = await graph.ainvoke(Command(resume="confirm"), config=config)

    assert result["reply"].startswith("Đã xoá: Bánh mì")
    assert [log.description for log in await _all_logs(db_session)] == ["Trà sữa"]
    deleted_log_id = memory_mocks["delete"].call_args.args[1]
    assert deleted_log_id not in [log.id for log in await _all_logs(db_session)]


async def test_update_with_several_matches_asks_which_then_updates(monkeypatch, db_session, memory_mocks):
    await _seed(
        db_session, "86", [_pending("Trà sữa", 45_000, "2026-08-02"), _pending("Trà sữa", 50_000, "2026-08-03")]
    )
    graph = _setup_graph(
        monkeypatch,
        ModifyParams(action="update", target=ModifyTarget(keyword="trà sữa"), changes=ModifyChanges(raw_amount_text="40k")),
    )
    config = {"configurable": {"thread_id": "modify-update"}}

    asked = await graph.ainvoke({"user_id": "86", "user_query": "trà sữa là 40k"}, config=config)
    assert len(asked["__interrupt__"][0].value["options"]) == 2

    confirm = await graph.ainvoke(Command(resume="2"), config=config)  # second listed = older, 45k
    question = confirm["__interrupt__"][0].value["question"]
    assert "45.000đ" in question and "→" in question and "40.000đ" in question

    result = await graph.ainvoke(Command(resume="ok"), config=config)

    assert result["reply"].startswith("Đã sửa: Trà sữa — 40.000đ")
    assert sorted(log.amount for log in await _all_logs(db_session)) == [40_000, 50_000]
    memory_mocks["delete"].assert_awaited_once()
    assert memory_mocks["write"].call_args.kwargs["amount_vnd"] == 40_000


async def test_declined_confirm_keeps_log(monkeypatch, db_session, memory_mocks):
    await _seed(db_session, "87", [_pending("Trà sữa", 45_000)])
    graph = _setup_graph(monkeypatch, ModifyParams(action="delete"))
    config = {"configurable": {"thread_id": "modify-decline"}}

    await graph.ainvoke({"user_id": "87", "user_query": "xoá cái cuối"}, config=config)
    result = await graph.ainvoke(Command(resume="thôi"), config=config)

    assert result["reply"] == "Đã giữ nguyên giao dịch."
    assert len(await _all_logs(db_session)) == 1
    memory_mocks["delete"].assert_not_awaited()


async def test_no_match_replies_without_pausing(monkeypatch, db_session, memory_mocks):
    await _seed(db_session, "88", [_pending("Trà sữa", 45_000)])
    graph = _setup_graph(monkeypatch, ModifyParams(action="delete", target=ModifyTarget(keyword="phở")))
    config = {"configurable": {"thread_id": "modify-none"}}

    result = await graph.ainvoke({"user_id": "88", "user_query": "xoá phở"}, config=config)

    assert "__interrupt__" not in result
    assert "không tìm thấy" in result["reply"]


async def test_invalid_selection_ends_flow(monkeypatch, db_session, memory_mocks):
    await _seed(db_session, "89", [_pending("Trà sữa", 45_000), _pending("Trà sữa", 50_000)])
    graph = _setup_graph(monkeypatch, ModifyParams(action="delete", target=ModifyTarget(keyword="trà sữa")))
    config = {"configurable": {"thread_id": "modify-bad-pick"}}

    await graph.ainvoke({"user_id": "89", "user_query": "xoá trà sữa"}, config=config)
    result = await graph.ainvoke(Command(resume="7"), config=config)

    assert "không rõ" in result["reply"]
    assert len(await _all_logs(db_session)) == 2
