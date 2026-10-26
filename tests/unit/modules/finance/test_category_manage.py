from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select

from app.models import FinanceBudget, FinanceCategory, FinanceLog
from app.modules.finance.agent import build_graph
from app.modules.finance.node.category.extract_params import category_extract_params_node
from app.modules.finance.schema.category_params import CategoryParams
from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.tools.categories import (
    CategoryError,
    category_names,
    create_category,
    delete_category,
    find_category_by_name,
    merge_categories,
    rename_category,
    seed_default_categories,
)
from app.modules.finance.tools.transactions import create_transactions
from app.modules.finance.tools.users import get_or_create_user_pk
from tests.unit.modules.finance.conftest import FakeLLM


def _pending(description: str, amount: int, category: str) -> dict:
    return {
        "amount_vnd": amount,
        "raw_amount_text": str(amount),
        "description": description,
        "log_type": "expense",
        "suggested_category": category,
        "occurred_at": "2026-08-03",
        "amount_adjusted": False,
    }


async def _user(db_session, user_id: str) -> int:
    user_pk = await get_or_create_user_pk(db_session, user_id)
    await seed_default_categories(db_session, user_pk)
    return user_pk


async def _category_of_logs(db_session, user_pk: int) -> list[str]:
    rows = await db_session.execute(
        select(FinanceCategory.name)
        .join(FinanceLog, FinanceLog.category_id == FinanceCategory.id)
        .where(FinanceLog.user_id == user_pk)
        .order_by(FinanceLog.id)
    )
    return [name for (name,) in rows.all()]


# --- tools ------------------------------------------------------------------


async def test_create_category_rejects_duplicates_case_insensitively(db_session):
    user_pk = await _user(db_session, "90")
    await create_category(db_session, user_pk, "Cà phê", "lifestyle")

    with pytest.raises(CategoryError):
        await create_category(db_session, user_pk, "cà PHÊ")


async def test_rename_category_keeps_logs_attached(db_session):
    user_pk = await _user(db_session, "91")
    await create_transactions(db_session, user_pk, [_pending("Áo", 200_000, "Mua sắm")])

    await rename_category(db_session, user_pk, "mua sắm", "Shopping")

    assert await _category_of_logs(db_session, user_pk) == ["Shopping"]


async def test_rename_onto_existing_name_is_refused(db_session):
    user_pk = await _user(db_session, "92")
    with pytest.raises(CategoryError, match="gộp"):
        await rename_category(db_session, user_pk, "Mua sắm", "Giải trí")


async def test_fallback_category_is_protected(db_session):
    user_pk = await _user(db_session, "93")
    with pytest.raises(CategoryError):
        await rename_category(db_session, user_pk, "Khác", "Linh tinh")
    with pytest.raises(CategoryError):
        await delete_category(db_session, user_pk, "Khác")


async def test_merge_moves_logs_and_budget_then_drops_source(db_session):
    user_pk = await _user(db_session, "94")
    await create_category(db_session, user_pk, "Cà phê")
    await create_transactions(
        db_session, user_pk, [_pending("Cà phê sữa", 30_000, "Cà phê"), _pending("Phở", 50_000, "Ăn uống")]
    )
    coffee = await find_category_by_name(db_session, user_pk, "Cà phê")
    db_session.add(FinanceBudget(user_id=user_pk, category_id=coffee.id, amount=500_000))
    await db_session.flush()

    moved = await merge_categories(db_session, user_pk, "Cà phê", "Ăn uống")

    assert moved == 1
    db_session.expire_all()
    assert await _category_of_logs(db_session, user_pk) == ["Ăn uống", "Ăn uống"]
    assert await find_category_by_name(db_session, user_pk, "Cà phê") is None
    food = await find_category_by_name(db_session, user_pk, "Ăn uống")
    budget = (await db_session.execute(select(FinanceBudget))).scalar_one()
    assert budget.category_id == food.id


async def test_merge_keeps_target_budget_when_both_have_one(db_session):
    user_pk = await _user(db_session, "95")
    await create_category(db_session, user_pk, "Cà phê")
    coffee = await find_category_by_name(db_session, user_pk, "Cà phê")
    food = await find_category_by_name(db_session, user_pk, "Ăn uống")
    db_session.add_all(
        [
            FinanceBudget(user_id=user_pk, category_id=coffee.id, amount=500_000),
            FinanceBudget(user_id=user_pk, category_id=food.id, amount=3_000_000),
        ]
    )
    await db_session.flush()

    await merge_categories(db_session, user_pk, "Cà phê", "Ăn uống")

    budgets = (await db_session.execute(select(FinanceBudget))).scalars().all()
    assert [(b.category_id, b.amount) for b in budgets] == [(food.id, 3_000_000)]


async def test_delete_category_moves_logs_to_khac(db_session):
    user_pk = await _user(db_session, "96")
    await create_transactions(db_session, user_pk, [_pending("Vé xem phim", 100_000, "Giải trí")])

    await delete_category(db_session, user_pk, "Giải trí")

    db_session.expire_all()
    assert await _category_of_logs(db_session, user_pk) == ["Khác"]


def test_category_names_falls_back_to_defaults():
    assert "Ăn uống" in category_names({})
    assert category_names({"available_categories": ["Cà phê"]}) == ["Cà phê"]


# --- extract ----------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        CategoryParams(action="create"),
        CategoryParams(action="rename", name="Mua sắm"),
        CategoryParams(action="merge", name="Cà phê"),
    ],
)
async def test_extract_asks_for_missing_names(monkeypatch, params):
    monkeypatch.setattr(
        "app.modules.finance.node.category.extract_params.create_llm_client", lambda: FakeLLM(params)
    )
    result = await category_extract_params_node({"user_query": "..."})
    assert "reply" in result and "category_params" not in result


# --- full graph -------------------------------------------------------------


@pytest.fixture
def graph_for(monkeypatch, db_session):
    @asynccontextmanager
    async def _fake_session():
        yield db_session

    monkeypatch.setattr("app.modules.finance.agent.get_session", _fake_session)

    def _build(params: CategoryParams):
        monkeypatch.setattr(
            "app.modules.finance.node.classify.create_llm_client",
            lambda: FakeLLM(ClassifyResult(sub_intent="category")),
        )
        monkeypatch.setattr(
            "app.modules.finance.node.category.extract_params.create_llm_client", lambda: FakeLLM(params)
        )
        return build_graph().compile(checkpointer=InMemorySaver())

    return _build


async def test_list_categories_groups_them(graph_for):
    graph = graph_for(CategoryParams(action="list"))
    result = await graph.ainvoke(
        {"user_id": "97", "user_query": "có những danh mục nào"},
        config={"configurable": {"thread_id": "cat-list"}},
    )
    assert "Thiết yếu: " in result["reply"] and "Ăn uống" in result["reply"]


async def test_create_runs_without_confirmation(graph_for, db_session):
    graph = graph_for(CategoryParams(action="create", name="Cà phê", group="lifestyle"))
    result = await graph.ainvoke(
        {"user_id": "98", "user_query": "thêm danh mục cà phê"},
        config={"configurable": {"thread_id": "cat-create"}},
    )
    assert result["reply"] == "Đã thêm danh mục 'Cà phê'."
    user_pk = await get_or_create_user_pk(db_session, "98")
    assert (await find_category_by_name(db_session, user_pk, "cà phê")).group == "lifestyle"


async def test_merge_asks_first_and_runs_on_confirm(graph_for, db_session):
    user_pk = await _user(db_session, "99")
    await create_category(db_session, user_pk, "Cà phê")
    await db_session.commit()
    graph = graph_for(CategoryParams(action="merge", name="Cà phê", target_name="Ăn uống"))
    config = {"configurable": {"thread_id": "cat-merge"}}

    paused = await graph.ainvoke({"user_id": "99", "user_query": "gộp cà phê vào ăn uống"}, config=config)
    assert "Gộp danh mục 'Cà phê'" in paused["__interrupt__"][0].value["question"]

    result = await graph.ainvoke(Command(resume="confirm"), config=config)
    assert result["reply"].startswith("Đã gộp 'Cà phê' vào 'Ăn uống'")
    assert await find_category_by_name(db_session, user_pk, "Cà phê") is None


async def test_delete_declined_keeps_category(graph_for, db_session):
    graph = graph_for(CategoryParams(action="delete", name="Giải trí"))
    config = {"configurable": {"thread_id": "cat-delete-no"}}

    await graph.ainvoke({"user_id": "100", "user_query": "xoá danh mục giải trí"}, config=config)
    result = await graph.ainvoke(Command(resume="thôi"), config=config)

    assert result["reply"] == "Đã giữ nguyên danh mục."
    user_pk = await get_or_create_user_pk(db_session, "100")
    assert await find_category_by_name(db_session, user_pk, "Giải trí") is not None


async def test_unknown_category_error_is_reported(graph_for):
    graph = graph_for(CategoryParams(action="rename", name="Không có", target_name="Mới"))
    result = await graph.ainvoke(
        {"user_id": "101", "user_query": "đổi tên không có thành mới"},
        config={"configurable": {"thread_id": "cat-missing"}},
    )
    assert result["reply"] == "Không có danh mục 'Không có'."


async def test_log_parse_prompt_sees_custom_categories(monkeypatch, graph_for, db_session):
    """load_categories feeds the user's own categories into log_parse."""
    user_pk = await _user(db_session, "102")
    await create_category(db_session, user_pk, "Cà phê")
    await db_session.commit()

    seen: list = []

    class _Recorder:
        def with_structured_output(self, schema, method=None):
            return self

        async def ainvoke(self, messages):
            seen.append(messages[1]["content"])
            from app.modules.finance.schema.parsed_transaction import ParseResult

            return ParseResult()

    graph_for(CategoryParams(action="list"))  # installs the session patch
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client", lambda: FakeLLM(ClassifyResult(sub_intent="log"))
    )
    monkeypatch.setattr("app.modules.finance.node.log.parse.create_llm_client", lambda: _Recorder())
    graph = build_graph().compile(checkpointer=InMemorySaver())

    await graph.ainvoke(
        {"user_id": "102", "user_query": "cà phê 30k"}, config={"configurable": {"thread_id": "cat-parse"}}
    )
    assert "Cà phê" in seen[0]
