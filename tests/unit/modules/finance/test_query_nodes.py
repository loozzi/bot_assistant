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


@pytest.mark.asyncio
async def test_run_query_node_total_unknown_category(db_session):
    user_pk = await get_or_create_user_pk(db_session, "31")
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    result = await query_run_node(
        {
            "session": db_session,
            "user_pk": user_pk,
            "query_params": {
                "metric": "total",
                "period_start": "2026-08-01",
                "period_end": "2026-08-31",
                "category": "Danh mục không tồn tại",
                "keyword": None,
            },
        }
    )
    assert "không tìm thấy" in result["reply"].lower()
    assert "cho Danh mục không tồn tại" not in result["reply"]


@pytest.mark.asyncio
async def test_run_query_node_by_category(db_session):
    user_pk = await get_or_create_user_pk(db_session, "32")
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
            },
            {
                "amount_vnd": 30_000,
                "raw_amount_text": "30k",
                "description": "Xe ôm",
                "log_type": "expense",
                "suggested_category": "Đi lại",
                "occurred_at": "2026-08-04",
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()

    result = await query_run_node(
        {
            "session": db_session,
            "user_pk": user_pk,
            "query_params": {
                "metric": "by_category",
                "period_start": "2026-08-01",
                "period_end": "2026-08-31",
                "category": None,
                "keyword": None,
            },
        }
    )
    reply = result["reply"]
    assert "Ăn uống" in reply
    assert "45.000" in reply
    assert "Đi lại" in reply
    assert "30.000" in reply


@pytest.mark.asyncio
async def test_run_query_node_search(db_session):
    user_pk = await get_or_create_user_pk(db_session, "33")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa trân châu",
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
                "metric": "search",
                "period_start": "2026-08-01",
                "period_end": "2026-08-31",
                "category": None,
                "keyword": "trà sữa",
            },
        }
    )
    reply = result["reply"]
    assert "Trà sữa trân châu" in reply
    assert "45.000" in reply


@pytest.mark.asyncio
async def test_run_query_node_search_excludes_transactions_outside_period(db_session):
    """Finding 4: a matching transaction outside the extracted period must
    not appear in the listed results NOR be included in the displayed total."""
    user_pk = await get_or_create_user_pk(db_session, "34")
    await seed_default_categories(db_session, user_pk)
    await create_transactions(
        db_session,
        user_pk,
        [
            {
                "amount_vnd": 45_000,
                "raw_amount_text": "45k",
                "description": "Trà sữa tháng 8",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-08-03",
                "amount_adjusted": False,
            },
            {
                "amount_vnd": 500_000,
                "raw_amount_text": "500k",
                "description": "Trà sữa tháng 7 (ngoài kỳ)",
                "log_type": "expense",
                "suggested_category": "Ăn uống",
                "occurred_at": "2026-07-01",
                "amount_adjusted": False,
            },
        ],
    )
    await db_session.commit()

    result = await query_run_node(
        {
            "session": db_session,
            "user_pk": user_pk,
            "query_params": {
                "metric": "search",
                "period_start": "2026-08-01",
                "period_end": "2026-08-31",
                "category": None,
                "keyword": "trà sữa",
            },
        }
    )
    reply = result["reply"]
    assert "Trà sữa tháng 8" in reply
    assert "ngoài kỳ" not in reply
    assert "45.000" in reply
    # total must be 45.000, not 545.000 (which would include the out-of-period txn)
    assert "545.000" not in reply
    assert "500.000" not in reply
