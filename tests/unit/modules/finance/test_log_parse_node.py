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
