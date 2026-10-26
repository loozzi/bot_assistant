from datetime import date

import pytest
from pydantic import ValidationError

from app.modules.finance.schema.classify_result import ClassifyResult
from app.modules.finance.schema.parsed_transaction import ParsedTransaction, ParseResult
from app.modules.finance.schema.query_params import QueryParams


def test_classify_result_accepts_known_labels():
    assert ClassifyResult(sub_intents=["log", "query"]).sub_intents == ["log", "query"]


def test_classify_result_accepts_single_label_shape():
    assert ClassifyResult(sub_intent="log").sub_intents == ["log"]
    assert ClassifyResult.model_validate({"sub_intent": "query"}).sub_intents == ["query"]


def test_classify_result_rejects_empty_list():
    with pytest.raises(ValidationError):
        ClassifyResult(sub_intents=[])


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


def test_query_params_accepts_income_and_net():
    for metric in ("income", "net"):
        params = QueryParams(metric=metric, period_start=date(2026, 8, 1), period_end=date(2026, 8, 31))
        assert params.metric == metric
