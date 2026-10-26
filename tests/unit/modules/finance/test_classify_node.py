import pytest

from app.modules.finance.node.classify import classify_node, order_intents
from app.modules.finance.schema.classify_result import ClassifyResult
from tests.unit.modules.finance.conftest import FakeLLM


@pytest.mark.asyncio
async def test_classify_node_returns_llm_sub_intent(monkeypatch):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="log")),
    )
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "log", "pending_intents": []}


@pytest.mark.asyncio
async def test_classify_node_falls_back_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("llm down")

    monkeypatch.setattr("app.modules.finance.node.classify.create_llm_client", _raise)
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "fallback", "pending_intents": []}


@pytest.mark.asyncio
async def test_classify_node_queues_extra_intents_writes_first(monkeypatch):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intents=["query", "log"])),
    )
    result = await classify_node({"user_query": "trà sữa 45k, tháng này tiêu bao nhiêu?", "user_id": "1"})
    assert result == {"sub_intent": "log", "pending_intents": ["query"]}


def test_order_intents_dedupes_drops_fallback_and_caps():
    assert order_intents(["query", "query", "fallback", "log"]) == ["log", "query"]
    assert order_intents(["fallback"]) == ["fallback"]
    assert len(order_intents(["query", "log", "modify", "category", "budget"])) == 3
