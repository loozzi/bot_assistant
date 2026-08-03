import pytest

from app.modules.finance.node.classify import classify_node
from app.modules.finance.schema.classify_result import ClassifyResult
from tests.unit.modules.finance.conftest import FakeLLM


@pytest.mark.asyncio
async def test_classify_node_returns_llm_sub_intent(monkeypatch):
    monkeypatch.setattr(
        "app.modules.finance.node.classify.create_llm_client",
        lambda: FakeLLM(ClassifyResult(sub_intent="log")),
    )
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "log"}


@pytest.mark.asyncio
async def test_classify_node_falls_back_on_llm_error(monkeypatch):
    def _raise():
        raise RuntimeError("llm down")

    monkeypatch.setattr("app.modules.finance.node.classify.create_llm_client", _raise)
    result = await classify_node({"user_query": "mua trà sữa 45k", "user_id": "1"})
    assert result == {"sub_intent": "fallback"}
