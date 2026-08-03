from unittest.mock import AsyncMock

import pytest

from app.modules.finance.tools.memory import write_transaction_memory


@pytest.mark.asyncio
async def test_write_transaction_memory_upserts_with_log_id(monkeypatch):
    fake_qdrant = AsyncMock()
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.get_qdrant_client", lambda: fake_qdrant
    )
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.get_settings",
        lambda: type("S", (), {"qdrant_collection": "secondbrain"})(),
    )
    monkeypatch.setattr(
        "app.modules.finance.tools.memory.embed_text", AsyncMock(return_value=[0.1, 0.2])
    )

    await write_transaction_memory(
        user_id="1",
        log_id=99,
        category_name="Ăn uống",
        description="Trà sữa",
        amount_vnd=45_000,
        log_type="expense",
        occurred_at="2026-08-03",
    )

    fake_qdrant.upsert.assert_awaited_once()
    _, kwargs = fake_qdrant.upsert.call_args
    point = kwargs["points"][0]
    assert point.payload["log_id"] == 99
    assert point.payload["user_id"] == "1"
    assert point.payload["source_type"] == "finance"
    assert point.payload["topics"] == ["Ăn uống"]


@pytest.mark.asyncio
async def test_write_transaction_memory_swallows_errors(monkeypatch):
    def _raise():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr("app.modules.finance.tools.memory.get_qdrant_client", _raise)

    # Must not raise.
    await write_transaction_memory(
        user_id="1",
        log_id=1,
        category_name="Ăn uống",
        description="x",
        amount_vnd=1000,
        log_type="expense",
        occurred_at="2026-08-03",
    )
