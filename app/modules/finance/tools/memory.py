from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client.http.models import PointStruct

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client
from app.infra.memory.episodic import detect_language
from app.infra.providers.embedding import embed_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_IMPORTANCE = 0.6


def _render_sentence(description: str, amount_vnd: int, category_name: str, occurred_at: str) -> str:
    amount = f"{amount_vnd:,}".replace(",", ".")
    return f"{description} {amount}đ ({category_name}) — {occurred_at}"


async def write_transaction_memory(
    user_id: str,
    log_id: int,
    category_name: str,
    description: str,
    amount_vnd: int,
    log_type: str,
    occurred_at: str,
) -> None:
    """Write one structured, embeddable point per transaction to Qdrant.

    Bypasses app.infra.memory.episodic's regex-based amount/category
    extraction — finance already knows both exactly. Never raises: a
    Qdrant/embedding outage should not block the user's transaction from
    being confirmed as saved (Postgres already has it).
    """
    sentence = _render_sentence(description, amount_vnd, category_name, occurred_at)

    try:
        embedding = await embed_text(sentence)
        qdrant = get_qdrant_client()
        collection = get_settings().qdrant_collection
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "user_id": user_id,
            "source_type": "finance",
            "timestamp": timestamp,
            "last_accessed": timestamp,
            "importance": _DEFAULT_IMPORTANCE,
            "language": detect_language(sentence),
            "topics": [category_name],
            "text": sentence,
            "summary_snippet": sentence[:200],
            "log_id": log_id,
            "amount_vnd": amount_vnd,
            "log_type": log_type,
        }

        await qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=str(uuid4()), vector=embedding, payload=payload)],
        )
    except Exception as exc:
        logger.warning("finance_memory_write_failed", log_id=log_id, error=str(exc))
