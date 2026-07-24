import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client
from app.infra.memory.working import get_redis_client
from app.infra.providers.embedding import embed_text
from app.utils.chunking import chunk_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

_QUERY_CACHE_TTL = 3600
_RETRIEVAL_LIMIT_PER_TYPE = 3

_VIETNAMESE_CHARS = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_MOOD_KEYWORDS = {"buồn": 0.2, "tệ": 0.2, "mệt": 0.3, "vui": 0.8, "hạnh phúc": 0.9, "bình thường": 0.5}
_FINANCE_CATEGORIES = ["food", "transport", "shopping", "utility", "ăn uống", "di chuyển", "mua sắm"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(prefix: str, user_id: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{user_id}:{digest}"


def detect_language(text: str) -> str:
    return "vi" if any(c in text.lower() for c in _VIETNAMESE_CHARS) else "en"


def _extract_mood_score(text: str) -> float:
    lowered = text.lower()
    for mood, score in _MOOD_KEYWORDS.items():
        if mood in lowered:
            return score
    return 0.5


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)]+", text)


def _extract_category(text: str) -> str:
    lowered = text.lower()
    for category in _FINANCE_CATEGORIES:
        if category in lowered:
            return category
    return "other"


def _extract_amount(text: str) -> float:
    match = re.search(r"[\d.,]+\s*[kK]?", text)
    if not match:
        return 0.0
    raw = match.group(0).strip().lower().replace(",", "").replace(".", "")
    if raw.endswith("k"):
        raw = raw[:-1] + "000"
    try:
        return float(raw)
    except ValueError:
        return 0.0


async def retrieve_memories(user_id: str, query: str, source_types: set[str]) -> list[dict]:
    """Retrieve relevant past memories for `query`, scoped to `user_id`.

    Tries an exact-match Redis cache first, then falls back to a per-source-type
    dense Qdrant search. Returns [] (never raises) so a memory-backend outage
    degrades to "no context" rather than failing the request.
    """
    if not query or not source_types:
        return []

    cache_key = _cache_key("query", user_id, query)
    try:
        redis = get_redis_client()
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as exc:
        logger.debug("memory_cache_read_failed", error=str(exc))

    query_vector = await embed_text(query)
    qdrant = get_qdrant_client()
    collection = get_settings().qdrant_collection

    retrieved: list[dict] = []
    for source_type in source_types:
        query_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="source_type", match=MatchValue(value=source_type)),
            ]
        )
        response = await qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=_RETRIEVAL_LIMIT_PER_TYPE,
            with_payload=True,
        )
        for point in response.points:
            retrieved.append({"source_type": source_type, "payload": point.payload, "score": point.score})
            await _touch_last_accessed(qdrant, collection, point.id)

    try:
        redis = get_redis_client()
        await redis.setex(cache_key, _QUERY_CACHE_TTL, json.dumps(retrieved))
    except Exception as exc:
        logger.debug("memory_cache_write_failed", error=str(exc))

    return retrieved


async def _touch_last_accessed(qdrant, collection: str, point_id) -> None:
    """Best-effort bump of last_accessed so decay only hits truly unretrieved memories."""
    try:
        await qdrant.set_payload(
            collection_name=collection,
            payload={"last_accessed": _now_iso()},
            points=[point_id],
        )
    except Exception as exc:
        logger.debug("memory_touch_failed", point_id=str(point_id), error=str(exc))


async def write_episodic_memory(
    user_id: str,
    query: str,
    intent: str,
    reply: str,
    source_type: str,
    default_importance: float,
) -> None:
    """Chunk `reply`, embed each chunk, and upsert it into Qdrant. Never raises."""
    if not reply:
        return

    qdrant = get_qdrant_client()
    collection = get_settings().qdrant_collection
    timestamp = _now_iso()

    for chunk in chunk_text(reply):
        embedding = await embed_text(chunk)

        payload = {
            "user_id": user_id,
            "source_type": source_type,
            "timestamp": timestamp,
            "last_accessed": timestamp,
            "importance": default_importance,
            "language": detect_language(chunk),
            "topics": [],
            "query": query,
            "summary_snippet": chunk[:200],
            "text": chunk,
        }

        if intent == "journal":
            payload["mood_score"] = _extract_mood_score(chunk)
        elif intent == "search":
            payload["urls"] = _extract_urls(chunk)
        elif intent == "finance":
            payload["category"] = _extract_category(chunk)
            payload["amount"] = _extract_amount(chunk)

        await qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=str(uuid4()), vector=embedding, payload=payload)],
        )
