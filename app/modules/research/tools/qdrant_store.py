import uuid
from datetime import datetime, timezone

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client

SOURCE_TYPE = "research"


async def upsert_research_point(
    *,
    user_id: str,
    url: str,
    title: str,
    summary: str,
    topics: list[str],
    importance: float,
    vector: list[float],
) -> None:
    """Store one research item as a Qdrant point, scoped to user_id + source_type."""
    client = get_qdrant_client()
    settings = get_settings()

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "user_id": user_id,
            "source_type": SOURCE_TYPE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance": importance,
            "topics": topics,
            "url": url,
            "title": title,
            "summary": summary,
        },
    )
    await client.upsert(collection_name=settings.qdrant_collection, points=[point])


async def search_research_points(
    *,
    user_id: str,
    query_vector: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Semantic search over this user's saved research items."""
    client = get_qdrant_client()
    settings = get_settings()

    results = await client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="source_type", match=MatchValue(value=SOURCE_TYPE)),
            ]
        ),
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {
            "title": r.payload.get("title", ""),
            "summary": r.payload.get("summary", ""),
            "url": r.payload.get("url", ""),
            "timestamp": r.payload.get("timestamp", ""),
            "score": r.score,
        }
        for r in results
    ]
