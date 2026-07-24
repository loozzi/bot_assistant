from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from qdrant_client.http.models import FieldCondition, Filter, PointStruct, Range

from app.config.settings import get_settings
from app.infra.db.vector import get_qdrant_client
from app.infra.memory.episodic import detect_language
from app.infra.providers.embedding import embed_text
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_scheduler: AsyncIOScheduler | None = None

_SCROLL_PAGE_SIZE = 100
_MAX_SCROLL_PAGES = 50  # bounds each job to at most 5000 points per run
_DECAY_WINDOW_DAYS = 30
_DECAY_FACTOR = 0.9
_COMPRESSION_IMPORTANCE_THRESHOLD = 0.2
_COMPRESSION_RESET_IMPORTANCE = 0.3
_COMPRESSION_SYSTEM_PROMPT = (
    "Summarize the following related notes into a single short paragraph "
    "that preserves the key facts. Reply in the same language as the notes."
)


async def _scroll_all(collection: str, scroll_filter=None):
    """Yield up to _MAX_SCROLL_PAGES pages of points from `collection`."""
    qdrant = get_qdrant_client()
    offset = None
    for _ in range(_MAX_SCROLL_PAGES):
        points, offset = await qdrant.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
        )
        yield points
        if offset is None:
            break


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

async def _decay_memory_importance() -> None:
    """Decrease importance by 10% for memories unretrieved for 30+ days."""
    logger.info("memory_decay_job_started")
    try:
        qdrant = get_qdrant_client()
        collection = get_settings().qdrant_collection
        cutoff = datetime.now(timezone.utc) - timedelta(days=_DECAY_WINDOW_DAYS)

        decayed = 0
        async for points in _scroll_all(collection):
            for point in points:
                last_seen_raw = point.payload.get("last_accessed") or point.payload.get("timestamp")
                if not last_seen_raw:
                    continue

                last_seen = datetime.fromisoformat(last_seen_raw)
                if last_seen > cutoff:
                    continue

                importance = point.payload.get("importance", 0.5)
                new_importance = max(0.0, importance * _DECAY_FACTOR)
                # Push last_accessed forward by one window so a still-unretrieved
                # memory decays again next window, not on every subsequent run.
                new_last_accessed = last_seen + timedelta(days=_DECAY_WINDOW_DAYS)

                await qdrant.set_payload(
                    collection_name=collection,
                    payload={
                        "importance": new_importance,
                        "last_accessed": new_last_accessed.isoformat(),
                    },
                    points=[point.id],
                )
                decayed += 1

        logger.info("memory_decay_job_finished", decayed=decayed)
    except Exception as exc:
        logger.warning("memory_decay_job_failed", error=str(exc))


async def _summarize_for_compression(combined_text: str) -> str:
    llm = create_llm_client()
    response = await llm.ainvoke(
        [
            {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": combined_text},
        ]
    )
    return response.content


async def _compress_low_importance_chunks() -> None:
    """Merge low-importance memories (per user + source_type) into one summary point."""
    logger.info("memory_compress_job_started")
    try:
        qdrant = get_qdrant_client()
        collection = get_settings().qdrant_collection

        low_importance_filter = Filter(
            must=[FieldCondition(key="importance", range=Range(lt=_COMPRESSION_IMPORTANCE_THRESHOLD))]
        )

        groups: dict[tuple[str, str], list] = defaultdict(list)
        async for points in _scroll_all(collection, scroll_filter=low_importance_filter):
            for point in points:
                key = (point.payload.get("user_id"), point.payload.get("source_type"))
                groups[key].append(point)

        compressed_groups = 0
        for (user_id, source_type), points in groups.items():
            if len(points) < 2:
                continue

            combined_text = "\n".join(p.payload.get("text", p.payload.get("summary_snippet", "")) for p in points)
            summary = await _summarize_for_compression(combined_text)
            embedding = await embed_text(summary)
            timestamp = datetime.now(timezone.utc).isoformat()

            await qdrant.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload={
                            "user_id": user_id,
                            "source_type": source_type,
                            "timestamp": timestamp,
                            "last_accessed": timestamp,
                            "importance": _COMPRESSION_RESET_IMPORTANCE,
                            "language": detect_language(summary),
                            "topics": [],
                            "query": "",
                            "summary_snippet": summary[:200],
                            "text": summary,
                            "compressed_from": len(points),
                        },
                    )
                ],
            )
            await qdrant.delete(collection_name=collection, points_selector=[p.id for p in points])
            compressed_groups += 1

        logger.info("memory_compress_job_finished", groups_compressed=compressed_groups)
    except Exception as exc:
        logger.warning("memory_compress_job_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def get_scheduler() -> AsyncIOScheduler:
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized — call init_scheduler() first")
    return _scheduler


def init_scheduler() -> AsyncIOScheduler:
    global _scheduler

    if _scheduler is not None:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")

    # Memory decay — runs every day at 03:00
    _scheduler.add_job(
        _decay_memory_importance,
        trigger=CronTrigger(hour=3, minute=0),
        id="memory_decay",
        replace_existing=True,
    )

    # Memory compression — runs every Sunday at 04:00
    _scheduler.add_job(
        _compress_low_importance_chunks,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="memory_compress",
        replace_existing=True,
    )

    logger.info("scheduler_initialized", jobs=len(_scheduler.get_jobs()))
    return _scheduler
