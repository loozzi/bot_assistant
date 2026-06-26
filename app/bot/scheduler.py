from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.utils.logger import get_logger

logger = get_logger(__name__)

_scheduler: AsyncIOScheduler | None = None


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

async def _decay_memory_importance() -> None:
    """Decrease importance score for Qdrant chunks not retrieved in 30 days."""
    logger.info("memory_decay_job_started")
    # TODO: implement in Phase 2 when Qdrant ingestion pipeline is ready
    logger.info("memory_decay_job_finished")


async def _compress_low_importance_chunks() -> None:
    """Compress chunks with importance < 0.2 into summaries."""
    logger.info("memory_compress_job_started")
    # TODO: implement in Phase 2
    logger.info("memory_compress_job_finished")


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
