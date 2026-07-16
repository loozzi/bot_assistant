from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config.settings import Settings, get_settings
from app.infra.memory.working import get_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_saver: AsyncRedisSaver | None = None


async def init_checkpointer(settings: Settings | None = None) -> None:
    global _saver

    if _saver is not None:
        return

    cfg = settings or get_settings()
    saver = AsyncRedisSaver(
        redis_client=get_redis_client(),
        ttl={"default_ttl": cfg.redis_ttl_session // 60, "refresh_on_read": True},
    )
    await saver.asetup()
    _saver = saver
    logger.info("checkpointer_initialized")


async def close_checkpointer() -> None:
    global _saver

    if _saver is None:
        return

    _saver = None
    logger.info("checkpointer_closed")


def get_checkpointer() -> AsyncRedisSaver:
    if _saver is None:
        raise RuntimeError("Checkpointer not initialized — call init_checkpointer() first")
    return _saver
