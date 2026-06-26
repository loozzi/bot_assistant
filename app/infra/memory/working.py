import redis.asyncio as aioredis

from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client: aioredis.Redis | None = None


async def init_redis(settings: Settings | None = None) -> None:
    global _client

    if _client is not None:
        return

    cfg = settings or get_settings()
    _client = aioredis.Redis(
        host=cfg.redis_host,
        port=cfg.redis_port,
        password=cfg.redis_password or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    await _client.ping()
    logger.info("redis_client_initialized", host=cfg.redis_host, port=cfg.redis_port)


async def close_redis() -> None:
    global _client

    if _client is None:
        return

    await _client.aclose()
    _client = None
    logger.info("redis_client_closed")


def get_redis_client() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    return _client
