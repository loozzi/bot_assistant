import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject, Update

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RATE_LIMIT_REQUESTS = 30    # max requests
_RATE_LIMIT_WINDOW = 60      # per N seconds


class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = None
        if isinstance(event, Update):
            msg = event.message or event.callback_query
            if msg:
                user = getattr(msg, "from_user", None)

        user_id = user.id if user else "unknown"
        logger.info("telegram_update", user_id=user_id, update_type=type(event).__name__)

        result = await handler(event, data)
        return result


class UserContextMiddleware(BaseMiddleware):
    """Inject `user_id` (str) into handler data from the Telegram user object."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        from_user = data.get("event_from_user")
        if from_user:
            data["user_id"] = str(from_user.id)
        else:
            data["user_id"] = "anonymous"

        return await handler(event, data)


class RateLimitMiddleware(BaseMiddleware):
    """Sliding-window rate limiter backed by Redis.

    Falls back gracefully if Redis is unavailable — requests are allowed through
    so the bot stays alive even if cache is down.
    """

    def __init__(
        self,
        requests: int = _RATE_LIMIT_REQUESTS,
        window: int = _RATE_LIMIT_WINDOW,
    ) -> None:
        self.requests = requests
        self.window = window

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user_id: str = data.get("user_id", "anonymous")

        try:
            from app.infra.memory.working import get_redis_client
            redis = get_redis_client()

            key = f"rate:{user_id}"
            now = time.time()
            window_start = now - self.window

            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window)
            results = await pipe.execute()

            count: int = results[2]

            if count > self.requests:
                logger.warning("rate_limit_exceeded", user_id=user_id, count=count)
                if isinstance(event, Message):
                    await event.answer(
                        "⚠️ Bạn đang gửi quá nhanh. Vui lòng thử lại sau ít giây."
                    )
                return None

        except Exception as exc:
            logger.warning("rate_limit_redis_error", error=str(exc))

        return await handler(event, data)
