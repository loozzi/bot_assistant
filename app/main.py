import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.bot.handlers import router as main_router
from app.bot.middlewares import LoggingMiddleware, RateLimitMiddleware, UserContextMiddleware
from app.bot.scheduler import init_scheduler
from app.config.settings import get_settings
from app.config.logging import setup_logging
from app.infra.db.session import close_db, init_db
from app.infra.db.vector import close_vector_db, init_vector_db
from app.infra.memory.checkpointer import close_checkpointer, init_checkpointer
from app.infra.memory.working import close_redis, init_redis
from app.infra.providers.embedding import close_embeddings, init_embeddings
from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def on_startup(bot: Bot) -> None:
    settings = get_settings()

    await init_db(settings)
    await init_redis(settings)
    await init_checkpointer(settings)
    await init_vector_db(settings)
    await init_embeddings(settings)

    registry.discover_and_register()

    # Warm-up the compiled graph so the first request isn't slow
    get_compiled_graph()

    scheduler = init_scheduler()
    scheduler.start()

    me = await bot.get_me()
    logger.info("bot_started", username=me.username)


async def on_shutdown(bot: Bot) -> None:
    from app.bot.scheduler import get_scheduler

    try:
        get_scheduler().shutdown(wait=False)
    except RuntimeError:
        pass

    await close_checkpointer()
    await close_redis()
    await close_vector_db()
    await close_embeddings()
    await close_db()

    logger.info("bot_stopped")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()

    # Middlewares — order matters: logging → user context → rate limit
    dp.update.outer_middleware(LoggingMiddleware())
    dp.message.middleware(UserContextMiddleware())
    dp.message.middleware(RateLimitMiddleware())
    dp.callback_query.middleware(UserContextMiddleware())

    dp.include_router(main_router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    logger.info("starting_polling")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
