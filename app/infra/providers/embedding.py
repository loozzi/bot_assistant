import asyncio
import functools

from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    logger.info(
        "embedding_model_loading",
        model=settings.embedding_model,
        device=settings.embedding_device,
    )
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info("embedding_model_loaded", model=settings.embedding_model)
    return model


async def embed_text(text: str, *, prefix: str) -> list[float]:
    """Embed `prefix + text` with the configured sentence-transformers model.

    `prefix` must be "passage: " for content being stored and "query: " for
    a recall query, per the e5 model's asymmetric training convention.
    """
    model = _load_model()
    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(None, model.encode, prefix + text)
    return vector.tolist()
