from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client: AsyncQdrantClient | None = None


async def init_vector_db(settings: Settings | None = None) -> None:
    global _client

    if _client is not None:
        return

    cfg = settings or get_settings()

    _client = AsyncQdrantClient(
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
        grpc_port=cfg.qdrant_grpc_port,
        prefer_grpc=cfg.qdrant_prefer_grpc,
        api_key=cfg.qdrant_api_key or None,
        timeout=cfg.qdrant_timeout,
    )

    await _ensure_collection(cfg)
    logger.info("qdrant_client_initialized", host=cfg.qdrant_host, port=cfg.qdrant_port)


async def _ensure_collection(cfg: Settings) -> None:
    assert _client is not None

    existing = {c.name for c in (await _client.get_collections()).collections}
    if cfg.qdrant_collection in existing:
        return

    await _client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config=VectorParams(size=cfg.embedding_dim, distance=Distance.COSINE),
    )
    logger.info("qdrant_collection_created", collection=cfg.qdrant_collection)


async def close_vector_db() -> None:
    global _client

    if _client is None:
        return

    await _client.close()
    _client = None
    logger.info("qdrant_client_closed")


def get_qdrant_client() -> AsyncQdrantClient:
    if _client is None:
        raise RuntimeError("Qdrant not initialized — call init_vector_db() first")
    return _client
