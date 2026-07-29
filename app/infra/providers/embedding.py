import httpx

from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when the Ollama embedding API is unreachable or returns an error."""


_client: httpx.AsyncClient | None = None
_model: str = ""


async def init_embeddings(settings: Settings | None = None) -> None:
    global _client, _model

    if _client is not None:
        return

    cfg = settings or get_settings()
    client = httpx.AsyncClient(base_url=cfg.ollama_base_url, timeout=30.0)

    # Fail fast at startup if Ollama isn't reachable, same as the other infra clients.
    response = await client.get("/api/tags")
    response.raise_for_status()

    _client = client
    _model = cfg.embedding_model
    logger.info("embedding_client_initialized", base_url=cfg.ollama_base_url, model=_model)


async def close_embeddings() -> None:
    global _client

    if _client is None:
        return

    await _client.aclose()
    _client = None
    logger.info("embedding_client_closed")


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("Embedding client not initialized — call init_embeddings() first")
    return _client


async def embed_texts(texts: list[str], *, prefix: str = "") -> list[list[float]]:
    """Embed a batch of texts via Ollama's /api/embed endpoint."""
    if not texts:
        return []

    client = _get_client()
    payload_inputs = [f"{prefix}{text}" for text in texts] if prefix else texts
    try:
        response = await client.post("/api/embed", json={"model": _model, "input": payload_inputs})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("embedding_request_failed", error=str(exc))
        raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise EmbeddingError(f"Ollama embedding response missing 'embeddings': {data}")
    return embeddings


async def embed_text(text: str, *, prefix: str = "") -> list[float]:
    """Embed a single text string."""
    embeddings = await embed_texts([text], prefix=prefix)
    return embeddings[0]
