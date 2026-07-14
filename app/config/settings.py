from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str

    # LLM
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Embedding
    embedding_model: str = "multilingual-e5-large"
    embedding_device: str = "cpu"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "secondbrain"
    postgres_user: str
    postgres_password: str

    # PostgreSQL pool tuning
    postgres_pool_size: int = Field(default=10, ge=1, le=100)
    postgres_max_overflow: int = Field(default=20, ge=0, le=100)
    postgres_pool_timeout: int = Field(default=30, ge=1)
    postgres_pool_recycle: int = Field(default=1800, ge=60)  # seconds

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_ttl_session: int = 14400

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "secondbrain"

    # Qdrant pool tuning
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_timeout: float = 30.0

    # App
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        """Sync DSN for Alembic migrations only — app runtime uses postgres_dsn (asyncpg)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
