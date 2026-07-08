"""Application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed environment configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_chat_model_id: str | None = "gpt-4o-mini"
    openai_embedding_model_id: str | None = "text-embedding-3-small"
    openai_base_url: str | None = None
    embedding_provider: str = "openai"
    huggingface_embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    anthropic_api_key: str | None = None
    anthropic_chat_model_id: str | None = None
    anthropic_base_url: str | None = None
    human_approval_enabled: bool = True
    rag_collection_name: str = "rfp_documents"
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 150
    Anthropic_model_use : bool = False
    OpenAI_model_use : bool = False


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings loaded from environment and `.env`."""

    return Settings()
