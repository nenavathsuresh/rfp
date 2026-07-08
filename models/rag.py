"""RAG configuration models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RagIndexingConfig:
    """Configuration for building a local RAG index."""

    collection_name: str = "vectorstore"
    chunk_size: int = 1200
    chunk_overlap: int = 150
