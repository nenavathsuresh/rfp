"""Application script to index RFP documents into the vector database."""

from pathlib import Path

from config.settings import get_settings
from knowledge.indexing.kstore import (
    create_embedding_generator,
    create_document_vector_collection_from_paths,
    describe_embedding_configuration,
)
from models.rag import RagIndexingConfig
from observability import CostTracker, wrap_embeddings_with_tracking


def create_vector_db(
    upload_dir: str | Path = "uploads",
    index_path: str | Path | None = None,
):
    """Index uploaded RFP documents into a local vector store."""

    settings = get_settings()

    source_dir = Path(upload_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    paths = tuple(path for path in source_dir.iterdir() if path.is_file())
    if not paths:
        print(f"No documents found in '{source_dir}'. Please upload files and run it again.")
        return None

    target_index = Path(index_path) if index_path else Path(source_dir) / "vectorstore"

    print(f"Found {len(paths)} document(s) to index: {[p.name for p in paths]}")
    print(describe_embedding_configuration(settings))

    embedding_provider = settings.embedding_provider.strip().lower()
    embedding_model = (
        settings.openai_embedding_model_id
        if embedding_provider == "openai"
        else settings.huggingface_embedding_model_id
    ) or "unknown"

    usage_tracker = CostTracker(
        report_name="indexing_usage_report",
        run_type="indexing",
    )

    embedding_generator = create_embedding_generator(settings)
    embedding_generator = wrap_embeddings_with_tracking(
        embedding_generator,
        usage_tracker,
        provider=embedding_provider,
        model=embedding_model,
    )

    config = RagIndexingConfig(
        collection_name=str(target_index),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    print("Indexing documents into vector collection...")
    collection = create_document_vector_collection_from_paths(
        paths=paths,
        config=config,
        embedding_generator=embedding_generator,
    )
    report_path = usage_tracker.save_report()
    print(f"Usage and cost report saved to '{report_path}'")
    print(f"Vector collection initialized and indexed successfully at '{target_index}'!")
    return collection, report_path
