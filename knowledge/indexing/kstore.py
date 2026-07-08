from pathlib import Path
from typing import Any, Optional

import pymupdf4llm
from openai import NotFoundError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import Settings
from models.rag import RagIndexingConfig


class DocumentRagConfigurationError(RuntimeError):
    """Raised when document indexing fails."""


def create_openai_embedding_generator(
    settings: Settings,
) -> OpenAIEmbeddings:
    """
    Create OpenAI embedding model.
    """

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model_id
        or "text-embedding-3-small",
        api_key=settings.openai_api_key,
    )


def create_huggingface_embedding_generator(
    settings: Settings,
) -> Embeddings:
    """
    Create a local Hugging Face sentence-transformer embedding model.
    """

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise DocumentRagConfigurationError(
            "Hugging Face embeddings require langchain-huggingface and "
            "sentence-transformers. Install requirements.txt, then retry."
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=settings.huggingface_embedding_model_id,
    )


def create_embedding_generator(settings: Settings) -> Embeddings:
    """Create the configured embedding model."""

    provider = settings.embedding_provider.strip().lower()
    if provider in {"huggingface", "hf", "sentence-transformers"}:
        return create_huggingface_embedding_generator(settings)
    if provider == "openai":
        return create_openai_embedding_generator(settings)

    raise DocumentRagConfigurationError(
        "Unsupported EMBEDDING_PROVIDER. Use 'openai' or 'huggingface'."
    )


def describe_embedding_configuration(settings: Settings) -> str:
    """Return a non-secret summary of the embedding provider settings."""

    provider = settings.embedding_provider.strip().lower()
    if provider in {"huggingface", "hf", "sentence-transformers"}:
        return (
            "Hugging Face embeddings "
            f"model={settings.huggingface_embedding_model_id}"
        )

    return f"OpenAI embeddings model={settings.openai_embedding_model_id}"


def extract_pdf_pages(pdf_path: Path) -> list[Document]:
    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        show_progress=False,
    )
    output_dir = pdf_path.parent / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save all pages into a single Markdown file
    md_path = output_dir / f"{pdf_path.stem}.md"

    with md_path.open("w", encoding="utf-8") as f:
        for index, page in enumerate(pages):
            text = page.get("text", "").strip()
            if text:
                f.write(f"# Page {index + 1}\n\n")
                f.write(text)
                f.write("\n\n---\n\n")
    print(f"Document data is extracted and also save sucessfully @{md_path}" )
    return [
        Document(
            page_content=page.get("text", "").strip(),
            metadata={"page": index + 1},
        )
        for index, page in enumerate(pages)
        if page.get("text", "").strip()
    ]


def load_documents(
    paths: tuple[str | Path, ...],
) -> list[Document]:
    """
    Load PDF/TXT/MD documents.
    """

    documents: list[Document] = []

    for p in paths:

        path = Path(p)

        if not path.exists():
            continue

        try:

            print(f"\nLoading: {path.name}")

            # -------------------------
            # PDF Files
            # -------------------------
            if path.suffix.lower() == ".pdf":

                pages = extract_pdf_pages(path)

                print(
                    f"Loaded {len(pages)} page(s) "
                    f"from {path.name}"
                )

                for page in pages:

                    page.metadata.update(
                        {
                            "document_name": path.name,
                            "source": str(path.absolute()),
                        }
                    )

                documents.extend(pages)

            # -------------------------
            # Text Files
            # -------------------------
            else:

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if not text.strip():
                    print(
                        f"Skipping empty file: {path.name}"
                    )
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "document_name": path.name,
                            "source": str(path.absolute()),
                        },
                    )
                )

        except Exception as ex:

            print(
                f"Failed loading {path.name}: {ex}"
            )

    print(
        f"\nTotal loaded documents/pages: "
        f"{len(documents)}"
    )

    return documents


def create_document_vector_collection(
    documents: list[Document],
    embedding_generator: Embeddings,
    chunk_size: int = 100,
    chunk_overlap: int = 50,
    index_path: str = "vectorstore",
) -> Any:
    """
    Chunk documents and create a vector index.
    """

    if not documents:

        raise DocumentRagConfigurationError(
            "No documents loaded."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("\nChunking documents...")

    chunks = splitter.split_documents(documents)

    print(
        f"Created {len(chunks)} chunk(s)"
    )

    if not chunks:

        raise DocumentRagConfigurationError(
            "No chunks generated."
        )

    print(
        "\nGenerating embeddings and "
        "building vector index..."
    )

    try:
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_generator,
        )
    except NotFoundError as exc:
        raise DocumentRagConfigurationError(
            "OpenAI embedding model was not found. Check OPENAI_EMBEDDING_MODEL_ID "
            "and make sure OPENAI_API_KEY is a valid OpenAI API key."
        ) from exc

    vectorstore.save_local(index_path)

    print(
        f"\nVector index saved to: {index_path}"
    )

    return vectorstore


def create_document_vector_collection_from_paths(
    paths: tuple[str | Path, ...],
    *,
    config: Optional[RagIndexingConfig] = None,
    embedding_generator: Optional[
        Embeddings
    ] = None,
) -> Any:
    """
    Load files -> chunk -> embed -> vector index.
    """

    effective_config = (
        config or RagIndexingConfig()
    )

    documents = load_documents(paths)

    return create_document_vector_collection(
        documents,
        embedding_generator=embedding_generator,
        chunk_size=effective_config.chunk_size,
        chunk_overlap=effective_config.chunk_overlap,
        index_path=effective_config.collection_name,
    )
