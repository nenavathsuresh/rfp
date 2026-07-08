"""Query and fetch helpers for the local vector RAG index."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class DocumentRetrievalError(RuntimeError):
    """Raised when the local RAG index cannot be loaded or queried."""


@dataclass(frozen=True)
class RetrievedDocument:
    """A normalized search result returned from the knowledge base."""

    content: str
    score: float | None
    document_name: str | None
    source: str | None
    page: int | None

    def as_citation(self) -> str:
        """Format the result for use by an agent tool."""

        parts = []
        if self.document_name:
            parts.append(self.document_name)
        if self.page is not None:
            parts.append(f"page {self.page + 1}")

        citation = ", ".join(parts) if parts else self.source or "knowledge base"
        score_text = f" score={self.score:.4f}" if self.score is not None else ""
        return f"[{citation}{score_text}]\n{self.content}"


def load_vector_index(
    index_path: str | Path,
    embedding_generator: Embeddings,
) -> Any:
    """Load the LangChain vector index created by the indexing pipeline."""

    path = Path(index_path)
    if not path.exists():
        raise DocumentRetrievalError(
            f"Vector index not found at '{path}'. Run app/create_vectordb.py first."
        )

    return FAISS.load_local(
        str(path),
        embedding_generator,
        allow_dangerous_deserialization=True,
    )


def query_fetch(
    query: str,
    vectorstore: Any,
    *,
    top_k: int = 3,
) -> list[RetrievedDocument]:
    """Fetch relevant chunks from a vector store for a natural-language query."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise DocumentRetrievalError("Query cannot be empty.")

    results = vectorstore.similarity_search_with_score(cleaned_query, k=top_k)

    fetched: list[RetrievedDocument] = []
    for document, score in results:
        metadata: dict[str, Any] = document.metadata or {}
        fetched.append(
            RetrievedDocument(
                content=document.page_content,
                score=float(score) if score is not None else None,
                document_name=metadata.get("document_name"),
                source=metadata.get("source"),
                page=metadata.get("page"),
            )
        )

    return fetched


def format_retrieval_results(results: list[RetrievedDocument]) -> str:
    """Render retrieved chunks into a compact citation-first response."""

    if not results:
        return "No relevant context found in the knowledge base."

    return "\n\n".join(result.as_citation() for result in results)
