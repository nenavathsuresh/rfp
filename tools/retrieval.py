"""Retrieval tool definition for Microsoft Agent Framework agents."""

from typing import Any, Callable
from knowledge.retrieval import format_retrieval_results, query_fetch


def create_document_rag_tool(vectorstore: Any) -> Callable[[str, int], str]:
    """Create an agent-callable tool for querying the local vector knowledge base."""

    def search_rfp_documents(query: str, top: int = 3) -> str:
        """Search indexed RFP documents and return cited supporting context."""

        results = query_fetch(query=query, vectorstore=vectorstore, top_k=8)
        print("\n\n================ Retrieved RAG Chunks ================")
        print(f"Query: {query}")
        for index, result in enumerate(results, start=1):
            print(f"\n[Chunk {index}]")
            print(result.as_citation())
        print("======================================================\n")
        return format_retrieval_results(results)

    return search_rfp_documents
