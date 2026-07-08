"""Chatbot service for answering questions against a local vector index."""

from pathlib import Path
from typing import Any

from client.client import create_claude_chat_client, create_openai_chat_client
from config.settings import get_settings
from knowledge.indexing.kstore import create_embedding_generator
from knowledge.retrieval import DocumentRetrievalError, load_vector_index
from tools.retrieval import create_document_rag_tool
from utils.helper import read_instructions


def _load_chatbot_instructions(project_root: str | Path | None = None) -> str:
    """Load chatbot instructions from YAML and include shared RAG guidance."""

    root = Path(project_root or Path.cwd())
    primary_data = read_instructions(root / "instructions" / "chatbot.yaml")
    return (
        f"{primary_data['system_instructions']}\n\n"
        f"Default user task for each run:\n{primary_data['user_prompt']}\n\n"
    )


def _extract_response_text(response: Any) -> str:
    """Extract readable text from Agent Framework response objects."""

    nested_response = getattr(response, "agent_response", None)
    if nested_response is not None and nested_response is not response:
        nested_text = _extract_response_text(nested_response)
        if nested_text:
            return nested_text

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    messages = getattr(response, "messages", None)
    if messages:
        message_text = [
            getattr(message, "text", "")
            for message in messages
            if getattr(message, "text", "")
        ]
        return "\n".join(message_text)

    return str(response) if response is not None else ""


def _create_chat_client() -> Any:
    """Create the configured chat client, preferring Anthropic when configured."""

    settings = get_settings()
    if settings.anthropic_chat_model_id:
        return create_claude_chat_client(settings)
    return create_openai_chat_client(settings)


def load_chatbot_vector_db(index_path: str | Path) -> Any:
    """Load the saved vector database with the configured embedding model."""

    embedding_generator = create_embedding_generator(get_settings())
    return load_vector_index(index_path, embedding_generator)


def create_vector_db_chatbot_agent(index_path: str | Path) -> Any:
    """Create an agent with access to the saved vector database."""

    vectorstore = load_chatbot_vector_db(index_path)
    search_tool = create_document_rag_tool(vectorstore)
    client = _create_chat_client()
    return client.as_agent(
        name="RFP Chatbot",
        instructions=_load_chatbot_instructions(),
        tools=search_tool,
    )


async def answer_query_from_vector_db(
    query: str,
    index_path: str | Path,
    *,
    top_k: int = 3,
) -> str:
    """Answer a user query using an agent backed by the saved vector database."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise DocumentRetrievalError("Query cannot be empty.")

    agent = create_vector_db_chatbot_agent(index_path)
    prompt = (
        f"Question: {cleaned_query}\n\n"
        f"Retrieve up to {top_k} relevant chunks, then answer from that context."
    )
    response = await agent.run(prompt)
    return _extract_response_text(response).strip()
