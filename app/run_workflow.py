"""Application script to run the Stage 1 Opportunity Analysis workflow."""

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from config.settings import get_settings
from knowledge.indexing.kstore import create_embedding_generator
from knowledge.retrieval import load_vector_index
from observability import CostTracker, wrap_embeddings_with_tracking
from tools.retrieval import create_document_rag_tool
from utils.helper import read_instructions
from workflow.orchestration import run_opportunity_analysis


def load_workflow_prompt(project_root: Path | None = None) -> str:
    """Load the generic workflow prompt passed into the RFP workflow."""
    root = project_root or Path.cwd()
    prompt_data = read_instructions(root / "instructions" / "workflow.yaml")
    return prompt_data["user_prompt"]


async def run_analysis_workflow(
    index_path: str | Path | None = None,
    markdown_data_path: str | Path | None = None,
    emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    human_response_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Load a vector index and run the RFP analysis workflow."""

    settings = get_settings()

    target_index_path = Path(index_path) if index_path else None
    if target_index_path and not target_index_path.exists():
        raise FileNotFoundError(
            f"No vector index found at '{target_index_path}'. Please create a vector database first."
        )

    embedding_provider = settings.embedding_provider.strip().lower()
    embedding_model = (
        settings.openai_embedding_model_id
        if embedding_provider == "openai"
        else settings.huggingface_embedding_model_id
    ) or "unknown"

    usage_tracker = CostTracker(
        report_name="workflow_usage_report",
        run_type="workflow",
    )

    embedding_generator = create_embedding_generator(settings)
    embedding_generator = wrap_embeddings_with_tracking(
        embedding_generator,
        usage_tracker,
        provider=embedding_provider,
        model=embedding_model,
    )

    print("Loading vector index...")
    if emit is not None:
        await emit({"type": "log", "message": "Loading vector index..."})
    vectorstore = load_vector_index(target_index_path, embedding_generator)

    print("Initializing RAG retrieval tool...")
    if emit is not None:
        await emit({"type": "log", "message": "Initializing RAG retrieval tool..."})
    search_tool = create_document_rag_tool(vectorstore)

    print("Starting RFP analysis workflow...")
    if emit is not None:
        await emit({"type": "log", "message": "Starting RFP analysis workflow..."})
    results = await run_opportunity_analysis(
        search_tool=search_tool,
        md_path=markdown_data_path,
        # rfp_text_summary=load_workflow_prompt(),
        usage_tracker=usage_tracker,
        provider_hint="anthropic" if settings.Anthropic_model_use else "openai",
        model_hint=settings.anthropic_chat_model_id if settings.Anthropic_model_use else settings.openai_chat_model_id,
        emit=emit,
        human_response_queue=human_response_queue,
    )
    report_path = usage_tracker.save_report()
    print(f"Usage and cost report saved to: {report_path}")
    if emit is not None:
        await emit({"type": "log", "message": f"Usage and cost report saved to: {report_path}"})
    return {
        "results": results,
        "usage_report": str(report_path),
    }
