from pathlib import Path
from typing import Any

from client.client import create_claude_chat_client, create_openai_chat_client
from config.settings import get_settings
from utils.helper import read_instructions


def _create_chat_client() -> Any:
    """Create the configured chat client for the evidence agent."""

    settings = get_settings()
    if settings.Anthropic_model_use:
        return create_claude_chat_client(settings)
    return create_openai_chat_client(settings)


def create_evidence_agent(project_root: Path | None = None) -> Any:
    """Construct the Evidence Agent for Q/A over extracted requirements."""

    root = project_root or Path.cwd()
    primary_data = read_instructions(root / "instructions" / "evidence.yaml")

    instructions = f"{primary_data['system_instructions']}\n\n"
    title = primary_data["title"]
    user_prompt = primary_data["user_prompt"]
    client = _create_chat_client()

    agent = client.as_agent(
        name=title,
        instructions=instructions,
    )
    setattr(agent, "user_prompt", user_prompt)
    return agent
