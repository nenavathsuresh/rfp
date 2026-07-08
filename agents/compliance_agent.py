from pathlib import Path
from typing import Any
from utils.helper import read_instructions
from client.client import create_claude_chat_client


def create_compliance_agent(search_tool: Any, project_root: Path | None = None) -> Any:
    """Load instructions from YAML and construct the Compliance Agent using AnthropicClient."""
    root = project_root or Path.cwd()
    
    primary_data = read_instructions(root / "instructions" / "compliance.yaml")
    shared_data = read_instructions(root / "instructions" / "shared_rag.yaml")
    
    instructions = (
        f"{primary_data['system_instructions']}\n\n"
        f"Default user task for each run:\n{primary_data['user_prompt']}\n\n"
        f"{shared_data['system_instructions']}"
    )
    title = primary_data["title"]
    user_prompt = primary_data["user_prompt"]
    client = create_claude_chat_client()

    agent = client.as_agent(
        name=title,
        instructions=instructions,
        tools=search_tool,
    )
    # Attach prompt so orchestration can always send it on each run.
    setattr(agent, "user_prompt", user_prompt)
    return agent
