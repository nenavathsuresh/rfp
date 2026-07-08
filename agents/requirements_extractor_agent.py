from pathlib import Path
from typing import Any
from utils.helper import read_instructions
from client.client import create_claude_chat_client,create_openai_chat_client

def create_requirements_extractor_agent(search_tool: Any, project_root: Path | None = None) -> Any:
    """Load instructions from YAML and construct the Requirements_Extractor  Agent using AnthropicClient."""
    root = project_root or Path.cwd()
    
    primary_data = read_instructions(root / "instructions" / "Requirements_Extractor.yaml")
    #shared_data = read_instructions(root / "instructions" / "shared_rag.yaml")
    
    instructions = (
        f"{primary_data['system_instructions']}\n\n"
    )
    title = primary_data["title"]
    user_prompt = primary_data["user_prompt"]
    total_prompt = primary_data['system_instructions']+primary_data["user_prompt"]
    client = create_openai_chat_client() #create_claude_chat_client()

    agent = client.as_agent(
        name=title,
        instructions=instructions,
        tools=search_tool,
    )
    # Attach prompt so orchestration can always send it on each run.
    setattr(agent, "total_prompt", total_prompt)
    return agent


def create_requirements_extractor_agent_markdown(project_root: Path | None = None) -> Any:
    """Load instructions from YAML and construct the Requirements_Extractor  Agent using AnthropicClient."""
    root = project_root or Path.cwd()
    
    primary_data = read_instructions(root / "instructions" / "Requirements_Extractor.yaml")
    
    instructions = (
        f"{primary_data['system_instructions']}\n\n"
    )
    title = primary_data["title"]
    user_prompt = primary_data["user_prompt"]
    client = create_openai_chat_client() #create_claude_chat_client()

    agent = client.as_agent(
        name=title,
        instructions=instructions,
    )
    # Attach prompt so orchestration can always send it on each run.
    setattr(agent, "user_prompt", user_prompt)
    return agent
