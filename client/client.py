"""LLM client integration helpers for Microsoft Agent Framework."""

from typing import TYPE_CHECKING, Any

from config.settings import Settings, get_settings

if TYPE_CHECKING:
    from agent_framework.anthropic import AnthropicClient
    from agent_framework.openai import OpenAIChatClient


class ClaudeAgentConfigurationError(RuntimeError):
    """Raised when Anthropic/Claude Agent Framework settings are incomplete."""


class OpenAIAgentConfigurationError(RuntimeError):
    """Raised when OpenAI Agent Framework settings are incomplete."""


def create_openai_chat_client(settings: Settings | None = None) -> "OpenAIChatClient":
    """Create an OpenAI-backed Microsoft Agent Framework chat client."""

    effective_settings = settings or get_settings()
    model = effective_settings.openai_chat_model_id
    if not model:
        raise OpenAIAgentConfigurationError("OPENAI_CHAT_MODEL_ID is required.")

    try:
        from agent_framework.openai import OpenAIChatClient
    except ImportError as exc:
        raise OpenAIAgentConfigurationError(
            "Install the OpenAI provider package with `pip install agent-framework-openai` "
            "to use OpenAIChatClient."
        ) from exc

    kwargs: dict[str, Any] = {"model": model}
    if effective_settings.openai_api_key:
        kwargs["api_key"] = effective_settings.openai_api_key
    if effective_settings.openai_base_url:
        kwargs["base_url"] = effective_settings.openai_base_url

    return OpenAIChatClient(**kwargs)


def create_claude_chat_client(settings: Settings | None = None) -> "AnthropicClient":
    """Create an Anthropic-backed Microsoft Agent Framework chat client."""

    effective_settings = settings or get_settings()
    model = (
        effective_settings.anthropic_chat_model_id
        or effective_settings.claude_chat_model_id
    )
    if not model:
        raise ClaudeAgentConfigurationError("ANTHROPIC_CHAT_MODEL_ID is required.")

    try:
        from agent_framework.anthropic import AnthropicClient
    except ImportError as exc:
        raise ClaudeAgentConfigurationError(
            "Install the Anthropic provider package with `pip install agent-framework-anthropic --pre` "
            "to use AnthropicClient."
        ) from exc

    kwargs: dict[str, Any] = {"model": model}
    if effective_settings.anthropic_api_key:
        kwargs["api_key"] = effective_settings.anthropic_api_key
    if effective_settings.anthropic_base_url:
        kwargs["base_url"] = effective_settings.anthropic_base_url

    return AnthropicClient(**kwargs)
