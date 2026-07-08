from pathlib import Path

class AnthropicConfigurationError(RuntimeError):
    """Raised when OpenAI Agent Framework settings are incomplete."""

def read_instructions(path: Path) -> dict:
    """Load and validate an agent instruction YAML file."""

    if not path.exists():
        raise AnthropicConfigurationError(
            f"Agent instructions file not found: {path}"
        )

    if path.suffix not in {".yaml", ".yml"}:
        raise AnthropicConfigurationError(
            f"Agent instructions must be YAML: {path}"
        )

    try:
        import yaml
    except ImportError as exc:
        raise AnthropicConfigurationError(
            "PyYAML is required to read YAML instructions."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise AnthropicConfigurationError(
            f"Instruction YAML must contain an object: {path}"
        )

    title = str(data.get("title") or "").strip()
    system_instructions = str(data.get("system_instructions") or "").strip()
    user_prompt = str(data.get("user_prompt") or "").strip()

    missing_fields = []
    if not title:
        missing_fields.append("title")
    if not system_instructions:
        missing_fields.append("system_instructions")
    if not user_prompt:
        missing_fields.append("user_prompt")

    if missing_fields:
        missing_list = ", ".join(f"`{field}`" for field in missing_fields)
        raise AnthropicConfigurationError(
            f"Instruction YAML missing required field(s) {missing_list}: {path}"
        )

    return {
        "title": title,
        "system_instructions": system_instructions,
        "user_prompt": user_prompt,
    }
