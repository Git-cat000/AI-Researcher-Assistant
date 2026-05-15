"""Provider capability metadata used by docs, CLI, and routing policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapability:
    provider: str
    adapter: str
    supports_tools: bool
    supports_streaming: bool
    supports_vision: bool
    supports_long_context: bool
    notes: str = ""
    credential_env: str | None = None
    base_url_env: str | None = None
    is_local: bool = False
    is_openai_compatible: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "adapter": self.adapter,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "supports_vision": self.supports_vision,
            "supports_long_context": self.supports_long_context,
            "notes": self.notes,
            "credential_env": self.credential_env,
            "base_url_env": self.base_url_env,
            "is_local": self.is_local,
            "is_openai_compatible": self.is_openai_compatible,
        }


MODEL_CAPABILITIES: dict[str, ModelCapability] = {
    "openai": ModelCapability(
        "openai", "OpenAILLM", True, True, True, True, credential_env="OPENAI_API_KEY", is_openai_compatible=True
    ),
    "openai_compatible": ModelCapability(
        "openai_compatible",
        "OpenAILLM",
        False,
        True,
        False,
        True,
        "Depends on the target OpenAI-compatible endpoint.",
        credential_env="OPENAI_API_KEY",
        is_openai_compatible=True,
    ),
    "azure_openai": ModelCapability(
        "azure_openai",
        "OpenAILLM",
        True,
        True,
        True,
        True,
        credential_env="AZURE_OPENAI_API_KEY",
        is_openai_compatible=True,
    ),
    "deepseek": ModelCapability(
        "deepseek",
        "OpenAILLM",
        True,
        True,
        False,
        True,
        credential_env="DEEPSEEK_API_KEY",
        is_openai_compatible=True,
    ),
    "openrouter": ModelCapability(
        "openrouter",
        "OpenAILLM",
        True,
        True,
        True,
        True,
        "Capabilities depend on the selected routed model.",
        credential_env="OPENROUTER_API_KEY",
        is_openai_compatible=True,
    ),
    "siliconflow": ModelCapability(
        "siliconflow",
        "OpenAILLM",
        True,
        True,
        False,
        True,
        "Capabilities depend on the selected model.",
        credential_env="SILICONFLOW_API_KEY",
        is_openai_compatible=True,
    ),
    "qwen": ModelCapability(
        "qwen",
        "OpenAILLM",
        True,
        True,
        True,
        True,
        credential_env="DASHSCOPE_API_KEY",
        is_openai_compatible=True,
    ),
    "anthropic": ModelCapability(
        "anthropic", "AnthropicLLM", True, True, True, True, credential_env="ANTHROPIC_API_KEY"
    ),
    "ollama": ModelCapability(
        "ollama",
        "OllamaLLM",
        False,
        True,
        False,
        False,
        "Local model capabilities depend on the installed Ollama model.",
        base_url_env="OLLAMA_BASE_URL",
        is_local=True,
    ),
    "local": ModelCapability(
        "local",
        "OllamaLLM",
        False,
        True,
        False,
        False,
        "Alias for a local Ollama-compatible runtime.",
        base_url_env="OLLAMA_BASE_URL",
        is_local=True,
    ),
}


def get_model_capability(provider: str) -> ModelCapability:
    normalized = provider.lower()
    if normalized not in MODEL_CAPABILITIES:
        raise KeyError(f"Unknown provider: {provider}")
    return MODEL_CAPABILITIES[normalized]


def list_model_capabilities() -> list[ModelCapability]:
    return list(MODEL_CAPABILITIES.values())
