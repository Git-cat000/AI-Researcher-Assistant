"""Factory functions for model adapters."""

from ai_researcher_assistant.core.config import LLMConfig, get_config
from ai_researcher_assistant.core.exceptions import ConfigurationError
from ai_researcher_assistant.llm.base import BaseLLM


def create_llm(config: LLMConfig | None = None) -> BaseLLM:
    """Create an LLM adapter from configuration."""

    config = config or get_config().llm
    provider = config.provider.lower()

    if provider in {"openai", "openai_compatible", "azure_openai", "deepseek", "openrouter", "siliconflow", "qwen"}:
        from ai_researcher_assistant.llm.openai import OpenAILLM

        return OpenAILLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    if provider == "anthropic":
        from ai_researcher_assistant.llm.anthropic import AnthropicLLM

        return AnthropicLLM(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if provider in {"ollama", "local"}:
        from ai_researcher_assistant.llm.local import OllamaLLM

        return OllamaLLM(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ConfigurationError(f"Unsupported LLM provider: {provider}")


def get_llm() -> BaseLLM:
    """Return an LLM adapter using the process default config."""

    return create_llm()
