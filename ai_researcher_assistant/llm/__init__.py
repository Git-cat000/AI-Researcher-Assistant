"""LLM adapters and factory helpers.

Provider SDKs are imported lazily so tests and local tooling can use the base
contracts without installing every optional model provider.
"""

from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse
from ai_researcher_assistant.llm.capabilities import (
    MODEL_CAPABILITIES,
    ModelCapability,
    get_model_capability,
    list_model_capabilities,
)
from ai_researcher_assistant.llm.factory import create_llm, get_llm


def __getattr__(name: str):
    if name == "OpenAILLM":
        from ai_researcher_assistant.llm.openai import OpenAILLM

        return OpenAILLM
    if name == "AnthropicLLM":
        from ai_researcher_assistant.llm.anthropic import AnthropicLLM

        return AnthropicLLM
    if name == "OllamaLLM":
        from ai_researcher_assistant.llm.local import OllamaLLM

        return OllamaLLM
    raise AttributeError(name)


__all__ = [
    "BaseLLM",
    "LLMResponse",
    "ModelCapability",
    "MODEL_CAPABILITIES",
    "get_model_capability",
    "list_model_capabilities",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "create_llm",
    "get_llm",
]
