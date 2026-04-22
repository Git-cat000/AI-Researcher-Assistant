"""Physicist Agent Framework - LLM Module"""

from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse
from ai_researcher_assistant.llm.openai import OpenAILLM
from ai_researcher_assistant.llm.anthropic import AnthropicLLM
from ai_researcher_assistant.llm.local import OllamaLLM
from ai_researcher_assistant.llm.factory import create_llm, get_llm

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "create_llm",
    "get_llm",
]
