"""Anthropic Claude adapter."""

from collections.abc import AsyncIterator
import os
from typing import Any

from ai_researcher_assistant.core.exceptions import LLMError
from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    """LLM adapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key.")

        try:
            import anthropic
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise LLMError("Anthropic SDK is not installed. Run `pip install anthropic`.") from exc

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        system_prompt = None
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                converted.append({"role": role, "content": content})
        return system_prompt, converted

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            system, converted_messages = self._convert_messages(messages)
            response = self.client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system or self._anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop"),
            )
            return self._parse_response(response)
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

    async def agenerate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            system, converted_messages = self._convert_messages(messages)
            response = await self.async_client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system or self._anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop"),
            )
            return self._parse_response(response)
        except Exception as exc:
            raise LLMError(f"Anthropic async API error: {exc}") from exc

    async def stream_generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        try:
            system, converted_messages = self._convert_messages(messages)
            async with self.async_client.messages.stream(
                model=self.model,
                messages=converted_messages,
                system=system or self._anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop"),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
        except Exception as exc:
            raise LLMError(f"Anthropic streaming error: {exc}") from exc

    def _parse_response(self, response: Any) -> LLMResponse:
        content = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            if response.usage
            else None,
            finish_reason=response.stop_reason,
            raw_response=response,
        )
