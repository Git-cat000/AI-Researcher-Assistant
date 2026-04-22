"""Anthropic Claude 适配器"""
import os
from typing import List, Dict, Any, Optional, AsyncIterator
import anthropic
from anthropic import AsyncAnthropic

from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse
from ai_researcher_assistant.core.exceptions import LLMError


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API 实现"""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict]]:
        """将 OpenAI 格式消息转换为 Anthropic 格式"""
        system_prompt = None
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            else:
                # Anthropic 支持 user 和 assistant
                converted.append({"role": role, "content": content})
        
        return system_prompt, converted

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            system, converted_messages = self._convert_messages(messages)
            
            response = self.client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system or anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop", None),
            )
            return self._parse_response(response)
        except Exception as e:
            raise LLMError(f"Anthropic API error: {str(e)}") from e

    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            system, converted_messages = self._convert_messages(messages)
            
            response = await self.async_client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system or anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop", None),
            )
            return self._parse_response(response)
        except Exception as e:
            raise LLMError(f"Anthropic async API error: {str(e)}") from e

    async def stream_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        try:
            system, converted_messages = self._convert_messages(messages)
            
            async with self.async_client.messages.stream(
                model=self.model,
                messages=converted_messages,
                system=system or anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop", None),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield event.delta.text
        except Exception as e:
            raise LLMError(f"Anthropic streaming error: {str(e)}") from e

    def _parse_response(self, response) -> LLMResponse:
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            } if response.usage else None,
            finish_reason=response.stop_reason,
            raw_response=response,
        )
