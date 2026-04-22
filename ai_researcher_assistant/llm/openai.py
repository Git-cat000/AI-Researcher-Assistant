"""OpenAI 适配器（兼容 DeepSeek、Azure OpenAI 等）"""
import os
from typing import List, Dict, Any, Optional, AsyncIterator
import openai
from openai import AsyncOpenAI

from ai_researcher_assistant.llm.base import BaseLLM, LLMResponse
from ai_researcher_assistant.core.exceptions import LLMError


class OpenAILLM(BaseLLM):
    """OpenAI 及兼容 API 的 LLM 实现"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop", None),
            )
            return self._parse_response(response)
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}") from e

    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop", None),
            )
            return self._parse_response(response)
        except Exception as e:
            raise LLMError(f"OpenAI async API error: {str(e)}") from e

    async def stream_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                stop=kwargs.get("stop", None),
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"OpenAI streaming error: {str(e)}") from e

    def _parse_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
