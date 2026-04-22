"""本地模型适配器（Ollama）"""
import os
import json
from typing import List, Dict, Any, Optional, AsyncIterator
import aiohttp
import requests

from physicist_agent.llm.base import BaseLLM, LLMResponse
from physicist_agent.core.exceptions import LLMError


class OllamaLLM(BaseLLM):
    """Ollama 本地模型适配器"""

    def __init__(
        self,
        model: str = "llama3",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.generate_url = f"{self.base_url}/api/chat"

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            }
            
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            raise LLMError(f"Ollama API error: {str(e)}") from e

    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_url, json=payload, timeout=120) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )
        except Exception as e:
            raise LLMError(f"Ollama async API error: {str(e)}") from e

    async def stream_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_url, json=payload, timeout=120) as resp:
                    resp.raise_for_status()
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            raise LLMError(f"Ollama streaming error: {str(e)}") from e
