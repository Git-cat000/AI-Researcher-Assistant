"""
LLM 抽象基类。
定义了与语言模型交互的统一接口。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """LLM 响应结构"""

    content: str
    model: str
    usage: dict[str, int] | None = None  # {"prompt_tokens": 100, "completion_tokens": 50}
    finish_reason: str | None = None
    raw_response: Any = None


class BaseLLM(ABC):
    """LLM 抽象基类"""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """
        同步生成回复。

        Args:
            messages: 标准格式的消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数（如 stop, top_p 等）

        Returns:
            LLMResponse 对象
        """
        pass

    @abstractmethod
    async def agenerate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """异步生成回复"""
        pass

    @abstractmethod
    def stream_generate(self, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """
        流式生成回复，逐步产出文本片段。

        Yields:
            生成的文本片段
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量。
        子类可以覆盖以提供精确计数。
        """
        # 粗略估算：英文约 4 字符/token，中文约 1.5 字符/token
        return len(text) // 4
