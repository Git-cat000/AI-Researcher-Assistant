"""
短期记忆：会话上下文管理。
使用滑动窗口控制 Token 数量，防止上下文溢出。
"""
from typing import List, Dict, Any, Optional
from collections import deque

from ai_researcher_assistant.memory.base import BaseMemory, MemoryItem
from ai_researcher_assistant.core.message import Message, MessageRole


class ShortTermMemory(BaseMemory):
    """
    短期记忆，基于滑动窗口。
    用于存储最近的对话历史。
    """

    def __init__(self, max_tokens: int = 10000, max_messages: int = 50):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self._messages: deque[Message] = deque()
        self._token_count: int = 0

    def add_message(self, message: Message) -> None:
        """添加一条消息（Message 对象）"""
        self._messages.append(message)
        self._token_count += self._estimate_tokens(message.content)
        self._trim()

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加一条记忆（兼容 BaseMemory 接口）。
        默认视为 user 消息。
        """
        role = MessageRole.USER
        if metadata and "role" in metadata:
            role = MessageRole(metadata["role"])
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.add_message(msg)
        return msg.id

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        for msg in self._messages:
            if msg.id == memory_id:
                return MemoryItem(
                    id=msg.id,
                    content=msg.content,
                    metadata=msg.metadata,
                    created_at=msg.timestamp,
                )
        return None

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[MemoryItem]:
        """
        短期记忆不支持语义搜索，返回最近的 top_k 条。
        """
        recent = list(self._messages)[-top_k:]
        return [
            MemoryItem(
                id=msg.id,
                content=msg.content,
                metadata=msg.metadata,
                created_at=msg.timestamp,
            )
            for msg in recent
        ]

    def delete(self, memory_id: str) -> bool:
        for i, msg in enumerate(self._messages):
            if msg.id == memory_id:
                self._token_count -= self._estimate_tokens(msg.content)
                del self._messages[i]
                return True
        return False

    def clear(self) -> None:
        self._messages.clear()
        self._token_count = 0

    def count(self) -> int:
        return len(self._messages)

    def get_messages(self) -> List[Message]:
        """获取所有消息（用于 LLM 上下文）"""
        return list(self._messages)

    def get_context_for_llm(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取格式化的上下文，用于 LLM API 调用。
        """
        max_tokens = max_tokens or self.max_tokens
        messages = []
        current_tokens = 0

        # 从旧到新添加，直到接近 token 限制
        for msg in reversed(self._messages):
            tokens = self._estimate_tokens(msg.content)
            if current_tokens + tokens > max_tokens:
                break
            messages.insert(0, msg.to_llm_format())
            current_tokens += tokens

        return messages

    def _trim(self) -> None:
        """修剪超出限制的消息"""
        # 按数量修剪
        while len(self._messages) > self.max_messages:
            oldest = self._messages.popleft()
            self._token_count -= self._estimate_tokens(oldest.content)

        # 按 Token 数修剪
        while self._token_count > self.max_tokens and self._messages:
            oldest = self._messages.popleft()
            self._token_count -= self._estimate_tokens(oldest.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（英文 4 字符/token，中文 1.5 字符/token）"""
        # 简化实现，实际可用 tiktoken
        return len(text) // 4
