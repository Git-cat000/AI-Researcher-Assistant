"""
记忆系统抽象基类。
定义了记忆存储与检索的统一接口。
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryItem:
    """记忆项"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class BaseMemory(ABC):
    """记忆系统抽象基类"""

    @abstractmethod
    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        添加一条记忆。

        Args:
            content: 记忆内容
            metadata: 元数据（如来源、类型等）

        Returns:
            记忆项 ID
        """
        pass

    @abstractmethod
    def get(self, memory_id: str) -> MemoryItem | None:
        """根据 ID 获取记忆"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        语义搜索记忆。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关的记忆项列表
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""
        pass

    @abstractmethod
    def count(self) -> int:
        """返回记忆总数"""
        pass


class BaseEmbedding(ABC):
    """嵌入模型抽象基类"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将文本转换为向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成向量"""
        pass
