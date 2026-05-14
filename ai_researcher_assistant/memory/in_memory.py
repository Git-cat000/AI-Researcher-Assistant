"""Dependency-free memory backend for local tests and lightweight RAG."""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt
from typing import Any

from ai_researcher_assistant.memory.base import BaseEmbedding, BaseMemory, MemoryItem


class HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding with no external service."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        counts = Counter(tokens)
        for token, count in counts.items():
            index = hash(token) % self.dimensions
            vector[index] += float(count)
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class InMemoryVectorMemory(BaseMemory):
    """Simple vector memory implementation for local and test workflows."""

    def __init__(self, embedding_model: BaseEmbedding | None = None):
        self.embedding_model = embedding_model or HashEmbedding()
        self._items: dict[str, MemoryItem] = {}

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        item = MemoryItem(content=content, metadata=metadata or {})
        item.embedding = self.embedding_model.embed(content)
        self._items[item.id] = item
        return item.id

    def add_batch(self, items: list[MemoryItem]) -> list[str]:
        if not items:
            return []
        embeddings = self.embedding_model.embed_batch([item.content for item in items])
        for item, embedding in zip(items, embeddings):
            item.embedding = embedding
            self._items[item.id] = item
        return [item.id for item in items]

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def search(self, query: str, top_k: int = 5, where: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        query_embedding = self.embedding_model.embed(query)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items.values():
            if where and not metadata_matches(item.metadata, where):
                continue
            score = cosine_similarity(query_embedding, item.embedding or [])
            clone = MemoryItem(
                id=item.id, content=item.content, metadata=dict(item.metadata), created_at=item.created_at
            )
            clone.embedding = item.embedding
            clone.metadata["_distance"] = 1.0 - score
            scored.append((score, clone))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def delete(self, memory_id: str) -> bool:
        return self._items.pop(memory_id, None) is not None

    def delete_many(self, where: dict[str, Any]) -> int:
        ids = [item.id for item in self._items.values() if metadata_matches(item.metadata, where)]
        for item_id in ids:
            del self._items[item_id]
        return len(ids)

    def list(self, where: dict[str, Any] | None = None, limit: int | None = None) -> list[MemoryItem]:
        items = [item for item in self._items.values() if not where or metadata_matches(item.metadata, where)]
        if limit is not None:
            items = items[:limit]
        return [
            MemoryItem(
                id=item.id,
                content=item.content,
                metadata=dict(item.metadata),
                created_at=item.created_at,
                embedding=item.embedding,
            )
            for item in items
        ]

    def clear(self) -> None:
        self._items.clear()

    def count(self) -> int:
        return len(self._items)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


def metadata_matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, expected in where.items():
        if key == "$and":
            return all(metadata_matches(metadata, condition) for condition in expected)
        value = metadata.get(key)
        if isinstance(expected, dict):
            if "$contains" in expected:
                needle = expected["$contains"]
                if isinstance(value, list) and needle not in value:
                    return False
                if isinstance(value, str) and needle not in value:
                    return False
            elif "$in" in expected:
                if value not in expected["$in"]:
                    return False
            else:
                return False
        elif value != expected:
            return False
    return True
