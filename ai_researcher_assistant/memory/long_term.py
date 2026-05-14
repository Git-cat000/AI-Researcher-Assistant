"""
长期记忆：基于向量数据库的持久化存储。
"""

import json
import os
from typing import Any

from ai_researcher_assistant.core.config import get_config
from ai_researcher_assistant.core.exceptions import MemoryError
from ai_researcher_assistant.memory.base import BaseEmbedding, BaseMemory, MemoryItem


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 实现"""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        import openai

        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class LongTermMemory(BaseMemory):
    """
    长期记忆，使用 ChromaDB 作为向量存储。
    支持语义检索和元数据过滤。
    """

    def __init__(
        self,
        collection_name: str = "ai_researcher_memory",
        persist_directory: str | None = None,
        embedding_model: BaseEmbedding | None = None,
    ):
        config = get_config()
        self.persist_directory = persist_directory or config.memory.vector_db_path

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise MemoryError("ChromaDB is not installed. Run `pip install chromadb`.") from exc

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 嵌入模型
        if embedding_model is None:
            embedding_model = OpenAIEmbedding(model=config.memory.embedding_model)
        self.embedding_model = embedding_model

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        item = MemoryItem(content=content, metadata=metadata or {})
        embedding = self.embedding_model.embed(content)

        # 将元数据转换为字符串（ChromaDB 要求）
        str_metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in item.metadata.items()}
        str_metadata["created_at"] = item.created_at.isoformat()

        self.collection.add(
            ids=[item.id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[str_metadata],
        )
        return item.id

    def add_batch(self, items: list[MemoryItem]) -> list[str]:
        """批量添加记忆项"""
        if not items:
            return []

        texts = [item.content for item in items]
        embeddings = self.embedding_model.embed_batch(texts)

        ids = [item.id for item in items]
        metadatas = [
            {k: json.dumps(v) if not isinstance(v, str) else v for k, v in item.metadata.items()} for item in items
        ]
        for meta, item in zip(metadatas, items, strict=True):
            meta["created_at"] = item.created_at.isoformat()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def get(self, memory_id: str) -> MemoryItem | None:
        result = self.collection.get(ids=[memory_id])
        if result and result["ids"]:
            return self._to_memory_item(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0] if result["metadatas"] else {},
            )
        return None

    def search(self, query: str, top_k: int = 5, where: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        """
        语义搜索。

        Args:
            query: 查询文本
            top_k: 返回数量
            where: 元数据过滤条件，如 {"category": "hep-th"}
        """
        query_embedding = self.embedding_model.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None

                item = self._to_memory_item(doc_id, doc, meta)
                if distance is not None:
                    item.metadata["_distance"] = distance
                items.append(item)

        return items

    def delete(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as exc:
            raise MemoryError(f"Failed to delete memory item {memory_id}") from exc

    def delete_many(self, where: dict[str, Any] | None = None) -> int:
        result = self.collection.get(where=where)
        ids = result.get("ids", []) if result else []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def list(self, where: dict[str, Any] | None = None, limit: int | None = None) -> list[MemoryItem]:
        result = self.collection.get(where=where, limit=limit)
        ids = result.get("ids", []) if result else []
        documents = result.get("documents", []) if result else []
        metadatas = result.get("metadatas", []) if result else []
        return [
            self._to_memory_item(item_id, document, metadatas[index] if metadatas else {})
            for index, (item_id, document) in enumerate(zip(ids, documents, strict=True))
        ]

    def clear(self) -> None:
        # 删除并重建集合
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def _to_memory_item(self, doc_id: str, document: str, metadata: dict) -> MemoryItem:
        """将 ChromaDB 结果转换为 MemoryItem"""
        # 反序列化元数据中的 JSON 字符串
        parsed_meta = {}
        for k, v in metadata.items():
            if k == "created_at":
                continue
            try:
                parsed_meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed_meta[k] = v

        from datetime import datetime

        created_at = datetime.fromisoformat(metadata["created_at"]) if "created_at" in metadata else datetime.now()

        return MemoryItem(
            id=doc_id,
            content=document,
            metadata=parsed_meta,
            created_at=created_at,
        )
