"""Academic retrieval-augmented generation support."""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_researcher_assistant.core.config import get_config
from ai_researcher_assistant.memory.base import BaseEmbedding, BaseMemory, MemoryItem
from ai_researcher_assistant.memory.in_memory import HashEmbedding, InMemoryVectorMemory

logger = logging.getLogger(__name__)


class AcademicRAG:
    """Paper-oriented RAG layer with chunking, deduplication, and context building."""

    def __init__(
        self,
        collection_name: str = "academic_papers",
        persist_directory: str | None = None,
        embedding_model: BaseEmbedding | None = None,
        memory: BaseMemory | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        prefer_persistent: bool = False,
    ):
        config = get_config()
        self.chunk_size = chunk_size or config.memory.chunk_size
        self.chunk_overlap = min(
            chunk_overlap if chunk_overlap is not None else config.memory.chunk_overlap, self.chunk_size - 1
        )
        self.memory = memory or self._create_memory(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            prefer_persistent=prefer_persistent,
        )

    def add_paper(
        self,
        title: str,
        abstract: str,
        full_text: str | None = None,
        authors: list[str] | None = None,
        arxiv_id: str | None = None,
        categories: list[str] | None = None,
        published_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        replace_existing: bool = True,
    ) -> list[str]:
        """Add or replace a paper in the knowledge base."""

        paper_key = arxiv_id or normalize_paper_key(title)
        if replace_existing:
            self.delete_paper(arxiv_id=arxiv_id, title=None if arxiv_id else title)

        base_metadata = {
            **(metadata or {}),
            "title": title,
            "authors": authors or [],
            "arxiv_id": arxiv_id,
            "paper_key": paper_key,
            "categories": categories or [],
            "published_date": published_date,
            "type": "paper",
        }

        items: list[MemoryItem] = []
        if abstract:
            items.append(
                MemoryItem(
                    content=f"Title: {title}\nAbstract: {abstract}",
                    metadata={**base_metadata, "section": "abstract", "chunk_index": -1},
                )
            )

        if full_text:
            for index, chunk in enumerate(self._chunk_text(full_text)):
                items.append(
                    MemoryItem(
                        content=chunk,
                        metadata={**base_metadata, "section": "full_text", "chunk_index": index},
                    )
                )

        return self._add_batch(items)

    def search_papers(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search papers and aggregate matching chunks by paper."""

        where = {"type": "paper"}
        if arxiv_id:
            where["arxiv_id"] = arxiv_id

        results = self.memory.search(query, top_k=max(top_k * 5, top_k), where=where)
        papers: dict[str, dict[str, Any]] = {}

        for item in results:
            if categories and not set(categories).intersection(set(item.metadata.get("categories", []))):
                continue

            paper_key = (
                item.metadata.get("paper_key")
                or item.metadata.get("arxiv_id")
                or normalize_paper_key(item.metadata.get("title", ""))
            )
            paper = papers.setdefault(
                paper_key,
                {
                    "title": item.metadata.get("title"),
                    "authors": item.metadata.get("authors", []),
                    "arxiv_id": item.metadata.get("arxiv_id"),
                    "categories": item.metadata.get("categories", []),
                    "published_date": item.metadata.get("published_date"),
                    "chunks": [],
                    "abstract": None,
                    "score": item.metadata.get("_distance", 1.0),
                },
            )
            paper["score"] = min(paper["score"], item.metadata.get("_distance", 1.0))

            if item.metadata.get("section") == "abstract":
                paper["abstract"] = item.content
            else:
                paper["chunks"].append(
                    {
                        "content": item.content,
                        "score": item.metadata.get("_distance", 1.0),
                        "chunk_index": item.metadata.get("chunk_index"),
                    }
                )

        sorted_papers = sorted(papers.values(), key=lambda paper: paper["score"])
        return sorted_papers[:top_k]

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 4000,
        include_full_text: bool = False,
        **filters: Any,
    ) -> str:
        """Build compact paper context for an LLM prompt."""

        papers = self.search_papers(query, top_k=top_k, **filters)
        context_parts = []
        current_tokens = 0

        for paper in papers:
            paper_text = self._format_paper_context(paper, include_full_text=include_full_text)
            tokens = max(1, len(paper_text) // 4)
            if current_tokens + tokens > max_tokens:
                break
            context_parts.append(paper_text)
            current_tokens += tokens

        return "".join(context_parts)

    def delete_paper(self, arxiv_id: str | None = None, title: str | None = None) -> int:
        """Delete all chunks for one paper by arXiv ID or title."""

        if not arxiv_id and not title:
            return 0
        where = (
            {"type": "paper", "arxiv_id": arxiv_id}
            if arxiv_id
            else {"type": "paper", "paper_key": normalize_paper_key(title or "")}
        )
        if hasattr(self.memory, "delete_many"):
            return self.memory.delete_many(where)  # type: ignore[attr-defined]
        deleted = 0
        for item in self._list_items(where=where):
            if self.memory.delete(item.id):
                deleted += 1
        return deleted

    def count_papers(self) -> int:
        """Return a real unique paper count rather than a chunk estimate."""

        keys = set()
        for item in self._list_items(where={"type": "paper"}):
            keys.add(
                item.metadata.get("paper_key")
                or item.metadata.get("arxiv_id")
                or normalize_paper_key(item.metadata.get("title", ""))
            )
        return len(keys)

    def _chunk_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                for sep in [". ", "? ", "! ", "\n", "。", "？", "！"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1 and last_sep > start:
                        end = last_sep + len(sep)
                        break
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return [chunk for chunk in chunks if chunk]

    def _create_memory(
        self,
        collection_name: str,
        persist_directory: str | None,
        embedding_model: BaseEmbedding | None,
        prefer_persistent: bool,
    ) -> BaseMemory:
        if not prefer_persistent:
            return InMemoryVectorMemory(embedding_model or HashEmbedding())

        try:
            from ai_researcher_assistant.memory.long_term import LongTermMemory, OpenAIEmbedding

            return LongTermMemory(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_model=embedding_model or OpenAIEmbedding(),
            )
        except Exception as exc:
            logger.warning("Falling back to in-memory RAG backend: %s", exc)
            return InMemoryVectorMemory(embedding_model or HashEmbedding())

    def _add_batch(self, items: list[MemoryItem]) -> list[str]:
        if not items:
            return []
        if hasattr(self.memory, "add_batch"):
            return self.memory.add_batch(items)  # type: ignore[attr-defined]
        return [self.memory.add(item.content, item.metadata) for item in items]

    def _list_items(self, where: dict[str, Any] | None = None) -> list[MemoryItem]:
        if hasattr(self.memory, "list"):
            return self.memory.list(where=where)  # type: ignore[attr-defined]
        return self.memory.search("", top_k=self.memory.count(), where=where or {})

    def _format_paper_context(self, paper: dict[str, Any], include_full_text: bool) -> str:
        lines = [f"### {paper['title']}"]
        if paper.get("authors"):
            lines.append(f"Authors: {', '.join(paper['authors'])}")
        if paper.get("arxiv_id"):
            lines.append(f"arXiv: {paper['arxiv_id']}")
        if paper.get("categories"):
            lines.append(f"Categories: {', '.join(paper['categories'])}")
        if paper.get("abstract"):
            lines.append("")
            lines.append(f"Abstract: {paper['abstract']}")
        if include_full_text and paper.get("chunks"):
            lines.append("")
            lines.append("Excerpts:")
            for chunk in sorted(paper["chunks"], key=lambda item: item.get("score", 1.0))[:3]:
                lines.append(f"- {chunk['content'][:500]}...")
        lines.append("\n---\n")
        return "\n".join(lines)


def normalize_paper_key(title: str) -> str:
    normalized = re.sub(r"\s+", "-", title.strip().lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff-]+", "", normalized)
    return normalized or "untitled-paper"
