"""Academic retrieval-augmented generation support."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
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
        citations: list[str] | None = None,
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
            "citations": citations or (metadata or {}).get("citations", []),
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
        authors: list[str] | None = None,
        published_year: str | int | None = None,
        section: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        rerank: bool = True,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search papers and aggregate matching chunks by paper."""

        allowed_modes = {"hybrid", "vector", "keyword", "bm25"}
        if retrieval_mode.lower() not in allowed_modes:
            raise ValueError(f"retrieval_mode must be one of {sorted(allowed_modes)}")

        where = {"type": "paper"}
        if arxiv_id:
            where["arxiv_id"] = arxiv_id
        if section:
            where["section"] = section

        candidate_limit = max(top_k * 8, top_k)
        results = self._hybrid_search_items(query, candidate_limit, where, retrieval_mode, vector_weight)
        papers: dict[str, dict[str, Any]] = {}

        for item in results:
            if not self._passes_filters(item.metadata, categories, authors, published_year, metadata_filter):
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
                    "citations": item.metadata.get("citations", []),
                    "chunks": [],
                    "abstract": None,
                    "score": item.metadata.get("_distance", 1.0),
                    "relevance_score": item.metadata.get("_hybrid_score", 0.0),
                },
            )
            paper["score"] = min(paper["score"], item.metadata.get("_distance", 1.0))
            paper["relevance_score"] = max(paper["relevance_score"], item.metadata.get("_hybrid_score", 0.0))

            if item.metadata.get("section") == "abstract":
                paper["abstract"] = item.content
            else:
                paper["chunks"].append(
                    {
                        "content": item.content,
                        "score": item.metadata.get("_distance", 1.0),
                        "relevance_score": item.metadata.get("_hybrid_score", 0.0),
                        "chunk_index": item.metadata.get("chunk_index"),
                    }
                )

        if rerank:
            for paper in papers.values():
                paper["relevance_score"] += self._rerank_boost(query, paper)

        sorted_papers = sorted(papers.values(), key=lambda paper: paper["relevance_score"], reverse=True)
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

    def update_paper(
        self,
        arxiv_id: str | None = None,
        title: str | None = None,
        abstract: str | None = None,
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
        **fields: Any,
    ) -> list[str]:
        """Incrementally update one paper by replacing it with merged metadata and new content."""

        existing = self.search_papers("", top_k=1, arxiv_id=arxiv_id) if arxiv_id else []
        if not existing and title:
            existing = [paper for paper in self.search_papers(title, top_k=5) if paper.get("title") == title]
        current = existing[0] if existing else {}
        merged_metadata = {**(metadata or {}), **fields}
        return self.add_paper(
            title=title or current.get("title") or "Untitled Paper",
            abstract=abstract or current.get("abstract") or "",
            full_text=full_text,
            authors=fields.get("authors") or current.get("authors"),
            arxiv_id=arxiv_id or current.get("arxiv_id"),
            categories=fields.get("categories") or current.get("categories"),
            published_date=fields.get("published_date") or current.get("published_date"),
            citations=fields.get("citations") or current.get("citations"),
            metadata=merged_metadata,
            replace_existing=True,
        )

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

    def citation_graph(self) -> dict[str, Any]:
        """Return a simple citation graph from paper metadata."""

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        for item in self._list_items(where={"type": "paper", "section": "abstract"}):
            source = item.metadata.get("arxiv_id") or item.metadata.get("paper_key")
            if not source:
                continue
            nodes[source] = {
                "id": source,
                "title": item.metadata.get("title"),
                "authors": item.metadata.get("authors", []),
            }
            for target in item.metadata.get("citations", []) or []:
                target_id = str(target)
                nodes.setdefault(target_id, {"id": target_id, "title": None, "authors": []})
                edges.append({"source": source, "target": target_id})
        return {"nodes": list(nodes.values()), "edges": edges}

    def citation_neighbors(self, paper_id: str) -> dict[str, list[str]]:
        graph = self.citation_graph()
        outgoing = [edge["target"] for edge in graph["edges"] if edge["source"] == paper_id]
        incoming = [edge["source"] for edge in graph["edges"] if edge["target"] == paper_id]
        return {"outgoing": outgoing, "incoming": incoming}

    def _hybrid_search_items(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any],
        retrieval_mode: str,
        vector_weight: float,
    ) -> list[MemoryItem]:
        retrieval_mode = retrieval_mode.lower()
        vector_weight = min(max(vector_weight, 0.0), 1.0)
        candidates: dict[str, MemoryItem] = {}

        if retrieval_mode in {"vector", "hybrid"}:
            for item in self.memory.search(query, top_k=top_k, where=where):
                vector_score = 1.0 - float(item.metadata.get("_distance", 1.0))
                item.metadata["_vector_score"] = vector_score
                item.metadata["_hybrid_score"] = max(0.0, vector_score) if retrieval_mode == "vector" else 0.0
                candidates[item.id] = item

        if retrieval_mode in {"keyword", "bm25", "hybrid"}:
            lexical = self._lexical_search(query, top_k=top_k, where=where)
            max_score = max((score for score, _ in lexical), default=1.0) or 1.0
            for score, item in lexical:
                normalized = score / max_score
                existing = candidates.get(item.id)
                if existing is None:
                    item.metadata["_distance"] = 1.0
                    item.metadata["_vector_score"] = 0.0
                    existing = item
                    candidates[item.id] = existing
                existing.metadata["_bm25_score"] = normalized

        for item in candidates.values():
            vector_score = float(item.metadata.get("_vector_score", 0.0))
            bm25_score = float(item.metadata.get("_bm25_score", 0.0))
            if retrieval_mode == "hybrid":
                item.metadata["_hybrid_score"] = vector_weight * vector_score + (1.0 - vector_weight) * bm25_score
            elif retrieval_mode in {"keyword", "bm25"}:
                item.metadata["_hybrid_score"] = bm25_score

        return sorted(candidates.values(), key=lambda item: item.metadata.get("_hybrid_score", 0.0), reverse=True)[
            :top_k
        ]

    def _lexical_search(self, query: str, top_k: int, where: dict[str, Any]) -> list[tuple[float, MemoryItem]]:
        items = self._list_items(where=where)
        query_terms = tokenize(query)
        if not query_terms:
            return [(0.0, item) for item in items[:top_k]]

        documents = [tokenize(item.content) for item in items]
        document_frequency = Counter(term for terms in documents for term in set(terms))
        scored: list[tuple[float, MemoryItem]] = []
        total_docs = max(len(items), 1)

        for item, terms in zip(items, documents, strict=True):
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                if not counts[term]:
                    continue
                idf = math.log((total_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5) + 1.0)
                score += idf * (counts[term] / (counts[term] + 1.5))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def _passes_filters(
        self,
        metadata: dict[str, Any],
        categories: list[str] | None,
        authors: list[str] | None,
        published_year: str | int | None,
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        if categories and not set(categories).intersection(set(metadata.get("categories", []))):
            return False
        if authors:
            paper_authors = {str(author).lower() for author in metadata.get("authors", [])}
            if not any(author.lower() in paper_authors for author in authors):
                return False
        if published_year is not None and not str(metadata.get("published_date", "")).startswith(str(published_year)):
            return False
        if metadata_filter:
            for key, expected in metadata_filter.items():
                value = metadata.get(key)
                if isinstance(value, list):
                    if expected not in value:
                        return False
                elif value != expected:
                    return False
        return True

    def _rerank_boost(self, query: str, paper: dict[str, Any]) -> float:
        query_terms = set(tokenize(query))
        title_terms = set(tokenize(paper.get("title") or ""))
        abstract_terms = set(tokenize(paper.get("abstract") or ""))
        boost = 0.0
        if query_terms & title_terms:
            boost += 0.2
        if query_terms & abstract_terms:
            boost += 0.1
        if paper.get("arxiv_id") and paper["arxiv_id"].lower() in query.lower():
            boost += 0.5
        return boost

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
