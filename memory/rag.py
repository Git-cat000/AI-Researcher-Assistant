"""
RAG 检索增强生成模块。
专为学术论文检索优化，支持 PDF 解析、分块存储和上下文构建。
"""
from typing import List, Dict, Any, Optional, Tuple
import re

from ai_researcher_assistant.memory.long_term import LongTermMemory, OpenAIEmbedding
from ai_researcher_assistant.memory.base import MemoryItem, BaseEmbedding
from ai_researcher_assistant.core.config import get_config


class AcademicRAG:
    """
    学术 RAG 系统。
    用于论文的存储、检索和上下文构建。
    """

    def __init__(
        self,
        collection_name: str = "academic_papers",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        config = get_config()
        self.chunk_size = chunk_size or config.memory.chunk_size
        self.chunk_overlap = chunk_overlap or config.memory.chunk_overlap
        
        self.memory = LongTermMemory(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model or OpenAIEmbedding(),
        )

    def add_paper(
        self,
        title: str,
        abstract: str,
        full_text: Optional[str] = None,
        authors: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        published_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        添加一篇论文到知识库。
        会自动将全文分块存储。
        
        Returns:
            存储的记忆项 ID 列表（可能多个分块）
        """
        # 构建元数据
        meta = metadata or {}
        meta.update({
            "title": title,
            "authors": authors or [],
            "arxiv_id": arxiv_id,
            "categories": categories or [],
            "published_date": published_date,
            "type": "paper",
        })
        
        items_to_add = []
        
        # 1. 存储摘要（作为单独的一条）
        if abstract:
            items_to_add.append(MemoryItem(
                content=f"Title: {title}\nAbstract: {abstract}",
                metadata={**meta, "section": "abstract"},
            ))
        
        # 2. 全文分块存储
        if full_text:
            chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                items_to_add.append(MemoryItem(
                    content=chunk,
                    metadata={**meta, "section": f"full_text_chunk_{i}", "chunk_index": i},
                ))
        
        return self.memory.add_batch(items_to_add)

    def search_papers(
        self,
        query: str,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索相关论文。
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            categories: 按 arXiv 分类过滤（如 ["hep-th", "quant-ph"]）
            arxiv_id: 精确匹配 arXiv ID
        
        Returns:
            论文信息列表，包含元数据和相似度
        """
        # 构建过滤条件
        where = {"type": "paper"}
        if categories:
            # ChromaDB 的 $in 操作符
            where["$and"] = [{"categories": {"$contains": cat}} for cat in categories]
        if arxiv_id:
            where["arxiv_id"] = arxiv_id
        
        results = self.memory.search(query, top_k=top_k, where=where)
        
        # 聚合同一论文的不同分块，去重
        papers = {}
        for item in results:
            paper_id = item.metadata.get("arxiv_id") or item.metadata.get("title")
            if paper_id not in papers:
                papers[paper_id] = {
                    "title": item.metadata.get("title"),
                    "authors": item.metadata.get("authors", []),
                    "arxiv_id": item.metadata.get("arxiv_id"),
                    "categories": item.metadata.get("categories", []),
                    "published_date": item.metadata.get("published_date"),
                    "chunks": [],
                    "abstract": None,
                    "score": item.metadata.get("_distance", 1.0),
                }
            
            if item.metadata.get("section") == "abstract":
                papers[paper_id]["abstract"] = item.content
            else:
                papers[paper_id]["chunks"].append({
                    "content": item.content,
                    "score": item.metadata.get("_distance", 1.0),
                })
        
        # 按分数排序
        sorted_papers = sorted(papers.values(), key=lambda x: x["score"])
        return sorted_papers[:top_k]

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 4000,
        include_full_text: bool = False,
        **filters
    ) -> str:
        """
        构建用于 LLM 的上下文文本。
        
        Args:
            query: 查询
            top_k: 检索论文数
            max_tokens: 最大 token 数（粗略）
            include_full_text: 是否包含全文分块
            **filters: 额外过滤条件
        
        Returns:
            格式化的上下文字符串
        """
        papers = self.search_papers(query, top_k=top_k, **filters)
        
        context_parts = []
        current_tokens = 0
        
        for paper in papers:
            paper_text = f"### {paper['title']}\n"
            if paper['authors']:
                paper_text += f"Authors: {', '.join(paper['authors'])}\n"
            if paper['arxiv_id']:
                paper_text += f"arXiv: {paper['arxiv_id']}\n"
            if paper['categories']:
                paper_text += f"Categories: {', '.join(paper['categories'])}\n"
            if paper['abstract']:
                paper_text += f"\nAbstract: {paper['abstract']}\n"
            
            if include_full_text and paper['chunks']:
                paper_text += "\nExcerpts:\n"
                for chunk in paper['chunks'][:3]:  # 最多 3 个分块
                    paper_text += f"- {chunk['content'][:500]}...\n"
            
            paper_text += "\n---\n"
            
            # 粗略 token 估算
            tokens = len(paper_text) // 4
            if current_tokens + tokens > max_tokens:
                break
            
            context_parts.append(paper_text)
            current_tokens += tokens
        
        return "".join(context_parts)

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本分割为重叠的分块"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # 尽量在句子边界切割
            if end < len(text):
                # 寻找最近的句号、问号或感叹号
                for sep in ['. ', '? ', '! ', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + 1
                        break
            chunk = text[start:min(end, len(text))]
            chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks

    def count_papers(self) -> int:
        """统计知识库中的论文数（去重）"""
        # 简单实现：获取所有记忆，统计唯一 arxiv_id
        # 实际可用 ChromaDB 的 distinct 查询
        return self.memory.count() // 5  # 粗略估计
