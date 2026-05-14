"""Memory contracts and persistence helpers.

ChromaDB-backed classes are imported lazily so short-term memory tests and
basic package imports do not require vector-store dependencies.
"""

from ai_researcher_assistant.memory.base import BaseEmbedding, BaseMemory, MemoryItem
from ai_researcher_assistant.memory.in_memory import HashEmbedding, InMemoryVectorMemory
from ai_researcher_assistant.memory.short_term import ShortTermMemory


def __getattr__(name: str):
    if name in {"LongTermMemory", "OpenAIEmbedding"}:
        from ai_researcher_assistant.memory.long_term import LongTermMemory, OpenAIEmbedding

        return {"LongTermMemory": LongTermMemory, "OpenAIEmbedding": OpenAIEmbedding}[name]
    if name == "AcademicRAG":
        from ai_researcher_assistant.memory.rag import AcademicRAG

        return AcademicRAG
    raise AttributeError(name)


__all__ = [
    "BaseMemory",
    "BaseEmbedding",
    "MemoryItem",
    "HashEmbedding",
    "InMemoryVectorMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "OpenAIEmbedding",
    "AcademicRAG",
]
