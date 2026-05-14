"""Typed context passed through the LLM harness."""

from typing import Any, TypedDict


class AgentContext(TypedDict, total=False):
    """Known context keys used by loops, tools, memory, and graph nodes."""

    llm: Any
    conversation: Any
    short_term_memory: Any
    rag: Any
    config: Any
    skill_registry: Any
    execution_state: Any
    plan_results: dict[str, Any]
    state: Any
