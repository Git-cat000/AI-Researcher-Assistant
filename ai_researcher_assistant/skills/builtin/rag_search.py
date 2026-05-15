"""Local RAG search skill."""

from __future__ import annotations

from typing import Any

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class RagSearchSkill(BaseSkill):
    """Search the current local academic RAG memory."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="rag_search",
            description="Search the current local academic RAG memory for relevant papers and context.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(name="query", description="Search query", type="string", required=True),
                SkillParameter(
                    name="top_k",
                    description="Number of papers to return",
                    type="integer",
                    required=False,
                    default=5,
                ),
                SkillParameter(
                    name="include_full_text",
                    description="Whether to include full-text excerpts in the returned context",
                    type="boolean",
                    required=False,
                    default=False,
                ),
            ],
            tags=["academic", "rag", "memory", "search"],
            instructions=(
                "Use this skill to search the local paper memory. It returns paper-level results and compact context. "
                "It does not call an LLM."
            ),
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        rag = context.get("rag")
        if rag is None:
            return {"success": False, "result": None, "error": "No local RAG memory is available in context"}

        query = parameters["query"]
        top_k = int(parameters.get("top_k", 5))
        include_full_text = bool(parameters.get("include_full_text", False))
        results = rag.search_papers(query, top_k=top_k)
        return {
            "success": True,
            "result": {
                "papers": results,
                "context": rag.build_context(query, top_k=top_k, include_full_text=include_full_text),
                "paper_count": rag.count_papers(),
            },
            "error": None,
        }
