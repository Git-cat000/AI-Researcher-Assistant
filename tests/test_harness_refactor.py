"""Regression tests for the LLM + harness refactor."""

import importlib

from ai_researcher_assistant.harness.parsing import (
    extract_action,
    extract_final_answer,
    extract_json_block,
    extract_thought,
)
from ai_researcher_assistant.llm import BaseLLM, LLMResponse
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.orchestration import ResearcherAgent
from ai_researcher_assistant.skills import SkillRegistry
from ai_researcher_assistant.skills.builtin.paper_writer import PaperWriterSkill
from ai_researcher_assistant.skills.loader import SkillLoader


def test_basic_import_does_not_require_optional_provider_sdks():
    package = importlib.import_module("ai_researcher_assistant")

    assert package.BaseLLM is not None
    assert package.ShortTermMemory is not None


def test_harness_parsing_extracts_react_parts():
    text = """Thought: Search first.
Action:
```json
{"skill": "arxiv_fetcher", "parameters": {"query": "rag"}}
```
"""

    assert extract_thought(text) == "Search first."
    assert extract_action(text) == {"skill": "arxiv_fetcher", "parameters": {"query": "rag"}}
    assert extract_json_block("[1, 2]") == [1, 2]
    assert extract_final_answer("Thought: done\nFinal Answer: Finished") == "Finished"


def test_skill_registry_instances_are_isolated():
    first = SkillRegistry()
    second = SkillRegistry()

    first.register(PaperWriterSkill())

    assert first.list_skills() == ["paper_writer"]
    assert second.list_skills() == []


def test_paper_writer_returns_prompt_without_llm_call():
    skill = PaperWriterSkill()

    result = skill.execute(
        {
            "action": "polish",
            "text": "we found the method works.",
            "field": "machine learning",
        },
        context={},
    )

    assert result["success"] is True
    assert result["result"]["requires_llm"] is True
    assert "prompt" in result["result"]
    assert "machine learning" in result["result"]["prompt"]


def test_loads_claude_codex_style_markdown_skill(tmp_path):
    skill_dir = tmp_path / "literature-review"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: literature-review
description: Build a careful academic literature review plan.
tags: [research, writing]
---
# Literature Review

Use this skill to plan a grounded literature review with claims, sources, and gaps.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry()
    loaded = SkillLoader(registry).load_from_directory(str(skill_dir))

    assert len(loaded) == 1
    assert registry.list_skills() == ["literature-review"]
    result = registry.execute("literature-review", {"topic": "RAG"}, {})
    assert result["success"] is True
    assert result["result"]["requires_llm"] is True
    assert "grounded literature review" in result["result"]["instructions"]


def test_rag_local_memory_add_search_context_and_deduplicate():
    rag = AcademicRAG(chunk_size=120, chunk_overlap=20)

    first_ids = rag.add_paper(
        title="Retrieval Augmented Generation for Scientific Discovery",
        abstract="A paper about retrieval augmented generation for scientific literature search.",
        full_text="RAG combines retrieval with generation. It improves grounded answers for research assistants.",
        arxiv_id="2401.00001",
        authors=["A. Researcher"],
        categories=["cs.CL"],
    )
    second_ids = rag.add_paper(
        title="Retrieval Augmented Generation for Scientific Discovery",
        abstract="Updated abstract about grounded retrieval augmented generation.",
        arxiv_id="2401.00001",
        authors=["A. Researcher"],
        categories=["cs.CL"],
    )

    assert first_ids
    assert second_ids
    assert rag.count_papers() == 1

    results = rag.search_papers("grounded retrieval generation", top_k=3)
    assert len(results) == 1
    assert results[0]["arxiv_id"] == "2401.00001"

    context = rag.build_context("grounded retrieval generation", include_full_text=True)
    assert "Retrieval Augmented Generation" in context
    assert "2401.00001" in context


class ScriptedLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__(model="scripted")
        self.responses = list(responses)

    def generate(self, messages, **kwargs):
        return LLMResponse(content=self.responses.pop(0), model="scripted")

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)

    async def stream_generate(self, messages, **kwargs):
        yield self.generate(messages, **kwargs).content


def test_agent_flow_with_markdown_skill(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        """---
name: literature-review
description: Build a careful academic literature review plan.
---
Use sources, compare claims, and identify gaps.
""",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    SkillLoader(registry).load_from_markdown(skill_file)

    llm = ScriptedLLM(
        [
            """Thought: I should use the literature-review skill.
Action:
```json
{"skill": "literature-review", "parameters": {"topic": "RAG"}}
```""",
            (
                "Thought: I now have enough information to answer the question.\n"
                "Final Answer: Markdown skill flow completed."
            ),
        ]
    )
    agent = ResearcherAgent(
        llm=llm,
        enable_builtin_skills=False,
        enable_rag=False,
        skill_registry=registry,
    )

    answer = agent.process("Plan a literature review about RAG.")

    assert answer == "Markdown skill flow completed."
