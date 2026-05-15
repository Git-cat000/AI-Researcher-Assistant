"""Tests for CLI, harness schema, and completed planning items."""

import json

import pytest
from typer.testing import CliRunner

from ai_researcher_assistant.cli import app
from ai_researcher_assistant.harness.schema import Action, CostTracker, PermissionPolicy, TokenBudget
from ai_researcher_assistant.llm import LLMResponse, get_model_capability
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry


def test_cli_version_and_providers():
    runner = CliRunner()

    version_result = runner.invoke(app, ["version", "--json"])
    assert version_result.exit_code == 0
    assert json.loads(version_result.stdout)["version"]

    providers_result = runner.invoke(app, ["providers", "--json"])
    assert providers_result.exit_code == 0
    providers = json.loads(providers_result.stdout)
    assert any(provider["provider"] == "openai" for provider in providers)


def test_cli_rag_search_jsonl(tmp_path):
    paper_jsonl = tmp_path / "papers.jsonl"
    paper_jsonl.write_text(
        json.dumps(
            {
                "title": "Hybrid Retrieval for Research Agents",
                "abstract": "BM25 and vector retrieval improve grounded research assistants.",
                "arxiv_id": "2401.10000",
                "authors": ["A. Researcher"],
                "categories": ["cs.IR"],
                "published_date": "2024-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["rag", "search", "hybrid retrieval", "--paper-jsonl", str(paper_jsonl), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["results"][0]["arxiv_id"] == "2401.10000"


def test_markdown_skill_schema_resource_policy(tmp_path):
    skill_dir = tmp_path / "skill"
    refs = skill_dir / "references"
    scripts = skill_dir / "scripts"
    refs.mkdir(parents=True)
    scripts.mkdir()
    (refs / "note.md").write_text("Grounded note", encoding="utf-8")
    (scripts / "run.py").write_text("print('no auto execution')", encoding="utf-8")
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: literature-review
description: Build a literature review.
parameters:
  topic:
    type: string
    description: Research topic
    required: true
---
Use references and compare claims.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry()
    skill = SkillLoader(registry).load_from_markdown(skill_file)
    result = skill.execute({"topic": "RAG", "read_resources": ["references/note.md", "scripts/run.py"]}, {})

    assert skill.manifest.parameters[0].name == "topic"
    assert result["result"]["script_policy"]["can_execute"] is False
    assert any("note.md" in path for path in result["result"]["resource_contents"])
    assert not any("run.py" in path for path in result["result"]["resource_contents"])


def test_harness_permission_and_budget_contracts():
    action = Action.from_mapping({"skill": "paper_reader", "parameters": {"url": "x"}})
    policy = PermissionPolicy(blocked_skills={"paper_reader"})
    allowed, reason = policy.check_action(action)
    tracker = CostTracker()
    tracker.add_response(LLMResponse(content="", model="mock", usage={"input_tokens": 3, "output_tokens": 5}))

    assert allowed is False
    assert "blocked" in reason
    assert TokenBudget.from_context({"token_budget": {"max_observation_tokens": 12}}).max_observation_tokens == 12
    assert tracker.to_dict()["total_tokens"] == 8
    assert get_model_capability("deepseek").credential_env == "DEEPSEEK_API_KEY"


def test_rag_hybrid_filters_and_citation_graph():
    rag = AcademicRAG(chunk_size=120, chunk_overlap=10)
    rag.add_paper(
        title="Hybrid Retrieval for Research Agents",
        abstract="BM25 retrieval and vector search improve grounded research assistants.",
        arxiv_id="2401.10000",
        authors=["A. Researcher"],
        categories=["cs.IR"],
        published_date="2024-01-01",
        citations=["2301.00001"],
    )
    rag.add_paper(
        title="Unrelated Vision Benchmark",
        abstract="Image models and visual evaluation.",
        arxiv_id="2401.20000",
        authors=["B. Researcher"],
        categories=["cs.CV"],
        published_date="2024-02-01",
    )

    results = rag.search_papers(
        "BM25 grounded retrieval",
        categories=["cs.IR"],
        authors=["A. Researcher"],
        published_year=2024,
        retrieval_mode="hybrid",
    )
    graph = rag.citation_graph()

    assert [paper["arxiv_id"] for paper in results] == ["2401.10000"]
    assert {"source": "2401.10000", "target": "2301.00001"} in graph["edges"]
    assert rag.citation_neighbors("2401.10000")["outgoing"] == ["2301.00001"]

    with pytest.raises(ValueError):
        rag.search_papers("retrieval", retrieval_mode="unknown")
