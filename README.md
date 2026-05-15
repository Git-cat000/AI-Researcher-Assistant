# AI Researcher Assistant

AI Researcher Assistant is a modular academic research-agent harness. It combines model adapters, memory/RAG, tool-like skills, and orchestration loops for paper discovery, PDF reading, literature memory, and academic writing assistance.

The core rule is:

```text
The model is the agent. The code is the harness.
```

The model decides what to do next. The harness supplies stable contracts, tools, memory, observations, permissions, token budgets, and execution structure.

Chinese documentation is available in [README.zh-CN.md](README.zh-CN.md). The detailed technical design and roadmap are in [TECHNICAL_DESIGN.zh-CN.md](TECHNICAL_DESIGN.zh-CN.md).

## Current Capabilities

- Runs ReAct, Plan-and-Execute, LLMCompiler-style, and graph orchestration loops.
- Supports OpenAI, Anthropic, Ollama, and OpenAI-compatible providers including Azure OpenAI, DeepSeek, OpenRouter, SiliconFlow, and Qwen/DashScope.
- Exposes a Typer-based CLI through the `ai-researcher` console script and `python -m ai_researcher_assistant.cli`.
- Loads Python skills and Claude Code / Codex compatible Markdown skills.
- Accepts Markdown skill folders with `SKILL.md`, `references/`, `scripts/`, and `assets/`.
- Parses Markdown skill parameter schemas from common frontmatter shapes.
- Keeps skills deterministic: skills return structured observations and do not call the LLM.
- Provides local dependency-free RAG with hash embeddings and optional ChromaDB persistence.
- Supports hybrid vector + BM25-style retrieval, filters, lightweight reranking, citation metadata, and citation graph export.
- Tracks harness-level actions, observations, permissions, token budgets, and model token usage.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional providers:

```bash
pip install -e ".[dev,chromadb,anthropic]"
pip install -e ".[dev,all]"
```

Copy the environment template when using real model providers:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

## CLI

The project uses Typer for a typed, composable command-line interface. The CLI stays thin and delegates to the harness instead of embedding agent strategy in command handlers.

```bash
ai-researcher version --json
ai-researcher doctor --json
ai-researcher providers --json
ai-researcher skills list --json
ai-researcher skills inspect ./my-skills/literature-review/SKILL.md --json
ai-researcher rag search "hybrid retrieval" --paper-jsonl papers.jsonl --json
ai-researcher rag graph --paper-jsonl papers.jsonl --json
ai-researcher ask "Find papers about RAG for scientific discovery" --provider openai --model gpt-4o
```

The same CLI can be run without relying on PATH:

```bash
python -m ai_researcher_assistant.cli doctor --json
```

`rag search` expects one JSON object per line:

```json
{"title":"Hybrid Retrieval for Research Agents","abstract":"...","arxiv_id":"2401.10000","authors":["A. Researcher"],"categories":["cs.IR"],"published_date":"2024-01-01","citations":["2301.00001"]}
```

## Python Quick Start

```python
import asyncio

from ai_researcher_assistant.orchestration import ResearcherAgent


async def main() -> None:
    agent = ResearcherAgent(name="Research Assistant")
    answer = await agent.aprocess(
        "Find two recent papers about retrieval-augmented generation and summarize them."
    )
    print(answer)
    agent.shutdown()


asyncio.run(main())
```

## Markdown Skills

A Claude Code / Codex compatible Markdown skill can be a directory with `SKILL.md`:

```text
my-skills/
  literature-review/
    SKILL.md
    references/
    scripts/
    assets/
```

Example:

```markdown
---
name: literature-review
description: Build a grounded literature review plan.
parameters:
  topic:
    type: string
    description: Research topic
    required: true
tags: [research, writing]
---

# Literature Review

Compare claims, methods, evidence, limitations, and open research gaps.
```

Markdown skills return instructions, metadata, resource paths, and optionally requested reference contents. `scripts/` paths are exposed as resources but are not executed by default.

## Local RAG

```python
from ai_researcher_assistant.memory import AcademicRAG

rag = AcademicRAG()
rag.add_paper(
    title="Retrieval Augmented Generation for Scientific Discovery",
    abstract="A paper about RAG for scientific literature search.",
    arxiv_id="2401.00001",
    authors=["A. Researcher"],
    categories=["cs.CL"],
    citations=["2301.00001"],
)

print(rag.search_papers("grounded retrieval", retrieval_mode="hybrid"))
print(rag.citation_graph())
```

## Repository Layout

```text
ai_researcher_assistant/
  core/             Stable config, messages, exceptions, base agent types
  harness/          Action, Observation, PermissionPolicy, TokenBudget, parsing
  llm/              Provider adapters, factory helpers, model capability metadata
  memory/           Short-term memory, vector memory, ChromaDB backend, academic RAG
  skills/           Skill base classes, registry, loader, Markdown compatibility, builtin skills
  orchestration/    Agent facade, execution loops, graph, state, middleware
examples/           Example workflows
tests/              Unit and regression tests with mock LLMs
```

## Development Commands

```bash
python -m ruff format ai_researcher_assistant tests examples
python -m ruff check ai_researcher_assistant tests examples
python -m mypy ai_researcher_assistant
python -m compileall ai_researcher_assistant tests examples
python -m pytest tests/ -v
python -m pip check
python -m build
```

Or:

```bash
make all
```

## License

MIT. See [LICENSE](LICENSE).
