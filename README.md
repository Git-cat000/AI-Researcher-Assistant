# AI Researcher Assistant

AI Researcher Assistant is a modular, LLM-driven framework for academic research workflows. It is designed to search arXiv, read papers, maintain a paper knowledge base with RAG, and assist with academic writing through a pluggable skill system.

The project follows a harness-first design principle inspired by `learn-claude-code`: the model makes decisions, while the code supplies stable tools, memory, permissions, observations, and execution loops. Repository guidance follows the `AGENTS.md` convention: instructions for coding agents should be explicit, executable, and kept close to the code.

## What It Does

- Searches arXiv papers through built-in skills.
- Reads and extracts PDF text and sections.
- Stores paper abstracts and chunks in a ChromaDB-backed long-term memory.
- Builds academic RAG context for LLM calls.
- Supports OpenAI, Anthropic, and Ollama adapters.
- Supports OpenAI-compatible provider aliases such as DeepSeek, OpenRouter, SiliconFlow, Qwen, and Azure OpenAI through `base_url`.
- Runs tasks through ReAct, Plan-and-Execute, LLMCompiler-style, or graph-based orchestration.
- Lets developers add custom skills as Python classes.

## Project Status

This repository is an early alpha. The current implementation is useful as an architecture prototype, but several issues are known:

- Markdown documentation had encoding corruption and duplicated content.
- Tests do not run in a fresh environment until dependencies are installed.
- Importing the package can eagerly import optional LLM provider packages.
- Global mutable configuration and singleton skill registry reduce test isolation.
- Some tools call the LLM internally, which violates the harness boundary.
- Several parsing and error-handling paths need hardening.

Chinese documentation is available in `README.zh-CN.md`, and the detailed technical design is in `TECHNICAL_DESIGN.zh-CN.md`.

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

Copy the environment template when using real model providers:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Then set the provider keys you need:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
```

## Quick Start

```python
import asyncio

from ai_researcher_assistant.orchestration import ResearcherAgent


async def main() -> None:
    agent = ResearcherAgent(name="Research Assistant")
    agent.initialize()

    answer = await agent.aprocess(
        "Find two recent papers about retrieval-augmented generation and summarize them."
    )
    print(answer)

    agent.shutdown()


asyncio.run(main())
```

## Commands

```bash
pip install -e ".[dev]"          # Install package with development dependencies
pytest tests/ -v                 # Run all tests
pytest tests/test_basic.py -v    # Run one test file
black ai_researcher_assistant/   # Format code, line length 120
ruff check ai_researcher_assistant/  # Lint selected rules
```

## Repository Layout

```text
ai_researcher_assistant/
  core/             Base classes, config dataclasses, messages, exceptions
  llm/              OpenAI, Anthropic, and Ollama adapters plus factory helpers
  memory/           Short-term memory, ChromaDB long-term memory, academic RAG
  skills/           Skill base classes, registry, loader, built-in skills
  orchestration/    Agent, loops, execution state, graph, middleware
examples/           Example research-assistant workflows
tests/              Unit tests using mock LLMs
```

## Architecture

```text
User task
  -> ResearcherAgent.aprocess()
  -> context construction
  -> execution loop
  -> skill call or graph node
  -> observation
  -> final answer
```

The intended dependency direction is:

```text
Agent loop -> LLM adapters
Agent loop -> tools/skills
Agent loop -> memory/RAG
Tools/skills must not call the LLM directly
```

## Model Providers

The core model layer uses `BaseLLM` and `create_llm()`. Supported provider values include:

```text
openai, openai_compatible, azure_openai, deepseek, openrouter,
siliconflow, qwen, anthropic, ollama, local
```

OpenAI-compatible providers share the `OpenAILLM` adapter and can be configured with `base_url`.

## Adding A Skill

The project now supports Claude Code / Codex style Markdown skills directly. A skill can be a folder containing `SKILL.md`, or a standalone `.md` file. `SKILL.md` should use YAML frontmatter followed by Markdown instructions:

```markdown
---
name: literature-review
description: Build a grounded academic literature review plan when the user asks for survey, comparison, or related-work help.
---

# Literature Review

Use sources, compare claims, identify gaps, and produce a structured review plan.
```

Load Markdown skills with:

```python
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry

registry = SkillRegistry()
loader = SkillLoader(registry)
loader.load_from_directory("./my_skills")  # discovers **/SKILL.md and standalone .md files
```

Python skills are still supported. Create a subclass of `BaseSkill` and implement `_build_manifest()` plus `execute()`.

```python
from ai_researcher_assistant.skills import BaseSkill, SkillManifest, SkillParameter


class MySearchSkill(BaseSkill):
    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_search",
            description="Searches a local academic source.",
            parameters=[
                SkillParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Search query",
                )
            ],
        )

    def execute(self, parameters, context):
        query = parameters["query"]
        return {"success": True, "result": {"query": query, "items": []}, "error": None}
```

Use `ai_researcher_assistant/skills/templates/custom_skill_template.md` as the documentation template for custom skills.

## Local RAG

`AcademicRAG` now defaults to a dependency-free in-memory vector backend with deterministic hash embeddings. This makes local tests and demos work without OpenAI embeddings or ChromaDB. Persistent ChromaDB-backed RAG is still available by passing `prefer_persistent=True`.

```python
from ai_researcher_assistant.memory import AcademicRAG

rag = AcademicRAG()
rag.add_paper(
    title="Retrieval Augmented Generation for Scientific Discovery",
    abstract="A paper about RAG for literature search.",
    arxiv_id="2401.00001",
)

print(rag.search_papers("grounded retrieval", top_k=1))
```

## Testing Notes

The test suite is intended to use mock LLMs and should not require live API keys. If tests fail during import with missing provider packages, install development dependencies first:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT. See `LICENSE`.
