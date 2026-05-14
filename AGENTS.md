# AGENTS.md

This file is the operating guide for coding agents working in this repository. Keep it short, exact, and executable. If implementation details change, update this file in the same change.

## Purpose

AI Researcher Assistant is an academic research agent framework. It combines LLM adapters, memory/RAG, tool-like skills, and orchestration loops to support paper discovery, PDF reading, literature memory, and academic writing assistance.

The central design rule is:

```text
The model is the agent. The code is the harness.
```

The harness provides tools, memory, observations, permissions, and execution structure. Business logic that decides what to do next belongs in the agent loop, not inside tools.

## Commands

```bash
pip install -e ".[dev]"                          # Install with development dependencies
pytest tests/ -v                                 # Run all tests
pytest tests/test_harness_refactor.py -v         # Run refactor regression tests
ruff format ai_researcher_assistant/ tests/      # Format code, line length 120
ruff check ai_researcher_assistant/ tests/       # Lint code
mypy ai_researcher_assistant/                    # Type check
python -m compileall ai_researcher_assistant tests examples  # Syntax/import compilation check
python -m pip check                              # Verify no broken dependencies
```

## Current Structure

```text
ai_researcher_assistant/
  core/
    base_agent.py       Base synchronous/asynchronous agent interface
    config.py           Dataclass configuration, lazy env loading
    exceptions.py       AgentError hierarchy
    message.py          Message, Conversation, token helpers
  harness/
    context.py          AgentContext TypedDict contract
    parsing.py          JSON/action/thought/final-answer extraction
    schema.py           Action, Observation, PermissionPolicy, TokenBudget
  llm/
    base.py             BaseLLM and LLMResponse
    openai.py           OpenAI-compatible adapter (lazy import)
    anthropic.py        Anthropic adapter (lazy import)
    local.py            Ollama adapter
    factory.py          LLM creation helpers (lazy provider imports)
  memory/
    base.py             Base memory and embedding abstractions
    short_term.py       Sliding-window conversation memory
    long_term.py        ChromaDB-backed vector memory (lazy import)
    in_memory.py        Dependency-free vector memory and hash embeddings
    rag.py              Paper-oriented RAG with local-memory default
  skills/
    base.py             BaseSkill and manifest types
    registry.py         Skill registry (plain instance, not singleton)
    loader.py           Skill loading helpers (Python + Markdown)
    markdown.py         Claude Code / Codex compatible Markdown skills
    builtin/            Built-in arXiv, PDF, and writing skills
    templates/          Custom skill template
  orchestration/
    agent.py            ResearcherAgent facade (lazy RAG/LLM init)
    loop.py             ReAct, Plan-and-Execute, LLMCompiler loops
    graph.py            LangGraph-like workflow graph
    middleware.py       Lifecycle hooks and telemetry
    state.py            Execution state and step records
```

## Desired Harness Boundaries

- `harness/` owns stable contracts: AgentContext, parsing, Action/Observation schemas, permissions, and token budgets.
- `orchestration/` owns task planning, LLM calls, and execution loops. Shared loop mechanics live in BaseLoop.
- `skills/` are tools. They validate inputs, perform deterministic or external work, and return structured observations. Never call an LLM.
- `memory/` stores and retrieves context. It should not decide task strategy.
- `llm/` adapts providers behind `BaseLLM`. Provider SDKs are imported lazily. Provider-specific details stay inside adapters.
- `core/` contains stable data contracts, config, message types, and exceptions.

## Skill Files

The skill loader must accept Claude Code and Codex style Markdown skills:

```text
my-skill/
  SKILL.md
  references/
  scripts/
  assets/
```

`SKILL.md` uses YAML frontmatter with at least `name` and `description`, followed by Markdown instructions. The project also accepts standalone `.md` files as Markdown skills. Markdown skills return instructions and resource paths as structured observations; they do not execute arbitrary code or call the LLM directly.

## Model Providers

Provider values currently supported by config and factory:

```text
openai, openai_compatible, azure_openai, deepseek, openrouter,
siliconflow, qwen, anthropic, ollama, local
```

OpenAI-compatible providers route through `OpenAILLM` with `base_url`. Anthropic and Ollama use dedicated adapters. Do not put provider-specific task strategy in adapters.

## Do Not

- Do not call an LLM from inside a skill. Return a prompt or structured observation and let the loop decide.
- Do not add new global mutable singletons.
- Do not open ChromaDB, network clients, or files at module import time.
- Do not use bare `except:`.
- Do not swallow exceptions without logging or returning a structured error.
- Do not add broad `dict[str, Any]` context keys without documenting them.
- Do not add dependencies without updating `pyproject.toml`, docs, and tests.
- Do not rewrite public import paths casually; preserve compatibility or add a migration note.

## Ask First

- Database schema or persistence layout changes.
- Public API removals or renames.
- New required runtime dependencies.
- Changes that require live API keys in tests.
- Large directory reorganizations.

## Refactor Status

- Fixed: `core/config.py` no longer loads `.env` or creates directories at import time.
- Fixed: `SkillRegistry` is an ordinary instance (singleton removed).
- Fixed: `PaperWriterSkill` returns a structured prompt instead of calling the LLM.
- Fixed: `orchestration/graph.py` uses `harness/parsing.py` for JSON extraction (no bare `except:`).
- Fixed: Optional LLM SDKs (`anthropic`, `chromadb`) are now optional dependencies.
- Fixed: `LongTermMemory`, `OpenAIEmbedding`, `AcademicRAG` import ChromaDB lazily.
- Fixed: `core/message.py` implements token-aware context window truncation.
- Fixed: `memory/rag.py::count_papers()` counts unique paper keys.
- Fixed: Shared loop utilities (`_init_conversation`, `_handle_loop_error`, `_synthesize_final_answer`) extracted to `BaseLoop`.
- Fixed: `extract_final_answer` handles multiple `Final Answer:` markers correctly.
- By design: `ResearcherAgent.process()` raises inside an active event loop; async callers must use `await aprocess()`.
- By design: `orchestration/agent.py` creates the LLM during `initialize()` for lazy startup.

## Style

- Python version: 3.10+.
- Prefer explicit dependency injection over globals.
- Prefer `str | None` and `list[dict[str, Any]]` over older `Optional`/`List` style in new code.
- Keep public functions typed.
- Keep comments short and useful.
- Use structured return values from skills:

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "message"}
```

## Testing

- Unit tests should use mock LLMs and no live API keys.
- Prefer `InMemoryVectorMemory` or temporary persistence for memory tests.
- Add at least one error-path test for each new component.
- Run `python -m compileall ai_researcher_assistant tests examples` after broad edits.
- Run `pytest tests/ -v` when dependencies are installed.

## Documentation Rules

- `README.md` is for users and contributors.
- `AGENTS.md` is for coding agents and must be direct, scoped, and command-oriented.
- `CLAUDE.md` is a Claude Code compatibility entry point and should point back to `AGENTS.md`.
- Skill templates belong under `ai_researcher_assistant/skills/templates/`.
- Keep docs in UTF-8.
