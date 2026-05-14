# CLAUDE.md

This file is the Claude Code entry point for AI Researcher Assistant.

Use `AGENTS.md` as the source of truth for repository commands, architecture boundaries, style rules, and known issues. This file exists for compatibility with Claude Code workflows and should stay intentionally small.

## Working Rules

- Read `AGENTS.md` before making code changes.
- Preserve the harness principle: the model is the agent, and the code is the harness.
- Keep skills as tools; do not move LLM decision-making into skills.
- Prefer narrow, test-backed changes.
- Update documentation when behavior, commands, or architecture change.

## Common Commands

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/test_harness_refactor.py -v
ruff format ai_researcher_assistant/ tests/
ruff check ai_researcher_assistant/ tests/
mypy ai_researcher_assistant/
python -m compileall ai_researcher_assistant tests examples
```

## Current Caveat

This repository is in alpha. Six known mypy errors remain in `llm/openai.py` (OpenAI SDK type-stub incompatibility with `list[dict[str, str]]`) — these are pre-existing and do not affect runtime. Anthropic and ChromaDB SDKs are optional dependencies; install with `pip install -e ".[all]"` to get full provider and vector-store support.
