# Technical Plan

This document describes the target technical direction for AI Researcher Assistant after the first LLM + harness refactor pass.

## Product Goal

AI Researcher Assistant should be a reliable academic research harness that lets an LLM plan, call tools, observe results, retrieve paper knowledge, and produce grounded research outputs.

Primary workflows:

- Search academic papers.
- Read and extract PDF content.
- Build a paper knowledge base.
- Retrieve paper context with RAG.
- Draft, polish, summarize, and format academic writing.
- Run with cloud models or local models.

## Core Principle

```text
The model is the agent. The code is the harness.
```

The harness provides:

- model adapters,
- memory and RAG,
- tools and skill manifests,
- action parsing,
- observation formatting,
- middleware,
- execution state,
- permissions and future sandbox controls.

The harness must not hide agent decisions inside tools.

## Current Refactor Baseline

Completed in the first pass:

- Added `ai_researcher_assistant/harness/`.
- Added `AgentContext` and shared model-output parsing helpers.
- Made configuration lazy and removed `.env` loading from import time.
- Made LLM provider imports lazy.
- Made ChromaDB-backed memory imports lazy.
- Made `SkillRegistry` an ordinary instance.
- Made `ResearcherAgent` lighter and RAG lazy.
- Changed `PaperWriterSkill` so it returns a model-ready prompt instead of calling the LLM.
- Fixed `graph.py` bare `except:` and missing `json` import.
- Expanded development dependencies and pytest configuration.
- Added Claude Code / Codex style Markdown skill loading.
- Added dependency-free local vector memory and hash embeddings.
- Changed `AcademicRAG` to use local memory by default and persistent ChromaDB only when requested.

Completed in the second pass:
- Extracted shared loop helpers (`_init_conversation`, `_handle_loop_error`, `_synthesize_final_answer`) into `BaseLoop`.
- Replaced inline parsing in `graph.py` and loops with `harness/parsing.py` imports.
- Fixed `extract_final_answer` to handle multiple `Final Answer:` markers via split instead of greedy regex.
- Moved `chromadb` and `anthropic` SDKs to optional dependency groups (`[chromadb]`, `[anthropic]`, `[all]`).
- Added `Makefile` with standard targets: `install`, `format`, `lint`, `typecheck`, `test`, `test-cov`, `all`, `clean`.
- Added `.pre-commit-config.yaml` with ruff and standard pre-commit hooks.
- Added `.github/workflows/ci.yml` with Python 3.10/3.11/3.12 matrix.
- Enhanced `tests/test_basic.py` with real ReAct loop execution tests (12 total, up from 4 smoke tests).
- Fixed `B905 zip strict=`, `B027 empty method`, `SIM117 nested with`, `SIM108 ternary` ruff rules.

## Model Layer

The model layer is centered on `BaseLLM`:

```text
BaseLLM
  generate()
  agenerate()
  stream_generate()
```

Supported provider families:

| Provider | Adapter | Notes |
|---|---|---|
| `openai` | `OpenAILLM` | Official OpenAI-compatible chat API |
| `openai_compatible` | `OpenAILLM` | Any compatible endpoint through `base_url` |
| `azure_openai` | `OpenAILLM` | Uses OpenAI-compatible route and Azure key/env |
| `deepseek` | `OpenAILLM` | Uses DeepSeek/OpenAI-compatible key/env |
| `openrouter` | `OpenAILLM` | OpenRouter-compatible gateway |
| `siliconflow` | `OpenAILLM` | SiliconFlow-compatible gateway |
| `qwen` | `OpenAILLM` | DashScope/OpenAI-compatible route |
| `anthropic` | `AnthropicLLM` | Claude messages API |
| `ollama`, `local` | `OllamaLLM` | Local model server |

Provider-specific task logic should not enter adapters. Adapters only translate messages, call the provider, normalize responses, and raise `LLMError`.

## Harness Layer

Target modules:

```text
harness/
  context.py    Typed context contract
  parsing.py    JSON/action/final-answer extraction
  loop.py       Future minimal loop home
  state.py      Future execution-state home
```

The current orchestration loops remain in `orchestration/loop.py` for compatibility, but shared parsing has already moved into `harness/parsing.py`.

Near-term work:

- Move repeated loop mechanics into harness primitives.
- Make action schemas explicit.
- Add tool permission checks before execution.
- Add token and budget accounting.
- Add structured observations instead of only text observations.

## Tool Layer

Skills are tools, not agents.

Rules:

- Skills validate parameters.
- Skills perform deterministic work or external calls.
- Skills return structured results.
- Skills do not call an LLM.
- Skills do not mutate global state.

Standard result:

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "message"}
```

`PaperWriterSkill` now follows this rule by returning a prompt and `requires_llm=True`.

Markdown skill compatibility:

- A skill may be a directory with `SKILL.md`.
- A skill may also be a standalone `.md` file.
- YAML frontmatter should include `name` and `description`.
- Optional supporting folders such as `references/`, `scripts/`, and `assets/` are discovered and returned as resource paths.
- Markdown skills are model-invoked instructions. They return structured observations for the harness instead of executing hidden behavior.

## Memory And RAG

Memory responsibilities:

- `ShortTermMemory`: conversation window.
- `LongTermMemory`: vector persistence.
- `AcademicRAG`: paper-specific chunking, metadata, and retrieval context.
- `InMemoryVectorMemory`: dependency-free local backend.
- `HashEmbedding`: deterministic local embedding fallback.

Near-term work:

- Persistent ChromaDB should become an explicit backend option in docs and examples.
- Make embeddings injectable and testable.
- Add in-memory memory fixtures for tests.
- Keep OpenAI embeddings optional when only local models are used.

## Configuration

Configuration should be explicit:

```text
.env -> load_config_from_env() -> AgentConfig -> injected components
```

Importing modules should not load `.env`, create directories, or open external clients.

Current compatibility helpers:

- `get_config()`
- `update_config()`
- `get_skill_registry()`

These should remain for older code but should not be used in new internal wiring.

## Testing Plan

Immediate test goals:

- Install dev dependencies and ensure `pytest-asyncio` is active.
- Add parser unit tests for `harness/parsing.py`.
- Add `SkillRegistry` isolation tests.
- Add `PaperWriterSkill` no-LLM tests.
- Add import tests proving optional SDKs are not required for basic imports.

Current verification:

```text
python -m compileall ai_researcher_assistant tests examples
pytest tests/ -v
```

Current local result after installing `pytest-asyncio`: `11 passed`.

## Migration Plan

1. Keep public imports stable for one minor release.
2. Add new harness modules behind existing orchestration APIs. (Done: `harness/context.py`, `harness/parsing.py`, `harness/schema.py`)
3. Move logic gradually from `orchestration/` to `harness/`. (Done: parsing and loop helpers)
4. Add deprecation warnings only after replacement APIs are ready.
5. Rename `skills/buildin/` to `skills/builtin/` with compatibility shims. (Done)

## Open Risks

- Existing tests are deeper than before (12 tests, 2 loop-execution tests) but still lack coverage for ChromaDB and graph paths.
- ChromaDB and embedding setup still need better test doubles.
- Provider aliases need provider-specific documentation and examples.
- Sync wrapper behavior is intentionally stricter inside active event loops.
- `chromadb` and `anthropic` SDKs are now optional; pip install -e ".[all]" for full support.
