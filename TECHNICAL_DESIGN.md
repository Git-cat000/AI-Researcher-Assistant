# AI Researcher Assistant Technical Design

This document describes the current architecture, implementation details, CLI session model, PDF ingestion flow, non-arXiv source extension strategy, quality gates, and roadmap for AI Researcher Assistant.

The core rule is:

```text
The model is the agent. The code is the harness.
```

The model reasons, plans, chooses skills, and writes final answers. The harness provides stable execution contracts, context, tools, memory, permissions, observations, and budget control.

## 1. Architecture

```text
User task
  -> CLI / CLI session / Python API
  -> ResearcherAgent.aprocess()
  -> AgentContext
  -> Loop(ReAct / Plan-and-Execute / LLMCompiler)
  -> LLM emits Thought / Action / Final Answer
  -> Harness parses Action and checks permissions
  -> SkillRegistry invokes Skill
  -> Optional: subagent_task creates a local child agent and returns summary
  -> Observation is appended back to the conversation
  -> LLM emits final answer
```

Dependency direction:

```text
orchestration -> llm
orchestration -> skills
orchestration -> memory
orchestration -> harness
skills do not call llm
memory does not decide task strategy
llm adapters do not contain business strategy
cli does not contain agent strategy
```

The project borrows the design philosophy of `learn-claude-code` s04-s12 and maps it into this academic-research harness:

```text
s04 subagent isolation        -> SubagentRunner + subagent_task
s05 on-demand skill knowledge -> compact skill catalog + Markdown Skill observations
s06 context compaction        -> ContextCompactor
s07 persistent task graph     -> TaskBoard
s08 background notification   -> JSON/JSONL control plane for future background executors
s09 team mailbox              -> TeamMailbox
s10 request-response protocol -> ProtocolStore
s11 autonomous claiming       -> TaskBoard.claim_next()
s12 worktree isolation bind   -> WorktreeIndex
```

## 2. core/

`core/` stores stable foundational types.

- `config.py`: dataclass-based Agent, LLM, Memory, and Skills configuration. It does not read `.env`, create directories, or initialize SDKs at import time. `load_config_from_env()` explicitly reads environment variables.
- `message.py`: defines `MessageRole`, `Message`, and `Conversation`, with coarse token estimation and context-window truncation.
- `exceptions.py`: shared internal exception hierarchy.
- `base_agent.py`: synchronous/asynchronous agent base class.

## 3. harness/

`harness/` owns stable contracts rather than task strategy.

- `context.py`: defines `AgentContext` and known shared context keys for loops, tools, memory, and graph nodes.
- `coordination.py`: implements persistent task DAGs, team mailboxes, request-response protocols, worktree binding indexes, and deterministic context compaction.
- `parsing.py`: centralizes LLM output parsing, including JSON blocks, Actions, Thoughts, and Final Answers.
- `schema.py`: defines `Action`, `Observation`, `PermissionPolicy`, `TokenBudget`, and `CostTracker`.

`BaseLoop` currently enforces:

- Action normalization.
- Permission checks before skill execution.
- Observation formatting and token truncation.
- LLM call token budgets.
- Model token-usage tracking.

## 4. llm/

`llm/` is the provider adapter layer. It only converts unified message formats into provider-specific API calls.

- `base.py`: defines `BaseLLM` and `LLMResponse`.
- `openai.py`: adapts OpenAI-compatible Chat Completions APIs.
- `anthropic.py`: adapts Anthropic Messages APIs.
- `local.py`: calls Ollama through HTTP.
- `factory.py`: creates adapters from `LLMConfig.provider`.
- `capabilities.py`: records provider capabilities, credential environment variables, local-model flags, and OpenAI-compatible flags.

Supported providers:

```text
openai, openai_compatible, azure_openai, deepseek, openrouter,
siliconflow, qwen, anthropic, ollama, local
```

## 5. skills/

`skills/` is the tool layer. Skills perform deterministic work or external calls and return structured results:

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "message"}
```

Boundaries:

- Skills do not call the LLM.
- Skills do not decide task strategy.
- Skills do not bypass `PermissionPolicy`.
- Skills that need credentials receive them through configuration or environment variables.

Main modules:

- `base.py`: `SkillParameter`, `SkillManifest`, and `BaseSkill`.
- `registry.py`: ordinary instance registry, no global singleton dependency for new code.
- `loader.py`: loads Python classes, Python modules, directories, `SKILL.md`, standalone `.md` files, and built-in skills.
- `markdown.py`: Claude Code / Codex compatible Markdown Skill layer.
- `builtin/rag_search.py`: exposes the current `AcademicRAG` as a deterministic retrieval skill.
- `builtin/subagent_task.py`: exposes local child-agent delegation as an async skill.
- `builtin/harness_coordination.py`: exposes task graphs, mailboxes, protocols, and worktree bindings through one deterministic control-plane skill.

Markdown Skills support:

- YAML frontmatter.
- `parameters`, `params`, `input_schema`, and `schema`.
- JSON-schema-like `properties + required`.
- `references/`, `scripts/`, and `assets/` discovery.
- `read_resources` / `resource_paths` for non-`scripts/` resource reading.
- `scripts/` files are exposed as paths by default, not executed.

The system prompt now uses `SkillRegistry.get_catalog_for_all()` as a compact skill catalog instead of injecting every skill's detailed instructions into every turn. This follows the s05-style two-layer idea: the model first sees what tools exist, then asks for or receives detailed content through actions and observations only when needed.

## 6. memory/

`memory/` handles short-term memory, vector memory, and paper-level RAG.

- `short_term.py`: stores recent conversation in a `deque`, with message-count and token-count trimming.
- `in_memory.py`: provides `HashEmbedding` and `InMemoryVectorMemory` so local tests do not require external embedding services.
- `long_term.py`: ChromaDB persistence backend with lazy dependency imports.
- `rag.py`: academic-paper RAG layer.

`AcademicRAG` implements:

- `add_paper()`
- `search_papers()`
- `build_context()`
- `delete_paper()`
- `update_paper()`
- `count_papers()`
- `citation_graph()`
- `citation_neighbors()`

Retrieval modes:

- `vector`
- `keyword`
- `bm25`
- `hybrid`

Filters:

- `categories`
- `arxiv_id`
- `authors`
- `published_year`
- `section`
- `metadata_filter`

## 7. orchestration/

`orchestration/` assembles models, skills, memory, and harness contracts into executable agent flows.

- `agent.py`: `ResearcherAgent` main entry point. Supports injected config, mock LLMs, independent `SkillRegistry`, and lazy RAG/LLM initialization.
- `loop.py`: implements `ReActLoop`, `PlanAndExecuteLoop`, and `LLMCompilerLoop`, with shared mechanics in `BaseLoop`.
- `graph.py`: lightweight LangGraph-like workflow graph.
- `middleware.py`: lifecycle hooks and telemetry.
- `state.py`: execution steps, status, and metadata.
- `subagent.py`: local parent-child agent orchestration. It creates an isolated child ReAct loop and returns only a summary to the parent.

## 8. Local Parent-Child Agents

The parent-child design reuses the context-isolation idea from `learn-claude-code` s04. The parent agent does not carry every intermediate read, search, and tool result. It delegates focused work to a child agent, and the child returns only a consumable summary.

Flow:

```text
Parent ReActLoop
  -> Action: subagent_task
  -> SubagentTaskSkill.aexecute()
  -> SubagentRunner.run()
  -> child SkillRegistry(filtered)
  -> child ReActLoop(fresh Conversation)
  -> child Final Answer
  -> SubagentResult(summary, artifacts, steps_taken, error)
  -> parent Observation
```

Default roles:

```text
literature_search_agent  -> arxiv_fetcher, rag_search
paper_reading_agent      -> paper_reader, rag_search
rag_retrieval_agent      -> rag_search
writing_agent            -> paper_writer, rag_search
```

Implementation principles:

- Child agents use a fresh `Conversation` and do not inherit the full parent history.
- Child agents share the same LLM adapter and optional `AcademicRAG`, but receive only the allowed skills.
- The child `SkillRegistry` filters out `subagent_task` to avoid recursive delegation.
- The parent receives `SubagentResult.to_dict()` as the observation, including summary, allowed skills, steps, and errors.
- The current implementation is local and in-process. If MCP or A2A are added later, the summary-only contract should remain stable.

## 9. Coordination Control Plane

`harness/coordination.py` provides a local file-backed control plane rooted at `.ai-researcher/`. It does not call the LLM and does not run dangerous git operations. It only maintains recoverable state.

### 9.1 TaskBoard

`TaskBoard` stores each task as `task_<id>.json`:

```json
{
  "id": 1,
  "subject": "Read local PDFs",
  "status": "pending",
  "blocked_by": [],
  "owner": "",
  "worktree": ""
}
```

Key behavior:

- `create()` creates a durable task.
- `update(..., status="completed")` automatically clears that task id from other tasks' `blocked_by` lists.
- `ready_tasks()` returns unblocked pending tasks.
- `claim_next(owner)` supports s11-style autonomous claiming.

### 9.2 TeamMailbox

`TeamMailbox` uses append-only JSONL inboxes:

```text
.ai-researcher/team/inbox/writer.jsonl
```

Each message contains sender, recipient, message_type, request_id, content, and timestamp. `read(..., drain=True)` reads and clears the inbox, making it suitable for injection before an agent loop turn.

### 9.3 ProtocolStore

`ProtocolStore` implements a shared request-response FSM:

```text
pending -> approved
pending -> rejected
```

It can represent plan approval, graceful shutdown, handoff, or other structured negotiation flows. Requests live in `.ai-researcher/team/protocols.json`, and each request has a stable `request_id`.

### 9.4 WorktreeIndex

`WorktreeIndex` records bindings between tasks and isolated directories:

```text
.ai-researcher/worktrees/index.json
.ai-researcher/worktrees/events.jsonl
```

It writes `task_id`, worktree name, path, and status into the index, then appends lifecycle events to the event stream. The current implementation records isolation intent only; it does not directly call `git worktree add/remove`. A future git adapter can be added behind `PermissionPolicy`.

### 9.5 ContextCompactor

`ContextCompactor` provides deterministic micro-compaction for execution history. It keeps recent observations and replaces older large observations with compact placeholders. This is useful for display, logs, and summary construction without re-injecting large PDF or retrieval outputs into context.

## 10. CLI

`ai_researcher_assistant/cli.py` uses Typer and is registered in `pyproject.toml`:

```toml
[project.scripts]
ai-researcher = "ai_researcher_assistant.cli:main"
```

Commands:

- `version`: print version.
- `doctor`: check built-in skills and provider metadata.
- `providers`: list provider capabilities.
- `ask`: run one agent task.
- `chat`: start a persistent command-line session.
- `skills list`: list built-in and external skills.
- `skills inspect`: parse one Markdown Skill.
- `rag search`: build local RAG from JSONL and search it.
- `rag graph`: build a citation graph from JSONL.
- `rag ingest-pdf`: convert a local PDF or direct PDF URL into a JSONL paper record.

CLI boundary: commands only parse arguments, load inputs, and format outputs. Core behavior is delegated to `ResearcherAgent`, `AgentCliSession`, `SkillLoader`, and `AcademicRAG`.

## 11. Persistent CLI Session

`cli_session.py` adds `AgentCliSession`.

Implementation:

- Each session is bound to a `cwd`.
- Prompt history is written to `cwd/.ai-researcher/history.txt`.
- `SkillRegistry` is created per session.
- `AcademicRAG` stays resident for the session.
- `ResearcherAgent` reuses short-term memory inside the session.
- Normal text goes through `ResearcherAgent.aprocess()`.
- If the session RAG has papers, `AgentCliSession.ask()` builds compact RAG context and attaches it to the current task.
- Slash commands do not call the LLM; the session layer handles them directly.

Current slash commands:

```text
/help
/pwd
/skills
/stats
/model
/model PROVIDER MODEL
/rag load PATH
/rag ingest-pdf PATH_OR_URL
/rag search QUERY
/clear
/exit
```

Run:

```bash
ai-researcher chat --cwd .
python -m ai_researcher_assistant.cli chat --cwd .
```

## 12. Local PDF Ingestion

`rag ingest-pdf` flow:

```text
PDF path or direct PDF URL
  -> PaperReaderSkill
  -> full_text / sections / pdf metadata
  -> normalized paper record
  -> JSONL output or session RAG
  -> AcademicRAG.add_paper()
```

Recommended workflow:

```bash
ai-researcher rag ingest-pdf ./papers/example.pdf --output papers.jsonl --title "Example Paper" --author "A. Researcher" --category cs.CL
ai-researcher rag search "retrieval augmented generation" --paper-jsonl papers.jsonl --include-full-text --json
ai-researcher chat --cwd .
/rag load papers.jsonl
/rag search retrieval augmented generation
```

For non-arXiv papers with a direct PDF URL:

```bash
ai-researcher rag ingest-pdf https://example.org/paper.pdf --output papers.jsonl --title "External Paper"
```

## 13. Non-arXiv Source Extensions

Paper sources should not be hard-coded into RAG or LLM adapters. Each source should be a separate Source Skill:

```text
semantic_scholar_fetcher
crossref_fetcher
pubmed_fetcher
openalex_fetcher
publisher_pdf_fetcher
institutional_repository_fetcher
```

Source Skills should:

- Query external APIs or pages.
- Handle pagination, rate limits, and errors.
- Normalize title, abstract, authors, doi, pdf_url, landing_url, published_date, and citations.
- Return structured paper candidates.

Source Skills should not:

- Call the LLM.
- Decide research strategy.
- Mutate agent state directly.
- Bypass `PermissionPolicy` to download arbitrary resources.

Recommended return shape:

```python
{
    "success": True,
    "result": {
        "papers": [
            {
                "title": "...",
                "abstract": "...",
                "authors": ["..."],
                "source": "openalex",
                "source_id": "...",
                "doi": "...",
                "pdf_url": "https://...",
                "landing_url": "https://...",
                "categories": ["..."],
                "published_date": "2025-01-01",
                "citations": ["..."],
            }
        ]
    },
    "error": None,
}
```

## 14. Engineering

`pyproject.toml`:

- Runtime dependencies include `openai`, `aiohttp`, `arxiv`, `pypdf`, `python-dotenv`, `pydantic`, `pyyaml`, `rich`, `prompt-toolkit`, and `typer`.
- `chromadb` and `anthropic` are optional extras.
- `dev` includes `pytest`, `ruff`, `mypy`, `pre-commit`, `build`, and related tooling.
- The CLI entry point is registered.
- Skill template package data is included.

`Makefile` and GitHub Actions cover formatting, linting, type checks, compilation, tests, dependency checks, and packaging.

## 15. Tests

Current tests cover:

- Short-term memory and Conversation.
- ReAct final-answer and skill-action paths.
- Config.
- CLI version/providers.
- CLI RAG JSONL search.
- CLI PDF ingest command.
- CLI session RAG slash commands.
- Markdown Skill schema, resource reading, and script non-execution policy.
- Harness Action, PermissionPolicy, TokenBudget, and CostTracker.
- RAG hybrid filters, citation graph, and invalid retrieval modes.
- Optional SDK lazy imports.
- SkillRegistry instance isolation.
- PaperWriter not calling the LLM internally.
- Markdown Skill in Agent flow.
- `rag_search` using the current context `AcademicRAG`.
- Parent agent delegation through `subagent_task`.
- `TaskBoard` dependency clearing and autonomous claiming.
- `TeamMailbox` JSONL send/read/drain behavior.
- `ProtocolStore` request-response approval FSM.
- `WorktreeIndex` task binding and event stream.
- `ContextCompactor` old-observation compaction.
- `harness_coordination` Skill persistence.

Recommended quality gate:

```bash
python -m ruff check ai_researcher_assistant tests examples
python -m mypy ai_researcher_assistant
python -m compileall ai_researcher_assistant tests examples
python -m pytest tests/ -v
python -m pip check
python -m build
python -m ai_researcher_assistant.cli doctor --json
python -m ai_researcher_assistant.cli chat --help
```

## 16. Roadmap

P0:

- Add fine-grained `PermissionPolicy` rules for URL domains, file path prefixes, and read/write modes.
- Add batch directory ingestion for `rag ingest-pdf`.
- Add a standardized Source Skill base class and example implementation.
- Add configurable `SubagentSpec` loading, for example from project YAML or Markdown agent profiles.
- Add a safe background executor that runs only PermissionPolicy-approved command templates and writes completion notifications into TeamMailbox.

P1:

- Add persistent RAG integration tests.
- Add citation-aware reranking.
- Add CLI local-knowledge-base save/restore.
- Add `py.typed`.
- Add subagent trace evaluation samples to verify summary-only context isolation.
- Add CLI subcommands for `harness_coordination`, such as `ai-researcher tasks list` and `ai-researcher team inbox`.

P2:

- Add Semantic Scholar, OpenAlex, Crossref, and similar source skills.
- Add an evaluation harness that replays agent flows using fixed mock LLM traces.
- Add release artifact verification, changelog, and version-release workflow.
- Evaluate MCP tool protocol and A2A multi-agent protocol adapters while preserving the local default path.
- Add an explicit `git worktree` adapter on top of WorktreeIndex, with remove/force operations gated by permissions.

## 17. Maintenance Rules

- New dependencies must update `pyproject.toml`, docs, and tests.
- New context keys must update `harness/context.py`.
- New skills must not call the LLM.
- New providers should only appear in `llm/` adapters, factory, and capabilities.
- Confirm public API compatibility before large structural changes.
- Run at least `make all` before submitting.
