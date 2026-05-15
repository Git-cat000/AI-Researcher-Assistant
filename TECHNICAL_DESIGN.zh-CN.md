# AI Researcher Assistant 中文技术文档

本文档说明 AI Researcher Assistant 当前架构、模块实现方式、CLI session、PDF 入库、非 arXiv 来源扩展方式、质量门和后续规划。

核心原则：

```text
模型是 Agent，代码是 Harness。
```

模型负责推理、规划、选择工具和生成最终答案；Harness 负责提供稳定的执行契约、上下文、工具、记忆、权限、观察结果和预算控制。

## 1. 总体架构

```text
用户任务
  -> CLI / CLI session / Python API
  -> ResearcherAgent.aprocess()
  -> AgentContext
  -> Loop(ReAct / Plan-and-Execute / LLMCompiler)
  -> LLM 生成 Thought / Action / Final Answer
  -> Harness 解析 Action 并检查权限
  -> SkillRegistry 调用 Skill
  -> 可选：subagent_task 创建本地子 Agent 并返回 summary
  -> Observation 回写对话上下文
  -> LLM 生成最终答案
```

依赖方向：

```text
orchestration -> llm
orchestration -> skills
orchestration -> memory
orchestration -> harness
skills 不调用 llm
memory 不决定任务策略
llm adapter 不写业务策略
cli 不写 Agent 策略
```

本轮参考 `learn-claude-code` s04-s12 后，项目采用的迁移原则是：把协作机制落在 Harness 控制面，而不是让工具或模型适配器承担策略。对应关系如下：

```text
s04 子 Agent 隔离       -> SubagentRunner + subagent_task
s05 按需 Skill 知识     -> compact skill catalog + Markdown Skill observation
s06 上下文压缩          -> ContextCompactor
s07 持久任务图          -> TaskBoard
s08 后台通知思想        -> JSON/JSONL 控制面可被后续后台执行器注入
s09 团队邮箱            -> TeamMailbox
s10 请求响应协议        -> ProtocolStore
s11 自主认领            -> TaskBoard.claim_next()
s12 worktree 隔离绑定   -> WorktreeIndex
```

## 2. core/

`core/` 存放稳定基础类型。

- `config.py`：使用 dataclass 管理 Agent、LLM、Memory、Skills 配置。模块导入时不读取 `.env`、不创建目录、不初始化 SDK；`load_config_from_env()` 才显式读取环境变量。
- `message.py`：定义 `MessageRole`、`Message`、`Conversation`，并提供粗粒度 token 估算和上下文窗口裁剪。
- `exceptions.py`：统一内部异常体系。
- `base_agent.py`：定义同步/异步 Agent 基类。

## 3. harness/

`harness/` 负责稳定契约而不是任务策略。

- `context.py`：定义 `AgentContext`，登记 loop、tools、memory、graph 共享的上下文键。
- `coordination.py`：实现持久任务图、团队邮箱、请求响应协议、worktree 绑定索引和确定性上下文压缩。
- `parsing.py`：集中解析 LLM 输出，包括 JSON block、Action、Thought 和 Final Answer。
- `schema.py`：定义 `Action`、`Observation`、`PermissionPolicy`、`TokenBudget`、`CostTracker`。

当前 `BaseLoop` 已接入：

- Action 规范化。
- Skill 执行前权限检查。
- Observation 格式化和 token 截断。
- LLM 调用 token budget。
- 模型 token usage 记录。

## 4. llm/

`llm/` 是模型适配层，只负责把统一消息格式转换为 provider API。

- `base.py`：定义 `BaseLLM` 和 `LLMResponse`。
- `openai.py`：适配 OpenAI-compatible Chat Completions API。
- `anthropic.py`：适配 Anthropic Messages API。
- `local.py`：通过 HTTP 调用 Ollama。
- `factory.py`：根据 `LLMConfig.provider` 创建适配器。
- `capabilities.py`：记录 provider 能力、凭据环境变量、本地模型标记和 OpenAI-compatible 标记。

支持 provider：

```text
openai, openai_compatible, azure_openai, deepseek, openrouter,
siliconflow, qwen, anthropic, ollama, local
```

## 5. skills/

`skills/` 是工具层。工具执行确定性工作或外部调用，并返回结构化结果：

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "message"}
```

实现边界：

- Skill 不调用 LLM。
- Skill 不决定任务策略。
- Skill 不绕过 `PermissionPolicy`。
- 需要 API key 的 Skill 通过配置或环境变量获得凭据。

主要模块：

- `base.py`：`SkillParameter`、`SkillManifest`、`BaseSkill`。
- `registry.py`：普通实例注册表，不再是全局单例。
- `loader.py`：支持 Python class、Python module、目录、`SKILL.md`、standalone `.md` 和内置技能。
- `markdown.py`：兼容 Claude Code / Codex 风格 Markdown Skill。
- `builtin/rag_search.py`：把当前 `AcademicRAG` 暴露为确定性检索 Skill。
- `builtin/subagent_task.py`：把本地子 Agent 委托暴露为异步 Skill。
- `builtin/harness_coordination.py`：把任务图、邮箱、协议和 worktree 绑定暴露为一个统一的确定性控制面 Skill。

Markdown Skill 支持：

- YAML frontmatter。
- `parameters`、`params`、`input_schema`、`schema` 参数声明。
- JSON-schema-like `properties + required`。
- `references/`、`scripts/`、`assets/` 资源发现。
- `read_resources` / `resource_paths` 读取非 `scripts/` 文件。
- `scripts/` 默认只暴露路径，不执行。

当前系统提示词使用 `SkillRegistry.get_catalog_for_all()` 输出轻量 Skill catalog，而不是把所有 Skill 的详细说明全部塞入每轮 system prompt。这样更接近 s05 的“双层加载”思想：模型先知道有哪些工具，真正需要时再通过 action/observation 获取完整资源或执行结果。

## 6. memory/

`memory/` 负责短期记忆、向量记忆和论文 RAG。

- `short_term.py`：用 `deque` 保存最近对话，支持按消息数和 token 数裁剪。
- `in_memory.py`：提供 `HashEmbedding` 和 `InMemoryVectorMemory`，本地测试不依赖外部 embedding 服务。
- `long_term.py`：ChromaDB 持久化后端，相关依赖延迟导入。
- `rag.py`：论文级 RAG 层。

`AcademicRAG` 已实现：

- `add_paper()`
- `search_papers()`
- `build_context()`
- `delete_paper()`
- `update_paper()`
- `count_papers()`
- `citation_graph()`
- `citation_neighbors()`

检索模式：

- `vector`
- `keyword`
- `bm25`
- `hybrid`

过滤字段：

- `categories`
- `arxiv_id`
- `authors`
- `published_year`
- `section`
- `metadata_filter`

## 7. orchestration/

`orchestration/` 把模型、工具、记忆和 harness 契约组织成 Agent 执行流。

- `agent.py`：`ResearcherAgent` 主入口，支持注入 config、mock LLM、独立 `SkillRegistry`，并延迟初始化 RAG 和 LLM。
- `loop.py`：实现 `ReActLoop`、`PlanAndExecuteLoop`、`LLMCompilerLoop`，共享机制集中在 `BaseLoop`。
- `graph.py`：轻量 LangGraph-like workflow graph。
- `middleware.py`：生命周期 hook 和 telemetry。
- `state.py`：执行步骤、状态和元数据。
- `subagent.py`：本地父子 Agent 编排，创建隔离的子 ReActLoop 并只向父 Agent 返回 summary。

## 8. 本地父子 Agent

父子 Agent 的目标是复用 `learn-claude-code` s04 的上下文隔离思想：父 Agent 不直接承载所有中间推理和检索痕迹，而是把局部任务交给子 Agent；子 Agent 完成后只返回可消费的 summary。

执行流程：

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

默认角色：

```text
literature_search_agent  -> arxiv_fetcher, rag_search
paper_reading_agent      -> paper_reader, rag_search
rag_retrieval_agent      -> rag_search
writing_agent            -> paper_writer, rag_search
```

关键实现原则：

- 子 Agent 使用新的 `Conversation`，不继承父 Agent 的完整历史。
- 子 Agent 共享同一个 LLM adapter 和可选 `AcademicRAG`，但只获得允许的 Skill。
- 子 Agent 的 SkillRegistry 会过滤掉 `subagent_task`，避免递归委托。
- 返回给父 Agent 的 observation 是 `SubagentResult.to_dict()`，包含 summary、允许使用的 skill、步数和错误信息。
- 当前实现是本地进程内子 Agent，不依赖 MCP 或 A2A。后续如果接入远程协议，应保留相同的 summary-only 返回契约。

## 9. 协调控制面

`harness/coordination.py` 提供本地文件控制面，默认根目录为 `.ai-researcher/`。它不调用 LLM，也不执行危险 git 操作，只维护可恢复的状态。

### 9.1 TaskBoard

`TaskBoard` 使用 `task_<id>.json` 保存每个任务：

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

关键能力：

- `create()` 创建持久任务。
- `update(..., status="completed")` 会自动从其他任务的 `blocked_by` 中清除已完成任务。
- `ready_tasks()` 返回未阻塞的 pending task。
- `claim_next(owner)` 支持 s11 风格的自主认领。

### 9.2 TeamMailbox

`TeamMailbox` 使用 append-only JSONL 收件箱：

```text
.ai-researcher/team/inbox/writer.jsonl
```

每条消息包含 sender、recipient、message_type、request_id、content 和 timestamp。`read(..., drain=True)` 支持读取后清空，适合 agent loop 在每轮推理前注入 inbox。

### 9.3 ProtocolStore

`ProtocolStore` 实现统一请求响应 FSM：

```text
pending -> approved
pending -> rejected
```

可用于计划审批、关停握手、任务移交等场景。所有请求保存在 `.ai-researcher/team/protocols.json`，每个请求都有稳定 `request_id`。

### 9.4 WorktreeIndex

`WorktreeIndex` 记录任务与隔离目录的绑定：

```text
.ai-researcher/worktrees/index.json
.ai-researcher/worktrees/events.jsonl
```

它会把 `task_id`、worktree name、path、status 写入索引，并向事件流追加 lifecycle event。当前实现只记录隔离意图，不直接调用 `git worktree add/remove`；后续可以在 PermissionPolicy 下增加显式 git 执行器。

### 9.5 ContextCompactor

`ContextCompactor` 为执行历史提供确定性 micro-compact：保留最近若干 observation，旧的大 observation 用占位摘要替换。它适合在展示、日志、后续 summary 构建时使用，避免把大段 PDF/检索输出反复塞回上下文。

## 10. CLI

`ai_researcher_assistant/cli.py` 使用 Typer 实现命令入口，并在 `pyproject.toml` 中注册：

```toml
[project.scripts]
ai-researcher = "ai_researcher_assistant.cli:main"
```

命令结构：

- `version`：输出版本。
- `doctor`：检查内置 skill 和 provider 元数据。
- `providers`：列出 provider 能力。
- `ask`：运行一次 agent 任务。
- `chat`：启动持续命令行 session。
- `skills list`：列出内置和外部 skill。
- `skills inspect`：解析单个 Markdown Skill。
- `rag search`：从 JSONL 构建本地 RAG 并搜索。
- `rag graph`：从 JSONL 构建引用图。
- `rag ingest-pdf`：把本地 PDF 或直接 PDF URL 转换成 JSONL 论文记录。

CLI 边界：命令只做参数解析、输入加载和输出格式化，核心行为委托给 `ResearcherAgent`、`AgentCliSession`、`SkillLoader` 和 `AcademicRAG`。

## 11. 持续 CLI Session

新增 `cli_session.py`，核心类型是 `AgentCliSession`。

实现方式：

- 每个 session 绑定一个 `cwd`。
- prompt 历史写入 `cwd/.ai-researcher/history.txt`。
- `SkillRegistry` 在 session 内独立创建。
- `AcademicRAG` 在 session 内常驻。
- `ResearcherAgent` 在 session 内复用短期记忆。
- 普通文本进入 `ResearcherAgent.aprocess()`。
- 如果 session RAG 中有论文，`AgentCliSession.ask()` 会构建紧凑 RAG context 并附加到本轮任务。
- slash command 不调用 LLM，直接由 session 层处理。

当前 slash command：

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

运行方式：

```bash
ai-researcher chat --cwd .
python -m ai_researcher_assistant.cli chat --cwd .
```

## 12. 本地 PDF 入库

`rag ingest-pdf` 的数据流：

```text
PDF path or direct PDF URL
  -> PaperReaderSkill
  -> full_text / sections / pdf metadata
  -> normalized paper record
  -> JSONL output or session RAG
  -> AcademicRAG.add_paper()
```

推荐操作：

```bash
ai-researcher rag ingest-pdf ./papers/example.pdf --output papers.jsonl --title "Example Paper" --author "A. Researcher" --category cs.CL
ai-researcher rag search "retrieval augmented generation" --paper-jsonl papers.jsonl --include-full-text --json
ai-researcher chat --cwd .
/rag load papers.jsonl
/rag search retrieval augmented generation
```

如果论文不在 arXiv，但有直接 PDF URL：

```bash
ai-researcher rag ingest-pdf https://example.org/paper.pdf --output papers.jsonl --title "External Paper"
```

## 13. 非 arXiv 来源扩展

论文来源不应写死在 RAG 或 LLM adapter 中。每个来源应作为单独 Source Skill：

```text
semantic_scholar_fetcher
crossref_fetcher
pubmed_fetcher
openalex_fetcher
publisher_pdf_fetcher
institutional_repository_fetcher
```

Source Skill 负责：

- 查询外部 API 或网页。
- 处理分页、速率限制和错误。
- 规范化 title、abstract、authors、doi、pdf_url、landing_url、published_date、citations。
- 返回结构化 paper candidates。

Source Skill 不负责：

- 调用 LLM。
- 决定研究策略。
- 直接改写 Agent 状态。
- 绕过 PermissionPolicy 下载任意资源。

推荐返回结构：

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

## 14. 工程化

`pyproject.toml`：

- 运行依赖包括 `openai`、`aiohttp`、`arxiv`、`pypdf`、`python-dotenv`、`pydantic`、`pyyaml`、`rich`、`prompt-toolkit`、`typer`。
- `chromadb`、`anthropic` 是 optional extras。
- `dev` extra 包含 `pytest`、`ruff`、`mypy`、`pre-commit`、`build` 等。
- 注册 CLI entry point。
- 包含 Skill template package data。

`Makefile` 和 GitHub Actions 覆盖格式化、lint、类型检查、编译、测试、依赖检查和构建。

## 15. 测试

当前测试覆盖：

- 短期记忆和 Conversation。
- ReAct final answer 和 skill action。
- Config。
- CLI version/providers。
- CLI RAG JSONL search。
- CLI PDF ingest 命令。
- CLI session RAG slash commands。
- Markdown Skill schema、资源读取和脚本非执行策略。
- Harness Action、PermissionPolicy、TokenBudget、CostTracker。
- RAG hybrid filter、citation graph 和非法检索模式。
- Optional SDK lazy import。
- SkillRegistry 实例隔离。
- PaperWriter 不内部调用 LLM。
- Markdown Skill 与 Agent flow。
- `rag_search` 使用当前 context 中的 `AcademicRAG`。
- 父 Agent 通过 `subagent_task` 委托给 summary-only 子 Agent。
- `TaskBoard` 依赖解锁和自主认领。
- `TeamMailbox` JSONL 消息发送、读取和 drain。
- `ProtocolStore` 请求响应审批状态机。
- `WorktreeIndex` 任务绑定和事件流。
- `ContextCompactor` 旧 observation 压缩。
- `harness_coordination` Skill 持久化状态。

推荐质量门：

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

## 16. 后续规划

P0：

- 给 `PermissionPolicy` 增加按 URL 域名、文件路径前缀和读写类型的细粒度规则。
- 给 `rag ingest-pdf` 增加批量目录导入。
- 为 Source Skill 增加标准化基类和示例实现。
- 为 `SubagentSpec` 增加可配置加载方式，例如从项目级 YAML 或 Markdown agent profile 加载。
- 为后台任务增加安全执行器，让长耗时命令只通过 PermissionPolicy 允许的命令模板运行，并把完成通知写入 TeamMailbox。

P1：

- 增加持久化 RAG 集成测试。
- 增加 citation-aware rerank。
- 增加 CLI 中的本地知识库保存/恢复能力。
- 增加 `py.typed`。
- 增加子 Agent 执行 trace 的评测样例，验证 summary-only 上下文隔离。
- 为 `harness_coordination` 增加 CLI 子命令，例如 `ai-researcher tasks list` 和 `ai-researcher team inbox`。

P2：

- 增加 Semantic Scholar、OpenAlex、Crossref 等来源 Skill。
- 增加评测 harness，用固定 mock LLM trace 回放 Agent 流程。
- 增加 release artifact 验证、changelog 和版本发布流程。
- 在保持本地默认路径的前提下，评估 MCP 工具协议和 A2A 多 Agent 协议适配层。
- 在 worktree 绑定基础上增加显式 `git worktree` 适配器，并要求每次 remove/force 操作通过权限策略。

## 17. 维护规则

- 新依赖必须更新 `pyproject.toml`、文档和测试。
- 新 context key 必须更新 `harness/context.py`。
- 新 Skill 不得调用 LLM。
- 新 provider 只应出现在 `llm/` adapter、factory 和 capabilities 中。
- 大范围结构调整前先确认公共 API 兼容性。
- 提交前至少运行 `make all`。
