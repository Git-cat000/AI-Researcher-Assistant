# AI Researcher Assistant 中文技术文档

本文档说明 AI Researcher Assistant 当前架构、各模块实现方式、技术细节、质量门和后续技术规划。当前版本继续遵循 LLM + Harness 原则：

```text
模型是 Agent，代码是 Harness。
```

模型负责推理、规划、选择工具和生成最终答案；Harness 负责提供稳定的执行契约、上下文、工具、记忆、权限、观察结果和预算控制。

## 1. 总体架构

```text
用户任务
  -> CLI 或 Python API
  -> ResearcherAgent.aprocess()
  -> AgentContext
  -> Loop(ReAct / Plan-and-Execute / LLMCompiler)
  -> LLM 生成 Thought / Action / Final Answer
  -> Harness 解析 Action 并检查权限
  -> SkillRegistry 调用 Skill
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
```

## 2. core/

`core/` 存放稳定基础类型。

- `config.py`：使用 dataclass 管理 Agent、LLM、Memory、Skills 配置。模块导入时不读取 `.env`、不创建目录、不初始化 SDK；`load_config_from_env()` 才显式读取环境变量。
- `message.py`：定义 `MessageRole`、`Message`、`Conversation`，并提供粗粒度 token 估算和上下文窗口裁剪。
- `exceptions.py`：统一内部异常体系，包括 `AgentError`、`LLMError`、`SkillError`、`MemoryError`、`ConfigurationError` 和 `SandboxError`。
- `base_agent.py`：定义同步/异步 Agent 基类。

设计要求：公共函数保持类型标注，配置和 provider 初始化必须延迟到实际使用阶段。

## 3. harness/

`harness/` 是本轮优化的关键边界层，负责稳定契约而不是任务策略。

### 3.1 context.py

`AgentContext` 是 TypedDict 契约，记录循环和工具共享的上下文键：

- `llm`
- `conversation`
- `short_term_memory`
- `rag`
- `config`
- `skill_registry`
- `execution_state`
- `permission_policy`
- `token_budget`
- `cost_tracker`

新增键必须在这里登记，避免散落的 `dict[str, Any]` 隐式协议继续扩散。

### 3.2 parsing.py

集中解析模型输出：

- `extract_json_block()`
- `extract_action()`
- `extract_thought()`
- `extract_final_answer()`

这样 ReAct、Plan-and-Execute 和图执行可以复用同一套边界规则，避免每个 loop 自己写脆弱的字符串处理。

### 3.3 schema.py

新增稳定 harness schema：

- `Action`：规范化模型请求的 skill 调用，要求 `skill` 为非空字符串，`parameters` 为对象。
- `Observation`：统一包装 Skill 执行结果，提供 `to_prompt(max_tokens=...)` 以便回写模型上下文。
- `PermissionPolicy`：支持 allow/block skill 策略，并保留网络、文件系统和脚本执行权限字段。
- `TokenBudget`：控制上下文和 Observation 的 token 预算。
- `CostTracker`：累计模型调用次数和 token usage，并写回 `execution_state.metadata["cost"]`。

当前 `BaseLoop` 已经通过这些契约执行 Action、截断 Observation、裁剪上下文并记录成本。

## 4. llm/

`llm/` 是模型适配层，只负责把统一消息格式转换为 provider API。

- `base.py`：定义 `BaseLLM` 和 `LLMResponse`。
- `openai.py`：适配 OpenAI-compatible Chat Completions API，支持 OpenAI、Azure OpenAI、DeepSeek、OpenRouter、SiliconFlow、Qwen/DashScope 和本地兼容网关。新版 OpenAI SDK 的复杂消息类型通过内部 cast 收窄，保持项目公共接口简单。
- `anthropic.py`：适配 Anthropic Messages API，延迟导入 SDK。
- `local.py`：通过 HTTP 调用 Ollama `/api/chat`。
- `factory.py`：根据 `LLMConfig.provider` 创建适配器，provider alias 在这里解析。
- `capabilities.py`：记录 provider 能力元数据，包括是否支持本地模型、流式输出、OpenAI-compatible 路由和默认环境变量。

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

### 5.1 base.py

定义：

- `SkillParameter`
- `SkillManifest`
- `BaseSkill`

每个 Skill 必须实现 `_build_manifest()` 和 `execute()`。

### 5.2 registry.py

`SkillRegistry` 现在是普通实例，不再是全局单例。这样不同 Agent 和不同测试可以隔离技能集合。

### 5.3 loader.py

`SkillLoader` 支持：

- 加载 Python class。
- 加载 Python module。
- 从目录递归发现 `**/SKILL.md`。
- 加载目录根部 `SKILL.md`。
- 加载 standalone `.md` Skill。
- 加载内置技能。

### 5.4 markdown.py

Markdown Skill 兼容 Claude Code 和 Codex 常见结构：

```text
skill-name/
  SKILL.md
  references/
  scripts/
  assets/
```

实现细节：

- 使用 YAML frontmatter 解析 `name`、`description`、`tags`、`version`、`author`。
- 支持 `parameters`、`params`、`input_schema`、`schema` 等常见参数声明形式。
- 支持 JSON-schema-like `properties + required` 结构。
- 自动发现 `references/`、`scripts/`、`assets/` 资源。
- `read_resources` 或 `resource_paths` 只允许读取 skill 根目录下的非 `scripts/` 文件。
- `scripts/` 默认只暴露路径，不执行。
- Markdown Skill 返回说明和上下文给 Harness，由 Agent Loop 决定下一步是否调用 LLM。

## 6. memory/

`memory/` 负责短期记忆、向量记忆和论文 RAG。

### 6.1 short_term.py

`ShortTermMemory` 用 `deque` 保存最近对话，支持按消息数和粗粒度 token 数裁剪。

### 6.2 in_memory.py

`HashEmbedding` 和 `InMemoryVectorMemory` 提供无外部依赖的本地检索能力：

- 文本分词。
- hash 映射到固定维度。
- 词频向量归一化。
- cosine similarity 搜索。
- metadata filter。

这让测试和本地 demo 不依赖 OpenAI Embedding 或 ChromaDB。

### 6.3 long_term.py

`LongTermMemory` 是 ChromaDB 持久化后端。`chromadb` 和 OpenAI embedding 相关 SDK 均延迟导入，基础安装不会强制拉取。

### 6.4 rag.py

`AcademicRAG` 是论文级 RAG 层。

已实现能力：

- `add_paper()`：按 arXiv ID 或标题生成 paper key，写入摘要 chunk 和全文 chunk。
- `search_papers()`：聚合同一论文的 chunk，返回 paper-level 结果。
- `build_context()`：生成紧凑论文上下文供 LLM 使用。
- `delete_paper()`：按 arXiv ID 或标题删除论文。
- `update_paper()`：用替换式写入更新论文内容和元数据。
- `count_papers()`：按唯一 paper key 计数，不再按 chunk 数估算。
- `citation_graph()`：从 `citations` 元数据生成节点和边。
- `citation_neighbors()`：查询某篇论文的入边和出边。

检索模式：

- `vector`：只使用向量检索。
- `keyword` / `bm25`：使用本地 BM25 风格词项评分。
- `hybrid`：按 `vector_weight` 融合向量分数和 BM25 分数。

过滤能力：

- `categories`
- `arxiv_id`
- `authors`
- `published_year`
- `section`
- `metadata_filter`

## 7. orchestration/

`orchestration/` 把模型、工具、记忆和 harness 契约组织成 Agent 执行流。

### 7.1 agent.py

`ResearcherAgent` 是主入口。它支持：

- 注入 config。
- 注入 mock LLM。
- 注入独立 SkillRegistry。
- 延迟初始化 RAG 和 LLM。
- 同步 `process()` 与异步 `aprocess()`。
- 执行后把 CostTracker 写入执行状态。

设计约束：如果当前线程已经有 event loop，调用者必须使用 `await aprocess()`。

### 7.2 loop.py

三种执行循环：

- `ReActLoop`
- `PlanAndExecuteLoop`
- `LLMCompilerLoop`

`BaseLoop` 现在负责共享机制：

- 初始化 conversation。
- 统一调用 LLM。
- 应用 TokenBudget。
- 记录 CostTracker。
- 统一执行 Action。
- 统一执行 PermissionPolicy。
- 统一格式化 Observation。
- 统一错误处理和最终答案合成。

这样任务策略仍然在 loop 中，Skill 不越界成为 Agent。

### 7.3 graph.py

提供轻量 LangGraph-like workflow graph。JSON 提取逻辑使用 `harness/parsing.py`，避免手写脆弱解析和裸 `except:`。

### 7.4 middleware.py / state.py

`middleware.py` 提供生命周期 hook 和 telemetry；`state.py` 保存执行步骤、状态和元数据。

## 8. CLI

新增 `ai_researcher_assistant/cli.py`，并在 `pyproject.toml` 中注册：

```toml
[project.scripts]
ai-researcher = "ai_researcher_assistant.cli:main"
```

CLI 技术选型：

- 使用 Typer，因为它基于 Python 类型标注声明参数，适合项目当前的 typed API 风格。
- Typer 底层基于 Click，继承成熟的命令解析、帮助页和子命令机制。
- 使用 Python packaging 的 `[project.scripts]` 生成跨平台 console script。

命令结构：

- `version`：输出版本。
- `doctor`：检查本地内置 skill 和 provider 元数据。
- `providers`：列出 provider 能力。
- `ask`：运行一次 agent 任务。
- `skills list`：列出内置和外部 skill。
- `skills inspect`：解析单个 Markdown Skill。
- `rag search`：从 JSONL 构建本地 RAG 并搜索。
- `rag graph`：从 JSONL 构建引用图。

CLI 边界：命令只做参数解析、输入加载和输出格式化，核心行为委托给 `ResearcherAgent`、`SkillLoader` 和 `AcademicRAG`。

## 9. 工程化

`pyproject.toml`：

- 声明运行依赖：`openai`、`aiohttp`、`arxiv`、`pypdf`、`python-dotenv`、`pydantic`、`pyyaml`、`typer`、`rich`。
- `chromadb`、`anthropic` 是 optional extras。
- `dev` extra 包含 `pytest`、`ruff`、`mypy`、`pre-commit`、`build` 等。
- 注册 CLI entry point。
- 包含 Skill template package data。

`Makefile`：

- `install`
- `format`
- `lint`
- `typecheck`
- `compile`
- `test`
- `check`
- `build`
- `cli`
- `all`

GitHub Actions：

- Python 3.10、3.11、3.12 matrix。
- Ruff lint。
- Ruff format check。
- mypy。
- compileall。
- pytest coverage。
- pip check。
- package build。

## 10. 测试

当前测试覆盖：

- 短期记忆。
- Conversation。
- ReAct final answer。
- ReAct skill action。
- Config。
- CLI version/providers。
- CLI RAG JSONL search。
- Markdown Skill schema、资源读取和脚本非执行策略。
- Harness Action、PermissionPolicy、TokenBudget。
- RAG hybrid filter 和 citation graph。
- Optional SDK lazy import。
- SkillRegistry 实例隔离。
- PaperWriter 不内部调用 LLM。
- Markdown Skill 与 Agent flow。

当前本地验证结果：

```text
python -m ruff check ai_researcher_assistant tests examples  # passed
python -m mypy ai_researcher_assistant                       # passed
python -m compileall ai_researcher_assistant tests examples  # passed
python -m pytest tests/ -v                                   # 17 passed
python -m pip check                                          # passed
python -m build                                              # passed
python -m ai_researcher_assistant.cli doctor --json          # passed
```

## 11. 已完成规划项

- 修复 `builtin` 拼写，只保留正确目录。
- 移除全局 SkillRegistry 单例。
- Provider SDK 延迟导入。
- ChromaDB 和 Anthropic 改为可选依赖。
- RAG 默认可本地运行。
- Markdown Skill 兼容 Claude Code / Codex。
- Markdown Skill 支持参数 schema 和资源读取策略。
- PaperWriter 不再内部调用 LLM。
- `harness/schema.py` 实现 Action、Observation、权限、预算和成本契约。
- 执行循环接入权限检查、Observation 截断、上下文预算和成本记录。
- RAG 增加 hybrid/BM25 检索、过滤、rerank、update、citation graph。
- 新增模型能力元数据。
- 新增 Typer CLI 和 console script。
- 补充 CLI、RAG、Skill、Harness 回归测试。
- CI 增加 package build。

## 12. 后续技术规划

优先级 P0：

- 为每个内置 Skill 增加更严格的输入校验和错误路径测试。
- 给 `PermissionPolicy` 增加按资源类型的细粒度规则，例如网络域名、文件路径前缀和只读/写入权限。
- 将 `CostTracker` 扩展为按 provider/model 记录，并支持价格表外部配置。

优先级 P1：

- 增加持久化 RAG 的集成测试，使用临时 ChromaDB 目录并跳过无依赖环境。
- 增加 citation-aware rerank，把直接引用、共同引用和入边数量纳入排序信号。
- 增加 CLI 导入 PDF/arXiv 的子命令，让本地知识库构建更完整。
- 增加 `py.typed`，逐步提升 typed package 质量。

优先级 P2：

- 支持更多本地 OpenAI-compatible gateway 的配置示例。
- 增加实验性评测 harness，用固定 mock LLM trace 回放 Agent 流程。
- 增加文献综述模板、论文对比模板和引用检查模板。
- 提供更完整的发布流程，例如版本号自动检查、changelog 和 release artifact 验证。

## 13. 维护规则

- 新依赖必须更新 `pyproject.toml`、文档和测试。
- 新 context key 必须更新 `harness/context.py`。
- 新 Skill 不得调用 LLM。
- 新 provider 只应出现在 `llm/` adapter、factory 和 capabilities 中。
- 大范围结构调整前先确认公共 API 兼容性。
- 提交前至少运行 `make all`。
