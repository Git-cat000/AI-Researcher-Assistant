# AI Researcher Assistant 中文技术文档

本文档说明 AI Researcher Assistant 当前的技术架构、每个模块的实现方式、核心数据流、扩展方式、质量门和后续技术规划。

## 1. 总体设计

项目采用 LLM + Harness 架构：

```text
用户任务
  -> ResearcherAgent
  -> Harness / Orchestration
  -> LLM 生成 Thought / Action / Final Answer
  -> SkillRegistry 调用工具
  -> Observation 回写上下文
  -> LLM 生成最终答案
```

核心边界：

- LLM 负责推理、规划、选择工具和生成答案。
- Harness 负责组织执行循环、解析模型输出、调用工具、收集观察结果。
- Skill 是工具，不是 Agent，不应该私自调用 LLM。
- Memory / RAG 负责存储和检索上下文，不负责决策。

## 2. 包结构

```text
ai_researcher_assistant/
  core/
  harness/
  llm/
  memory/
  skills/
  orchestration/
```

### 2.1 core/

`core/` 存放稳定的数据契约和基础类型。

主要文件：

- `config.py`
- `message.py`
- `exceptions.py`
- `base_agent.py`

#### config.py

实现配置 dataclass：

- `AgentConfig`
- `LLMConfig`
- `MemoryConfig`
- `SkillsConfig`

设计要点：

- 配置是懒加载的。
- 导入模块时不自动读取 `.env`。
- 导入模块时不创建目录、不连接数据库、不初始化模型 SDK。
- `load_config_from_env()` 显式读取 `.env`。
- `get_config()` 仅保留为兼容默认配置入口。

`LLMConfig.provider` 支持：

```text
openai, openai_compatible, azure_openai, deepseek,
openrouter, siliconflow, qwen, anthropic, ollama, local
```

OpenAI-compatible provider 统一走 `OpenAILLM`，通过 `base_url` 接入。

#### message.py

实现对话消息结构：

- `MessageRole`
- `Message`
- `Conversation`

`Conversation.get_context_window()` 会把内部消息转换成模型 API 所需的：

```python
{"role": "user", "content": "..."}
```

当前 token 裁剪仍是粗略实现，后续可接入 `tiktoken` 或 provider tokenizer。

#### exceptions.py

统一异常体系：

```text
AgentError
  LLMError
  SkillError
  MemoryError
  ConfigurationError
  SandboxError
```

所有 provider、memory、skill 的异常应该包装成项目内部异常，避免上层直接依赖第三方 SDK 的异常类型。

## 3. harness/

`harness/` 是本轮重构新增模块，用来承载 Agent Harness 的稳定契约。

### 3.1 context.py

定义 `AgentContext`：

```python
class AgentContext(TypedDict, total=False):
    llm: Any
    conversation: Any
    short_term_memory: Any
    rag: Any
    config: Any
    skill_registry: Any
    execution_state: Any
    plan_results: dict[str, Any]
    state: Any
```

作用：

- 明确执行循环和工具之间共享哪些上下文。
- 作为 Harness 上下文的文档化契约。
- 当前运行时仍使用普通 `dict[str, Any]`，避免 TypedDict 与可变上下文在类型系统中的兼容问题。
- 为后续 schema 校验和更严格的 mypy 配置打基础。

### 3.2 parsing.py

集中解析 LLM 输出：

- `extract_json_block()`
- `extract_action()`
- `extract_thought()`
- `extract_final_answer()`

ReAct 输出通常形如：

```text
Thought: ...
Action:
```json
{"skill": "paper_reader", "parameters": {"path": "..."}}
```
```

解析逻辑从多个 loop 中抽离出来后，有两个好处：

- 不同执行循环复用同一套解析规则。
- 可以用单元测试覆盖边界情况。

## 4. llm/

`llm/` 是模型适配层，只负责把统一消息格式转换为不同模型服务商 API 调用。

### 4.1 base.py

定义统一接口：

```python
class BaseLLM:
    def generate(...)
    async def agenerate(...)
    def stream_generate(...) -> AsyncIterator[str]
```

返回结构为 `LLMResponse`：

```python
LLMResponse(
    content="...",
    model="...",
    usage={...},
    finish_reason="stop",
    raw_response=...
)
```

### 4.2 openai.py

`OpenAILLM` 支持 OpenAI-compatible chat completion API。

可接入：

- OpenAI
- Azure OpenAI
- DeepSeek
- OpenRouter
- SiliconFlow
- Qwen / DashScope OpenAI-compatible endpoint
- 本地 OpenAI-compatible gateway

实现方式：

- SDK 在构造 `OpenAILLM` 时才导入。
- 如果未安装 `openai`，抛出 `LLMError`。
- 同步、异步、流式接口都转成 `LLMResponse` 或文本流。

### 4.3 anthropic.py

`AnthropicLLM` 适配 Claude Messages API。

实现方式：

- 将 OpenAI 风格 messages 转换为 Anthropic 所需格式。
- system message 单独提取为 `system` 参数。
- SDK 懒加载。
- provider 异常包装为 `LLMError`。

### 4.4 local.py

`OllamaLLM` 通过 HTTP 调用本地 Ollama：

- `/api/chat`
- 支持同步请求。
- 支持异步请求。
- 支持 stream。

本地模型通过 `OLLAMA_BASE_URL` 或 `LLMConfig.base_url` 配置。

### 4.5 factory.py

`create_llm()` 根据 `LLMConfig.provider` 创建具体适配器。

设计重点：

- provider alias 在 factory 层解析。
- adapter 内不写任务逻辑。
- 任务策略留给 orchestration / harness。

## 5. skills/

`skills/` 是工具系统。

### 5.1 base.py

核心类型：

- `SkillParameter`
- `SkillManifest`
- `BaseSkill`

每个 Skill 必须实现：

```python
def _build_manifest(self) -> SkillManifest
def execute(self, parameters, context) -> dict
```

标准返回：

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "错误信息"}
```

### 5.2 registry.py

`SkillRegistry` 管理工具注册和调用。

本轮优化：

- 从 `__new__` 单例改成普通实例。
- 不同 Agent 可以拥有不同 registry。
- 单元测试之间不会共享状态。
- `get_skill_registry()` 仅作为兼容默认入口保留。

### 5.3 loader.py

`SkillLoader` 支持：

- 从 Python class 加载。
- 从 Python module 加载。
- 从目录加载。
- 从 Markdown 文件加载。
- 加载内置技能。

目录加载逻辑：

1. 如果路径是 `.md` 文件，按 Markdown Skill 加载。
2. 如果目录根部有 `SKILL.md`，加载该 Skill。
3. 递归发现 `**/SKILL.md`。
4. 加载目录根部 standalone `.md`。
5. 加载 `.py` 中的 `BaseSkill` 子类。

### 5.4 markdown.py

新增 `MarkdownSkill`，兼容 Claude Code 和 Codex 风格的 Skill 文件。

支持结构：

```text
skill-name/
  SKILL.md
  references/
  scripts/
  assets/
```

`SKILL.md` 示例：

```markdown
---
name: literature-review
description: 当用户需要文献综述时使用。
tags: [research, writing]
---

# Literature Review

请比较论文贡献、方法、证据和局限。
```

实现细节：

- 解析 YAML frontmatter。
- 如果安装了 `yaml`，使用 `yaml.safe_load()`。
- 如果没有安装 `yaml`，使用内置的简单 YAML 子集解析器。
- `name` 会标准化为 kebab-case。
- `references/`、`scripts/`、`assets/` 会被发现并作为资源路径返回。
- Markdown Skill 本身不执行脚本、不调用模型。

执行结果示例：

```python
{
    "success": True,
    "result": {
        "skill": "literature-review",
        "description": "...",
        "instructions": "...",
        "frontmatter": {...},
        "skill_file": ".../SKILL.md",
        "skill_root": "...",
        "resources": {
            "references": [...],
            "scripts": [...],
            "assets": [...]
        },
        "requires_llm": True
    },
    "error": None
}
```

## 6. memory/

`memory/` 负责短期记忆、向量记忆和 RAG。

### 6.1 base.py

定义：

- `MemoryItem`
- `BaseMemory`
- `BaseEmbedding`

`BaseMemory` 统一接口：

```python
add()
get()
search()
delete()
clear()
count()
```

### 6.2 short_term.py

`ShortTermMemory` 用 `deque` 保存最近对话。

功能：

- 按最大消息数裁剪。
- 按粗略 token 数裁剪。
- 提供 `get_context_for_llm()` 生成模型上下文。

用途：

- 保存用户输入和助手回答。
- 构造下一轮 LLM prompt。

### 6.3 in_memory.py

新增无依赖本地向量记忆：

- `HashEmbedding`
- `InMemoryVectorMemory`

`HashEmbedding` 实现方式：

- 将文本分词。
- 使用 Python hash 将 token 映射到固定维度。
- 统计词频。
- 做 L2 归一化。

优点：

- 不需要 OpenAI Embedding。
- 不需要 ChromaDB。
- 测试和本地 demo 可以直接运行。

`InMemoryVectorMemory` 实现：

- 内部用 `dict[str, MemoryItem]` 保存数据。
- 添加时生成 embedding。
- 搜索时计算 cosine similarity。
- 支持 metadata filter。
- 支持 `add_batch()`、`list()`、`delete_many()`。

### 6.4 long_term.py

`LongTermMemory` 是 ChromaDB 持久化后端。

实现方式：

- 构造时懒加载 `chromadb`。
- 默认 embedding 是 `OpenAIEmbedding`。
- 支持 add、batch add、search、delete、list、clear、count。

适用场景：

- 长期保存论文知识库。
- 跨进程或跨会话复用向量库。

### 6.5 rag.py

`AcademicRAG` 是面向论文的 RAG 层。

默认行为：

- 使用 `InMemoryVectorMemory`。
- 使用 `HashEmbedding`。
- 不依赖外部服务。

持久化行为：

```python
rag = AcademicRAG(prefer_persistent=True)
```

主要方法：

- `add_paper()`
- `search_papers()`
- `build_context()`
- `delete_paper()`
- `count_papers()`

`add_paper()` 实现：

1. 根据 `arxiv_id` 或 title 生成 `paper_key`。
2. 默认先删除已有同 key 论文，避免重复。
3. 把 abstract 存为一个 chunk。
4. 把 full text 按 `chunk_size` 和 `chunk_overlap` 切片。
5. 写入 memory backend。

`search_papers()` 实现：

1. 调 memory.search。
2. 根据 `paper_key` 聚合同一论文的 chunks。
3. 区分 abstract 和 full_text chunk。
4. 按最小 distance 排序。
5. 返回 paper-level 结果，而不是裸 chunk 列表。

`build_context()` 实现：

1. 调 `search_papers()`。
2. 格式化标题、作者、arXiv ID、分类、摘要。
3. 可选加入全文片段。
4. 按粗略 token 预算截断。

## 7. orchestration/

`orchestration/` 负责把模型、工具、记忆组织成 Agent 执行流程。

### 7.1 agent.py

`ResearcherAgent` 是主入口。

构造参数：

- `config`
- `name`
- `llm`
- `llm_factory`
- `enable_rag`
- `enable_builtin_skills`
- `skill_registry`

优化点：

- 构造时不立刻创建 RAG。
- 支持注入 mock LLM。
- 支持注入独立 SkillRegistry。
- `process()` 在已有 event loop 中会要求使用 `aprocess()`。
- `_build_conversation()` 会避免 metadata 中的 `role` 重复传参。

### 7.2 loop.py

包含三种执行循环：

- `ReActLoop`
- `PlanAndExecuteLoop`
- `LLMCompilerLoop`

#### ReActLoop

流程：

```text
Thought
  -> Action JSON
  -> Skill execution
  -> Observation
  -> 下一轮 Thought
  -> Final Answer
```

关键实现：

- 使用 `harness.parsing` 解析模型输出。
- 如果模型输出包含 `Final Answer:`，会直接完成，不再继续请求模型。
- Skill 执行通过 `SkillRegistry.aexecute()`。
- Observation 会回写到 conversation。

#### PlanAndExecuteLoop

流程：

1. LLM 生成 JSON plan。
2. 逐步执行 plan 中的 skill。
3. 收集每步结果。
4. LLM 综合生成最终答案。

适合结构化多步骤任务。

#### LLMCompilerLoop

流程：

1. LLM 把任务拆成 DAG。
2. 找出依赖已满足的子任务。
3. 可并行执行独立 skill。
4. 汇总 DAG 结果。

适合多个互不依赖的检索或分析任务。

### 7.3 graph.py

提供类 LangGraph 的状态图：

- `StateGraph`
- `CompiledGraph`
- `AgentGraphBuilder`
- `Node`
- `Edge`

支持：

- LLM node
- Skill node
- Function node
- Conditional edge
- Finish point

当前已修复：

- JSON 解析不再使用裸 `except:`。
- `json` 在模块层导入。

### 7.4 middleware.py

中间件 hook：

- `before_think`
- `after_think`
- `before_act`
- `after_act`
- `on_error`

内置：

- `LoggingMiddleware`
- `TelemetryMiddleware`

用途：

- 日志记录。
- 统计执行步骤。
- 未来可加入权限检查、token 预算、审计日志。

### 7.5 state.py

执行状态：

- `ExecutionState`
- `ExecutionStep`
- `ExecutionStatus`

记录内容：

- 当前任务。
- 当前步骤。
- thought。
- action。
- observation。
- error。
- final answer。

## 8. 端到端流程示例

以 Markdown Skill 为例：

```text
用户: 规划一篇 RAG 文献综述
  -> ResearcherAgent.aprocess()
  -> ReActLoop 调用 LLM
  -> LLM 输出 Action: literature-review
  -> SkillRegistry 执行 MarkdownSkill
  -> MarkdownSkill 返回 instructions + resources
  -> ReActLoop 将 Observation 写入 Conversation
  -> LLM 输出 Final Answer
```

这个流程已经有测试覆盖：

```text
tests/test_harness_refactor.py::test_agent_flow_with_markdown_skill
```

## 9. 测试体系

当前测试覆盖：

- 基础 memory 操作。
- Conversation 格式化。
- ReActLoop 初始化。
- 配置加载。
- 基础导入不依赖可选 provider SDK。
- harness parsing。
- SkillRegistry 隔离。
- PaperWriterSkill 不直接调用 LLM。
- Markdown Skill 加载。
- 本地 RAG 添加、去重、检索、上下文构建。
- Agent + Markdown Skill 端到端流程。

本地验证命令：

```bash
python -m ruff check ai_researcher_assistant tests examples
python -m mypy ai_researcher_assistant
python -m compileall ai_researcher_assistant tests examples
python -m pytest tests/ -v
python -m pip check
python -m build
```

当前结果：

```text
ruff: All checks passed
mypy: Success, no issues found in 36 source files
11 passed
pip check: No broken requirements found
build: wheel/sdist generated successfully
```

## 10. 后续技术规划

### 10.1 Skill 系统

- `skills/builtin/` 是唯一的内置技能入口，历史拼写错误目录已删除。
- 支持 Markdown Skill 中声明参数 schema。
- 支持 Markdown Skill 的资源按需读取策略。
- 为 scripts 增加明确权限边界，默认不自动执行。

### 10.2 RAG 系统

- 增加 BM25 + vector hybrid retrieval。
- 支持 reranker。
- 支持按论文、章节、年份、作者过滤。
- 支持 citation graph。
- 支持增量更新和删除。

### 10.3 Harness

- 将 loop 中重复逻辑继续下沉到 `harness/`。
- 定义标准 `Action` schema。
- 定义标准 `Observation` schema。
- 增加权限管理。
- 增加 token budget 和 cost tracking。

### 10.4 模型层

- 将 provider extras 从主依赖中拆分。
- 增加更多本地 OpenAI-compatible server 示例。
- 增加模型 capability 描述，例如是否支持 tool call、vision、long context。

### 10.5 工程质量

- 增加 CI。
- 增加 coverage。
- 提高 mypy 严格度，逐步为更多未标注函数开启 `check_untyped_defs`。
- 补全 long_term memory 和 graph integration tests。
