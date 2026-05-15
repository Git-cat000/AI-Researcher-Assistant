# AI Researcher Assistant 中文说明

AI Researcher Assistant 是一个面向学术研究场景的 LLM Agent Harness。它把模型适配、记忆/RAG、工具化 Skill、执行循环和 CLI 组合起来，用于论文检索、PDF 阅读、文献记忆、引用关系整理和学术写作辅助。

核心原则：

```text
模型是 Agent，代码是 Harness。
```

也就是说，模型负责推理、规划、选择工具和生成最终答案；代码负责提供稳定的工具、记忆、权限、观察结果、上下文预算和执行结构。工具不应该偷偷调用 LLM，也不应该把任务策略写死在内部。

详细技术设计和后续规划见 [TECHNICAL_DESIGN.zh-CN.md](TECHNICAL_DESIGN.zh-CN.md)。

## 当前能力

- 支持 ReAct、Plan-and-Execute、LLMCompiler 风格循环和图编排。
- 支持 OpenAI、Anthropic、Ollama，以及 Azure OpenAI、DeepSeek、OpenRouter、SiliconFlow、Qwen/DashScope 等 OpenAI-compatible 服务。
- 新增 Typer CLI，可通过 `ai-researcher` 或 `python -m ai_researcher_assistant.cli` 使用。
- 支持 Python Skill，也支持 Claude Code / Codex 兼容的 Markdown Skill。
- Markdown Skill 可直接使用 `SKILL.md`，并兼容 `references/`、`scripts/`、`assets/` 目录。
- Markdown frontmatter 可声明参数 schema，Loader 会转换为项目内部 `SkillParameter`。
- Skill 只返回结构化观察结果，不直接调用 LLM。
- RAG 默认使用本地内存向量库和确定性 hash embedding，无需外部 embedding 服务即可运行测试和 demo。
- RAG 支持向量检索、BM25 风格关键词检索、混合检索、过滤、轻量 rerank、引用元数据和引用图导出。
- Harness 层新增 Action、Observation、PermissionPolicy、TokenBudget 和 CostTracker。

## 安装

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

需要 ChromaDB 持久化或 Anthropic Claude 时：

```bash
pip install -e ".[dev,chromadb,anthropic]"
pip install -e ".[dev,all]"
```

复制环境变量模板：

```bash
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

常用环境变量：

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
SILICONFLOW_API_KEY=...
DASHSCOPE_API_KEY=...
AZURE_OPENAI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

## CLI 使用

本项目使用 Typer 实现 CLI。CLI 只负责参数解析、格式化输出和调用 Harness，不把 Agent 策略写进命令处理函数。

```bash
ai-researcher version --json
ai-researcher doctor --json
ai-researcher providers --json
ai-researcher skills list --json
ai-researcher skills inspect ./my-skills/literature-review/SKILL.md --json
ai-researcher rag search "hybrid retrieval" --paper-jsonl papers.jsonl --json
ai-researcher rag graph --paper-jsonl papers.jsonl --json
ai-researcher ask "查找 RAG 在科研发现中的应用论文" --provider openai --model gpt-4o
```

如果 `ai-researcher` 不在 PATH 中，也可以直接运行：

```bash
python -m ai_researcher_assistant.cli doctor --json
```

`rag search` 的 JSONL 输入格式为一行一篇论文：

```json
{"title":"Hybrid Retrieval for Research Agents","abstract":"...","arxiv_id":"2401.10000","authors":["A. Researcher"],"categories":["cs.IR"],"published_date":"2024-01-01","citations":["2301.00001"]}
```

## Python 快速开始

```python
import asyncio

from ai_researcher_assistant.orchestration import ResearcherAgent


async def main() -> None:
    agent = ResearcherAgent(name="Research Assistant")
    answer = await agent.aprocess("请总结两篇关于 RAG 科研助手的论文。")
    print(answer)
    agent.shutdown()


asyncio.run(main())
```

## Markdown Skill

项目兼容 Claude Code / Codex 风格的 Markdown Skill。一个 Skill 可以是单个 `.md` 文件，也可以是包含 `SKILL.md` 的目录：

```text
my-skills/
  literature-review/
    SKILL.md
    references/
    scripts/
    assets/
```

`SKILL.md` 示例：

```markdown
---
name: literature-review
description: 当用户需要文献综述、相关工作比较或研究脉络整理时使用。
parameters:
  topic:
    type: string
    description: 研究主题
    required: true
tags: [research, writing]
---

# Literature Review

围绕研究问题比较论文贡献、方法、证据、局限和未来方向。
```

执行 Markdown Skill 时，系统会返回说明、frontmatter、参数、资源路径和请求读取的参考资料内容。`scripts/` 中的文件只作为资源暴露，默认不会执行。

## 本地 RAG

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

results = rag.search_papers("grounded retrieval", retrieval_mode="hybrid")
graph = rag.citation_graph()
```

如果需要 ChromaDB 持久化：

```python
rag = AcademicRAG(prefer_persistent=True)
```

## 目录结构

```text
ai_researcher_assistant/
  core/             配置、消息、异常、Agent 基类
  harness/          Action、Observation、权限、TokenBudget、解析工具
  llm/              模型适配器、provider factory、模型能力元数据
  memory/           短期记忆、本地向量记忆、ChromaDB、AcademicRAG
  skills/           Skill 基类、注册表、加载器、Markdown 兼容层、内置技能
  orchestration/    ResearcherAgent、执行循环、图编排、状态、中间件
examples/           示例
tests/              单元测试和回归测试
```

## 质量门

当前本地验证通过：

```text
ruff check: passed
mypy: passed
compileall: passed
pytest: 17 passed
pip check: passed
python -m build: passed
CLI doctor: passed
```

推荐提交前运行：

```bash
make all
python -m ai_researcher_assistant.cli doctor --json
```

## 设计边界

- `harness/` 保存稳定执行契约，不写业务策略。
- `orchestration/` 负责循环、规划、LLM 调用和执行顺序。
- `skills/` 是工具层，执行确定性或外部工作，返回结构化观察结果。
- `memory/` 负责保存和检索上下文，不决定任务策略。
- `llm/` 只适配 provider API，不把任务策略写进模型适配器。

## 许可证

MIT，见 [LICENSE](LICENSE)。
