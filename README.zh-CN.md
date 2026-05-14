# AI Researcher Assistant

AI Researcher Assistant 是一个面向学术研究场景的 LLM Agent Harness 框架。它的目标不是把所有逻辑塞进模型适配器或工具里，而是让大模型负责推理和决策，让代码提供稳定的模型接口、工具系统、记忆系统、RAG、执行循环和观察结果。

核心原则：

```text
模型是 Agent，代码是 Harness。
```

## 项目能做什么

- 搜索 arXiv 论文。
- 读取 PDF，提取论文文本和章节。
- 把论文摘要、全文切片写入知识库。
- 基于 RAG 检索论文上下文。
- 辅助学术写作、润色、摘要生成和 LaTeX 格式化。
- 支持云端主流模型和本地模型。
- 支持 Python Skill，也支持 Claude Code / Codex 风格的 Markdown Skill。

## 当前状态

项目处于 alpha 阶段，但当前主干已经可以完成本地测试、类型检查和打包验证。最近一轮整理后的稳定边界是：

- 模型 SDK 改为懒加载，基础导入不再强制安装全部模型服务商 SDK。
- Skill Registry 从全局单例改为普通实例，方便测试隔离和多 Agent 并存。
- 内置技能只保留 `ai_researcher_assistant/skills/builtin/`，历史错误拼写目录已删除。
- `PaperWriterSkill` 不再内部调用 LLM，而是返回结构化 prompt，由 Harness 决定后续模型调用。
- 新增 `harness/`，集中放置上下文契约和模型输出解析逻辑。
- 新增本地无依赖向量记忆，RAG 默认可在没有 OpenAI Embedding 和 ChromaDB 的情况下运行。
- 新增 Claude Code / Codex 兼容的 Markdown Skill 加载能力。
- chromadb 和 anthropic SDK 已改为可选依赖，基础安装不强制拉取。
- 三个执行循环中的重复逻辑已抽取为 BaseLoop 共享方法。
- 新增 Makefile、pre-commit、GitHub Actions CI 工程化配套。
- 当前质量门覆盖 Ruff、mypy、pytest、compileall、pip check 和本地构建。

## 安装

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

需要 ChromaDB 持久化或 Anthropic Claude 模型时：

```bash
pip install -e ".[dev,chromadb,anthropic]"
# 或一次性安装全部可选依赖
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

## 模型配置

`LLMConfig.provider` 当前支持：

```text
openai
openai_compatible
azure_openai
deepseek
openrouter
siliconflow
qwen
anthropic
ollama
local
```

OpenAI 兼容服务可以通过 `base_url` 接入，例如 DeepSeek、OpenRouter、SiliconFlow、Qwen、Azure OpenAI 或本地 OpenAI-compatible gateway。

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

## 快速开始

```python
import asyncio

from ai_researcher_assistant.orchestration import ResearcherAgent


async def main() -> None:
    agent = ResearcherAgent(name="Research Assistant")
    answer = await agent.aprocess("请规划一篇关于 RAG 在科研助手中的应用综述。")
    print(answer)


asyncio.run(main())
```

## 使用 Markdown Skill

项目兼容 Claude Code / Codex 风格的 Markdown Skill。一个 Skill 可以是目录中的 `SKILL.md`：

```text
my_skills/
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
tags: [research, writing]
---

# Literature Review

请围绕研究问题建立文献综述结构，比较核心论文的贡献、方法、局限与未来方向。
```

加载方式：

```python
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry

registry = SkillRegistry()
loader = SkillLoader(registry)
loader.load_from_directory("./my_skills")
```

Markdown Skill 不会直接执行任意代码，也不会自行调用 LLM。它会被包装成结构化工具，向 Harness 返回说明、frontmatter、资源路径和调用参数。

## 使用本地 RAG

默认情况下，`AcademicRAG` 使用本地内存向量库和确定性 hash embedding，因此不依赖 OpenAI Embedding 或 ChromaDB。

```python
from ai_researcher_assistant.memory import AcademicRAG

rag = AcademicRAG()
rag.add_paper(
    title="Retrieval Augmented Generation for Scientific Discovery",
    abstract="A paper about RAG for scientific literature search.",
    arxiv_id="2401.00001",
    authors=["A. Researcher"],
    categories=["cs.CL"],
)

results = rag.search_papers("grounded retrieval generation", top_k=1)
context = rag.build_context("grounded retrieval generation")
```

如果需要 ChromaDB 持久化：

```python
rag = AcademicRAG(prefer_persistent=True)
```

## 目录结构

```text
ai_researcher_assistant/
  core/             配置、消息、异常、Agent 基类
  harness/          AgentContext 与模型输出解析工具
  llm/              模型适配层
  memory/           短期记忆、长期记忆、本地向量记忆、RAG
  skills/           Skill 基类、注册表、加载器、内置技能、Markdown Skill
  orchestration/    ResearcherAgent、执行循环、图编排、中间件、执行状态
examples/           示例
tests/              单元测试和流程测试
```

## 测试

```bash
make all           # 一键执行全部质量门
pytest tests/ -v   # 12 个测试
```

或逐步执行：

```bash
python -m ruff check ai_researcher_assistant tests examples
python -m mypy ai_researcher_assistant
python -m compileall ai_researcher_assistant tests examples
python -m pytest tests/ -v
python -m pip check
```

当前本地验证结果：

```text
ruff: All checks passed
mypy: 4 pre-existing (OpenAI SDK type stubs)
12 passed
pip check: No broken requirements found
```

## 技术文档

更详细的模块实现方式、数据流、扩展方式和技术规划见：

[TECHNICAL_DESIGN.zh-CN.md](TECHNICAL_DESIGN.zh-CN.md)
