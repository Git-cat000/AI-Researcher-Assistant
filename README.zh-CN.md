# AI Researcher Assistant 中文说明

AI Researcher Assistant 是一个面向学术研究场景的 LLM Agent Harness。它把模型适配、记忆/RAG、工具化 Skill、执行循环和 CLI 组合起来，用于论文检索、PDF 阅读、文献记忆、引用关系整理和学术写作辅助。

核心原则：

```text
模型是 Agent，代码是 Harness。
```

模型负责推理、规划、选择工具和生成最终答案；代码负责提供稳定的工具、记忆、权限、观察结果、上下文预算和执行结构。工具不应该偷偷调用 LLM，也不应该把任务策略写死在内部。

详细技术设计和后续规划见 [TECHNICAL_DESIGN.zh-CN.md](TECHNICAL_DESIGN.zh-CN.md)。

## 当前能力

- 支持 ReAct、Plan-and-Execute、LLMCompiler 风格循环和图编排。
- 支持 OpenAI、Anthropic、Ollama，以及 Azure OpenAI、DeepSeek、OpenRouter、SiliconFlow、Qwen/DashScope 等 OpenAI-compatible 服务。
- 提供 Typer CLI，可通过 `ai-researcher` 或 `python -m ai_researcher_assistant.cli` 使用。
- 新增 `ai-researcher chat` 持续会话模式，支持单工作目录长期运行。
- 支持 Python Skill，也支持 Claude Code / Codex 兼容的 Markdown Skill。
- 支持本地父子 Agent 委托：父 Agent 通过 `subagent_task` 分派任务，子 Agent 使用隔离上下文工作，最后只把 summary 返回给父 Agent。
- 新增 `rag_search` 内置 Skill，让主循环和子 Agent 都能检索当前本地 RAG 记忆。
- RAG 默认使用本地内存向量库和确定性 hash embedding，无需外部 embedding 服务即可运行。
- RAG 支持向量检索、BM25 风格关键词检索、混合检索、过滤、轻量 rerank、引用元数据和引用图导出。
- 新增 `rag ingest-pdf`，可以把本地 PDF 或直接 PDF URL 转成 JSONL 论文记录并写入 RAG。

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

## CLI 使用

```bash
ai-researcher version --json
ai-researcher doctor --json
ai-researcher providers --json
ai-researcher skills list --json
ai-researcher skills inspect ./my-skills/literature-review/SKILL.md --json
ai-researcher rag ingest-pdf ./papers/example.pdf --output papers.jsonl --title "Example Paper"
ai-researcher rag search "hybrid retrieval" --paper-jsonl papers.jsonl --json
ai-researcher rag graph --paper-jsonl papers.jsonl --json
ai-researcher chat --cwd .
ai-researcher ask "查找 RAG 在科研发现中的应用论文" --provider openai --model gpt-4o
```

如果 `ai-researcher` 不在 PATH 中，也可以直接运行：

```bash
python -m ai_researcher_assistant.cli doctor --json
python -m ai_researcher_assistant.cli chat --cwd .
```

## 持续会话模式

`ai-researcher chat` 会在一个工作目录中持续运行，体验更接近 Claude Code 这类命令行工具。会话历史保存在当前工作目录的 `.ai-researcher/history.txt`。

常用命令：

```text
/help
/pwd
/skills
/stats
/model
/model openai gpt-4o
/rag load papers.jsonl
/rag ingest-pdf ./papers/example.pdf
/rag ingest-pdf https://example.org/paper.pdf
/rag search retrieval agents
/clear
/exit
```

普通文本会发送给 Agent。如果当前 session 已加载本地论文库，系统会自动把相关 RAG 上下文附加到本轮任务中。

## 添加本地 PDF 到 RAG

详细步骤：

1. 把 PDF 放到工作目录，例如 `papers/example.pdf`。
2. 将 PDF 解析成 JSONL 论文库：

```bash
ai-researcher rag ingest-pdf ./papers/example.pdf --output papers.jsonl --title "Example Paper" --author "A. Researcher" --category cs.CL
```

3. 搜索这个本地论文库：

```bash
ai-researcher rag search "retrieval augmented generation" --paper-jsonl papers.jsonl --include-full-text --json
```

4. 在持续会话中加载并使用：

```bash
ai-researcher chat --cwd .
/rag load papers.jsonl
/rag search retrieval augmented generation
请总结本地论文中关于 retrieval 的方法差异。
```

`ingest-pdf` 也支持直接 PDF URL。因此如果论文不在 arXiv，但能拿到 PDF 链接，也可以直接入库：

```bash
ai-researcher rag ingest-pdf https://example.org/paper.pdf --output papers.jsonl --title "External Paper"
```

## 不限于 arXiv 的论文来源设计

当前内置搜索 Skill 是 `arxiv_fetcher`，但架构并不限制论文来源。建议把每个外部来源做成独立 `BaseSkill`，例如：

- `semantic_scholar_fetcher`
- `crossref_fetcher`
- `pubmed_fetcher`
- `openalex_fetcher`
- `publisher_pdf_fetcher`
- `institutional_repository_fetcher`

推荐返回统一论文结构：

```python
{
    "success": True,
    "result": {
        "papers": [
            {
                "title": "...",
                "abstract": "...",
                "authors": ["..."],
                "source": "semantic_scholar",
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

设计边界：

- Source Skill 只负责检索、下载候选元数据、规范化字段。
- Source Skill 不调用 LLM。
- PDF 解析交给 `paper_reader` 或 `rag ingest-pdf`。
- 写入 RAG 时统一转换成 `title`、`abstract`、`full_text`、`authors`、`categories`、`published_date`、`citations`、`metadata`。
- 需要 API key 的来源应通过配置或环境变量传入，不要在 Skill 中读取私有文件。

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

## 本地父子 Agent

项目现在支持受 `learn-claude-code` s04 启发的本地父子 Agent 工作方式。父 Agent 可以调用 `subagent_task`，Harness 会创建一个新的子 ReActLoop、独立对话上下文和受限 SkillRegistry。子 Agent 完成任务后，只把结构化 summary 作为 observation 返回给父 Agent，父 Agent 不接收子 Agent 的完整中间上下文。

默认子 Agent 角色：

```text
literature_search_agent  -> arxiv_fetcher, rag_search
paper_reading_agent      -> paper_reader, rag_search
rag_retrieval_agent      -> rag_search
writing_agent            -> paper_writer, rag_search
```

这个设计适合把文献检索、论文阅读、本地 RAG 检索、写作草稿等任务拆给专门角色处理，同时保持父 Agent 的上下文干净。当前实现是本地进程内子 Agent，不是 MCP 或 A2A 远程协议接入。

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

推荐提交前运行：

```bash
make all
python -m ai_researcher_assistant.cli doctor --json
python -m ai_researcher_assistant.cli chat --help
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
