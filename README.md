# AI Researcher Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

一个专为**学术研究辅助**设计的智能、模块化 Agent 框架。  
它能够搜索 arXiv 论文、阅读解析 PDF、维护基于向量数据库的知识库（RAG），并协助进行学术写作和润色——所有功能均由大语言模型驱动。

---

## ✨ 特性

- 🧠 **模块化架构** – 核心、LLM 适配器、记忆、技能、编排各层清晰分离。
- 🤖 **多 LLM 支持** – 兼容 OpenAI、Anthropic Claude 以及通过 Ollama 运行的本地模型。
- 📚 **学术技能内置** – 内置 arXiv 论文抓取、PDF 阅读、论文写作与润色等技能。
- 🔍 **RAG 记忆系统** – 将论文存入向量数据库（ChromaDB），支持语义检索。
- 🔄 **灵活的编排方式** – 提供 ReAct、Plan-and-Execute、LLMCompiler 三种执行循环，也支持类似 LangGraph 的图式工作流。
- 🛠️ **易于扩展** – 通过编写简单的 Python 类或 Markdown 文件即可添加自定义技能。
- 📊 **可观测性** – 内置日志与遥测中间件，可追踪执行步骤与 Token 消耗。

---

## 📦 安装

### 环境要求

- Python 3.10 或更高版本
- 至少一个 LLM 服务商的 API 密钥（OpenAI 或 Anthropic）

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/Git-cat000/AI-Researcher-Assistant-assistant.git
cd AI-Researcher-Assistant

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate   # Windows 下使用: venv\Scripts\activate

# 以可编辑模式安装，同时安装依赖
pip install -e .
```

### 可选依赖

- 开发和测试依赖：

  ```bash
  pip install -e ".[dev]"
  ```

- 如需使用本地模型（Ollama）：

  ```bash
  pip install ollama   # 或参考 https://ollama.ai 的说明
  ```

---

## ⚙️ 配置

1. 复制示例环境变量文件：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填入你的 API 密钥：

   ```env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   OLLAMA_BASE_URL=http://localhost:11434   # 如使用 Ollama
   ```

3. 也可以在代码中以编程方式修改配置：

   ```python
   from ai_researcher_assistant.core.config import update_config

   update_config(
       llm={"provider": "openai", "model": "gpt-4o", "temperature": 0.0},
       memory={"vector_db_path": "./my_data/vector_db"},
       data_dir="./my_data"
   )
   ```

---

## 🚀 快速开始

### 1. 基础科研问答

```python
import asyncio
from ai_researcher_assistant.orchestration import ResearcherAgent

async def main():
    agent = ResearcherAgent(name="物理研究助手")
    agent.initialize()
    
    question = "AdS/CFT 对偶最近有哪些进展？请找两篇近期论文并总结。"
    answer = await agent.aprocess(question)
    
    print(answer)
    agent.shutdown()

asyncio.run(main())
```

### 2. 使用 RAG 知识库

```python
agent = ResearcherAgent(name="文献管理员", enable_rag=True)
agent.initialize()

# 向知识库中添加一篇论文
agent.add_paper_to_knowledge_base(
    title="凝聚态物理中的全息对偶",
    abstract="我们将全息方法应用于奇异金属和高温超导体的建模...",
    arxiv_id="2301.12345",
    authors=["J. Maldacena", "S. Hartnoll"]
)

# 提出一个会触发 RAG 检索的问题
answer = await agent.aprocess("刚才添加的那篇全息对偶论文里关于奇异金属说了什么？")
```

### 3. 学术写作辅助

```python
answer = await agent.aprocess(
    "请将以下文本润色成适合物理学期刊的学术风格："
    "'我们算了一下，发现黑洞不会丢失信息。'"
)
```

更多示例请查看 `examples/` 目录。

---

## 🧱 架构概览

框架采用清晰的分层设计：

| 模块                 | 描述                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `core/`              | 基类、配置管理、消息结构、异常定义。                                                          |
| `llm/`               | OpenAI、Anthropic 及本地模型的适配器，通过 `BaseLLM` 提供统一接口。                            |
| `memory/`            | 短期记忆（对话历史）与长期记忆（向量数据库），以及专用于论文的 `AcademicRAG`。                  |
| `skills/`            | 封装的能力（如 `ArxivFetcherSkill`、`PaperReaderSkill`），包含注册表与加载器。                |
| `orchestration/`     | 执行循环（ReAct、Plan-Execute、LLMCompiler）、状态管理、中间件，以及最终的 `ResearcherAgent`。 |

执行流程示意：

```
用户输入 → ResearcherAgent → ReActLoop（或图执行）→ 技能调用 → LLM 推理 → 最终答案
                                      ↓
                            （短期记忆 & RAG 记忆）
```

---

## 🛠️ 添加自定义技能

### 方式一：编写 Python 类

```python
from ai_researcher_assistant.skills import BaseSkill, SkillManifest, SkillParameter

class MyCustomSkill(BaseSkill):
    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill",
            description="执行某个有用操作。",
            parameters=[
                SkillParameter("query", "string", required=True, description="搜索查询")
            ]
        )

    def execute(self, parameters, context):
        query = parameters["query"]
        # 在此处实现你的逻辑
        return {"success": True, "result": f"已搜索 {query}"}
```

### 方式二：Markdown 定义（即将支持）

在技能目录中放置一个 `SKILL.md` 文件，可参考 `skills/templates/` 下的模板。

然后注册你的技能：

```python
from ai_researcher_assistant.skills import get_skill_registry, SkillLoader

registry = get_skill_registry()
loader = SkillLoader(registry)
loader.load_from_directory("./my_skills")
```

---

## 🧪 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 🤝 贡献指南

欢迎贡献！请参阅 `CONTRIBUTING.md` 了解以下内容：

- 报告 Bug
- 提出新功能建议
- 提交 Pull Request 的流程
- 代码规范

提交 PR 前，请确保运行：

```bash
black ai_researcher_assistant/
ruff check ai_researcher_assistant/
pytest
```

---

## 📄 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。

---

## 🙏 致谢

- 灵感来源于 **LangChain**、**CrewAI** 和 **AutoGen** 等优秀框架。
- 向量存储使用了 **ChromaDB**。
- 构建在 **OpenAI**、**Anthropic** 和 **Ollama** 的 API 之上。

---


如有问题或建议，请在 GitHub 上提交 Issue。

*祝研究愉快！🧪📖*
