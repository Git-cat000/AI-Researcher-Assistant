# AI Researcher Assistant  
**An AI-powered research assistant for daily paper updates, weather, and academic news**  


## 基础架构
\text
physicist-agent/                      
├── physicist_agent/                  # 核心源码包
│   ├── __init__.py
│   ├── core/                         # 基础抽象
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── message.py
│   │   ├── config.py
│   │   └── exceptions.py
│   │
│   ├── llm/                          # LLM 适配层
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── local.py
│   │
│   ├── memory/                       # 记忆系统
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── short_term.py
│   │   ├── long_term.py
│   │   └── rag.py                    # 论文向量检索
│   │
│   ├── skills/                       # 技能系统
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── loader.py
│   │   ├── registry.py
│   │   ├── builtin/                  
│   │   │   ├── arxiv_fetcher.py      # 🆕 论文抓取技能
│   │   │   ├── paper_reader.py       # 🆕 PDF 解析技能
│   │   │   ├── paper_writer.py       # 🆕 论文写作润色技能
│   │   │   ├── web_search.py
│   │   │   └── code_interpreter.py   # 公式推导可用
│   │   └── templates/
│   │
│   ├── orchestration/                # 编排层
│   │   ├── __init__.py
│   │   ├── loop.py
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── middleware.py
│   │
│   └── utils/                        # 工具
│       ├── __init__.py
│       ├── logging.py
│       ├── telemetry.py
│       └── sandbox.py
│
├── scripts/                          # 辅助脚本
│   ├── sync_arxiv_daily.py           # 🆕 每日自动同步 arXiv等相关论文
│   └── build_knowledge_base.py       # 🆕 构建初始知识库
│
├── tests/
├── examples/
│   ├── 01_basic_qa.py
│   ├── 02_paper_search.py
│   ├── 03_writing_assistant.py
│   └── 04_full_physicist.py          # 🆕 完整物理学家 Agent 示例
│
├── docs/
├── pyproject.toml
├── README.md
└── .env.example
\text

## 📌 预期功能说明  
1. **启动问候**  
   - 开机后自动与用户打招呼，支持自定义用户名（如“你好，张三！”）  
   - 考虑语音交互功能 

2. **天气更新**  
   - 自动获取用户指定城市的实时天气信息  
   - 集成主流天气 API

3. **学术动态追踪（核心功能）**  
   - 自动抓取指定领域的最新论文（支持 [arXiv](https://arxiv.org/ )、[Semantic Scholar](https://www.semanticscholar.org/ ) 等平台）  
   - 支持关键词订阅（如“NLP”、“量子计算”等学科分类）  
   - 生成论文摘要并AI助手推送通知

## 🚀 未来扩展计划  
- 集成 GitHub Issues 跟踪（记录每日科研任务）  
- 支持多语言界面（中/英文切换）  
- 添加 GUI 界面展示天气和论文信息  
