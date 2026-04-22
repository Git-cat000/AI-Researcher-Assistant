"""
AI Researcher Assistant - 智能学术研究助手框架

一个专为学术研究设计的模块化 Agent 框架，支持：
- 多 LLM 适配（OpenAI、Anthropic、Ollama）
- RAG 论文知识库
- arXiv 抓取、PDF 阅读、论文写作等内置技能
- ReAct / Plan-Execute / 图编排等多种执行模式
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 核心模块
from ai_researcher_assistant.core import (
    AgentConfig,
    LLMConfig,
    MemoryConfig,
    SkillsConfig,
    get_config,
    update_config,
    BaseAgent,
    Message,
    Conversation,
    MessageRole,
    AgentError,
    LLMError,
    SkillError,
    MemoryError,
    ConfigurationError,
)

# LLM 适配层
from ai_researcher_assistant.llm import (
    BaseLLM,
    LLMResponse,
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    create_llm,
    get_llm,
)

# 记忆系统
from ai_researcher_assistant.memory import (
    BaseMemory,
    BaseEmbedding,
    MemoryItem,
    ShortTermMemory,
    LongTermMemory,
    OpenAIEmbedding,
    AcademicRAG,
)

# 技能系统
from ai_researcher_assistant.skills import (
    BaseSkill,
    SkillManifest,
    SkillParameter,
    SkillRegistry,
    get_skill_registry,
    SkillLoader,
)

# 编排层
from ai_researcher_assistant.orchestration import (
    ResearcherAgent,
    ReActLoop,
    PlanAndExecuteLoop,
    LLMCompilerLoop,
    LoopType,
    LoopConfig,
    create_loop,
    ExecutionState,
    ExecutionStep,
    ExecutionStatus,
    Middleware,
    MiddlewareManager,
    LoggingMiddleware,
    TelemetryMiddleware,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Core
    "AgentConfig",
    "LLMConfig",
    "MemoryConfig",
    "SkillsConfig",
    "get_config",
    "update_config",
    "BaseAgent",
    "Message",
    "Conversation",
    "MessageRole",
    "AgentError",
    "LLMError",
    "SkillError",
    "MemoryError",
    "ConfigurationError",
    # LLM
    "BaseLLM",
    "LLMResponse",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "create_llm",
    "get_llm",
    # Memory
    "BaseMemory",
    "BaseEmbedding",
    "MemoryItem",
    "ShortTermMemory",
    "LongTermMemory",
    "OpenAIEmbedding",
    "AcademicRAG",
    # Skills
    "BaseSkill",
    "SkillManifest",
    "SkillParameter",
    "SkillRegistry",
    "get_skill_registry",
    "SkillLoader",
    # Orchestration
    "ResearcherAgent",
    "ReActLoop",
    "PlanAndExecuteLoop",
    "LLMCompilerLoop",
    "LoopType",
    "LoopConfig",
    "create_loop",
    "ExecutionState",
    "ExecutionStep",
    "ExecutionStatus",
    "Middleware",
    "MiddlewareManager",
    "LoggingMiddleware",
    "TelemetryMiddleware",
]