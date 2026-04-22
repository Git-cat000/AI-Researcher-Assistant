"""
学术研究助手 Agent。
整合 LLM、记忆、技能和编排循环，提供完整的 Agent 功能。
"""
from typing import Optional, Dict, Any, List, AsyncIterator
import logging

from ai_researcher_assistant.core.base_agent import BaseAgent
from ai_researcher_assistant.core.config import AgentConfig, get_config
from ai_researcher_assistant.core.message import Conversation, MessageRole
from ai_researcher_assistant.llm import BaseLLM, create_llm
from ai_researcher_assistant.memory import ShortTermMemory, AcademicRAG
from ai_researcher_assistant.skills import SkillRegistry, get_skill_registry, SkillLoader
from ai_researcher_assistant.orchestration.loop import ReActLoop
from ai_researcher_assistant.orchestration.state import ExecutionState
from ai_researcher_assistant.orchestration.middleware import MiddlewareManager, LoggingMiddleware, TelemetryMiddleware

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    学术研究助手 Agent。
    
    特性：
    - 自动管理对话历史和短期记忆
    - 集成学术 RAG 知识库
    - 支持多种学术技能（arXiv 抓取、PDF 阅读、写作润色）
    - ReAct 自主推理循环
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        name: str = "AI Researcher Assistant",
        llm: Optional[BaseLLM] = None,
        enable_rag: bool = True,
        enable_builtin_skills: bool = True,
    ):
        super().__init__(config, name)
        
        self.llm = llm or create_llm(self.config.llm)
        
        self.short_term_memory = ShortTermMemory(max_tokens=self.config.memory.short_term_max_tokens)
        self.rag = AcademicRAG() if enable_rag else None
        
        self.skill_registry = get_skill_registry()
        if enable_builtin_skills:
            self._load_builtin_skills()
        
        self.middleware = MiddlewareManager()
        self._setup_middleware()
        
        self.loop = ReActLoop(
            llm=self.llm,
            skill_registry=self.skill_registry,
            middleware_manager=self.middleware,
        )
        
        self.execution_state: Optional[ExecutionState] = None
        self._initialized = True

    def _load_builtin_skills(self):
        loader = SkillLoader(self.skill_registry)
        loader.load_builtin_skills()
        logger.info(f"Loaded {len(self.skill_registry.list_skills())} built-in skills")

    def _setup_middleware(self):
        if self.config.enable_telemetry:
            self.middleware.add(LoggingMiddleware())
            self.telemetry = TelemetryMiddleware()
            self.middleware.add(self.telemetry)
        else:
            self.telemetry = None

    def initialize(self):
        super().initialize()
        logger.info(f"ResearcherAgent '{self.name}' initialized")

    def process(self, user_input: str) -> str:
        import asyncio
        return asyncio.run(self.aprocess(user_input))

    async def aprocess(self, user_input: str) -> str:
        self.short_term_memory.add(user_input, metadata={"role": "user"})
        
        context = self._build_context()
        
        self.execution_state = await self.loop.run(user_input, context)
        
        answer = self.execution_state.final_answer or "Sorry, I couldn't complete the task."
        
        self.short_term_memory.add(answer, metadata={"role": "assistant"})
        
        return answer

    async def stream_process(self, user_input: str) -> AsyncIterator[str]:
        answer = await self.aprocess(user_input)
        for char in answer:
            yield char

    def _build_context(self) -> Dict[str, Any]:
        return {
            "llm": self.llm,
            "conversation": self._build_conversation(),
            "short_term_memory": self.short_term_memory,
            "rag": self.rag,
            "config": self.config,
        }

    def _build_conversation(self) -> Conversation:
        conv = Conversation()
        conv.add_system(self.loop.build_system_prompt())
        for msg in self.short_term_memory.get_messages():
            conv.add(msg.role, msg.content, **msg.metadata)
        return conv

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.rag:
            return self.rag.search_papers(query, top_k=top_k)
        return []

    def add_paper_to_knowledge_base(
        self,
        title: str,
        abstract: str,
        full_text: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        if self.rag:
            return self.rag.add_paper(title, abstract, full_text, **kwargs)
        return []

    def get_execution_history(self) -> List[Dict[str, Any]]:
        if self.execution_state:
            return [step.to_dict() for step in self.execution_state.steps]
        return []

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "short_term_messages": self.short_term_memory.count(),
            "rag_papers": self.rag.count_papers() if self.rag else 0,
            "skills_available": self.skill_registry.list_skills(),
        }
        if self.telemetry:
            stats.update(self.telemetry.get_stats())
        if self.execution_state:
            stats["execution_status"] = self.execution_state.status.value
            stats["steps_taken"] = self.execution_state.current_step
        return stats

    def reset_conversation(self):
        super().reset_conversation()
        self.short_term_memory.clear()
        self.execution_state = None

    def shutdown(self):
        logger.info(f"Shutting down ResearcherAgent '{self.name}'")