"""ResearcherAgent facade for the LLM harness."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from ai_researcher_assistant.core.base_agent import BaseAgent
from ai_researcher_assistant.core.config import AgentConfig
from ai_researcher_assistant.core.message import Conversation
from ai_researcher_assistant.llm import BaseLLM, create_llm
from ai_researcher_assistant.memory import AcademicRAG, ShortTermMemory
from ai_researcher_assistant.orchestration.loop import ReActLoop
from ai_researcher_assistant.orchestration.middleware import LoggingMiddleware, MiddlewareManager, TelemetryMiddleware
from ai_researcher_assistant.orchestration.state import ExecutionState
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Academic research agent assembled around a stable harness loop."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        name: str = "AI Researcher Assistant",
        llm: BaseLLM | None = None,
        llm_factory: Callable[[Any], BaseLLM] = create_llm,
        enable_rag: bool = True,
        enable_builtin_skills: bool = True,
        skill_registry: SkillRegistry | None = None,
    ):
        super().__init__(config, name)
        self._llm_factory = llm_factory
        self._provided_llm = llm
        self._enable_rag = enable_rag
        self._enable_builtin_skills = enable_builtin_skills

        self.llm: BaseLLM | None = llm
        self.short_term_memory = ShortTermMemory(max_tokens=self.config.memory.short_term_max_tokens)
        self.rag: AcademicRAG | None = None
        self.skill_registry = skill_registry or SkillRegistry()
        self.middleware = MiddlewareManager()
        self.telemetry: TelemetryMiddleware | None = None
        self.loop: ReActLoop | None = None
        self.execution_state: ExecutionState | None = None

    def initialize(self) -> None:
        """Initialize runtime resources without creating RAG until it is needed."""

        if self._initialized:
            return

        self.llm = self.llm or self._llm_factory(self.config.llm)
        if self._enable_builtin_skills and not self.skill_registry.list_skills():
            self._load_builtin_skills()

        self._setup_middleware()
        self.loop = ReActLoop(
            llm=self.llm,
            skill_registry=self.skill_registry,
            middleware_manager=self.middleware,
        )
        self._initialized = True
        logger.info("ResearcherAgent '%s' initialized", self.name)

    async def start(self) -> None:
        """Async lifecycle alias for harness-friendly callers."""

        self.initialize()

    async def stop(self) -> None:
        """Async lifecycle alias for harness-friendly callers."""

        self.shutdown()

    def _load_builtin_skills(self) -> None:
        loader = SkillLoader(self.skill_registry)
        loader.load_builtin_skills()
        logger.info("Loaded %d built-in skills", len(self.skill_registry.list_skills()))

    def _setup_middleware(self) -> None:
        if self.config.enable_telemetry:
            self.middleware.add(LoggingMiddleware())
            self.telemetry = TelemetryMiddleware()
            self.middleware.add(self.telemetry)

    def process(self, user_input: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aprocess(user_input))
        raise RuntimeError("ResearcherAgent.process() cannot run inside an active event loop; use await aprocess().")

    async def aprocess(self, user_input: str) -> str:
        if not self._initialized:
            self.initialize()
        if self.loop is None:
            raise RuntimeError("ResearcherAgent failed to initialize its execution loop.")

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

    def _build_context(self) -> dict[str, Any]:
        return {
            "llm": self.llm,
            "conversation": self._build_conversation(),
            "short_term_memory": self.short_term_memory,
            "rag": self.rag,
            "config": self.config,
            "skill_registry": self.skill_registry,
        }

    def _build_conversation(self) -> Conversation:
        conv = Conversation()
        if self.loop is not None:
            conv.add_system(self.loop.build_system_prompt())
        for msg in self.short_term_memory.get_messages():
            metadata = dict(msg.metadata)
            metadata.pop("role", None)
            conv.add(msg.role, msg.content, **metadata)
        return conv

    def _ensure_rag(self) -> AcademicRAG | None:
        if not self._enable_rag:
            return None
        if self.rag is None:
            self.rag = AcademicRAG()
        return self.rag

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        rag = self._ensure_rag()
        return rag.search_papers(query, top_k=top_k) if rag else []

    def add_paper_to_knowledge_base(
        self,
        title: str,
        abstract: str,
        full_text: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        rag = self._ensure_rag()
        return rag.add_paper(title, abstract, full_text, **kwargs) if rag else []

    def get_execution_history(self) -> list[dict[str, Any]]:
        if self.execution_state:
            return [step.to_dict() for step in self.execution_state.steps]
        return []

    def get_stats(self) -> dict[str, Any]:
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

    def reset_conversation(self) -> None:
        super().reset_conversation()
        self.short_term_memory.clear()
        self.execution_state = None

    def shutdown(self) -> None:
        logger.info("Shutting down ResearcherAgent '%s'", self.name)
