"""Local parent-child agent orchestration.

Subagents run with a fresh conversation and return only a structured summary to
the parent harness. This mirrors the context-isolation idea from
learn-claude-code s04 without introducing remote A2A/MCP transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_researcher_assistant.core.config import AgentConfig
from ai_researcher_assistant.llm import BaseLLM
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.orchestration.loop import LoopConfig, ReActLoop
from ai_researcher_assistant.skills import SkillRegistry


@dataclass(frozen=True)
class SubagentSpec:
    """Configuration for one local child agent role."""

    name: str
    description: str
    allowed_skills: tuple[str, ...] = ()
    system_prompt: str = ""


@dataclass
class SubagentTask:
    """Task delegated from a parent agent to a child agent."""

    agent_name: str
    task: str
    expected_output: str = "Return a concise, evidence-grounded summary."
    allowed_skills: list[str] | None = None
    max_steps: int = 6
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentResult:
    """Summary-only result returned from a child agent to the parent."""

    agent_name: str
    task: str
    status: str
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps_taken: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "task": self.task,
            "status": self.status,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "steps_taken": self.steps_taken,
            "error": self.error,
        }


DEFAULT_SUBAGENTS: dict[str, SubagentSpec] = {
    "literature_search_agent": SubagentSpec(
        name="literature_search_agent",
        description="Searches paper sources and returns candidate papers with selection rationale.",
        allowed_skills=("arxiv_fetcher", "rag_search"),
        system_prompt=(
            "Focus on finding relevant academic papers. Return search terms, candidate papers, and why they matter."
        ),
    ),
    "paper_reading_agent": SubagentSpec(
        name="paper_reading_agent",
        description="Reads one PDF or paper record and extracts contributions, method, evidence, and limitations.",
        allowed_skills=("paper_reader", "rag_search"),
        system_prompt="Focus on careful paper reading. Return only key claims, method, evidence, and limitations.",
    ),
    "rag_retrieval_agent": SubagentSpec(
        name="rag_retrieval_agent",
        description="Searches the local paper memory and returns relevant snippets and paper IDs.",
        allowed_skills=("rag_search",),
        system_prompt="Focus on local RAG retrieval. Return relevant papers, snippets, and citation hints.",
    ),
    "writing_agent": SubagentSpec(
        name="writing_agent",
        description="Drafts academic summaries or outlines from provided evidence.",
        allowed_skills=("paper_writer", "rag_search"),
        system_prompt="Focus on synthesis and writing. Do not invent sources beyond observations.",
    ),
}


class SubagentRunner:
    """Runs local child agents with isolated context and summary-only output."""

    def __init__(
        self,
        llm: BaseLLM,
        parent_skill_registry: SkillRegistry,
        rag: AcademicRAG | None = None,
        config: AgentConfig | None = None,
        specs: dict[str, SubagentSpec] | None = None,
    ) -> None:
        self.llm = llm
        self.parent_skill_registry = parent_skill_registry
        self.rag = rag
        self.config = config
        self.specs = specs or DEFAULT_SUBAGENTS

    async def run(self, task: SubagentTask) -> SubagentResult:
        spec = self.specs.get(task.agent_name)
        if spec is None:
            return SubagentResult(
                agent_name=task.agent_name,
                task=task.task,
                status="failed",
                summary="",
                error=f"Unknown subagent: {task.agent_name}",
            )

        child_registry = self._build_child_registry(task, spec)
        loop = ReActLoop(
            llm=self.llm,
            skill_registry=child_registry,
            config=LoopConfig(max_steps=max(1, task.max_steps), temperature=0.0),
        )
        loop.base_system_prompt = self._build_child_system_prompt(spec)
        child_context: dict[str, Any] = {
            "llm": self.llm,
            "rag": self.rag,
            "config": self.config,
            "skill_registry": child_registry,
            "token_budget": {"max_observation_tokens": 1200},
            "subagent": spec.name,
        }

        delegated_prompt = self._build_task_prompt(task, spec)
        state = await loop.run(delegated_prompt, child_context)
        summary = state.final_answer or ""
        status = "completed" if state.status.value == "completed" else "failed"
        cost_tracker: Any = child_context.get("cost_tracker")
        to_dict = getattr(cost_tracker, "to_dict", None)
        cost = to_dict() if callable(to_dict) else None
        error = None if status == "completed" else (state.final_answer or "Subagent failed")

        return SubagentResult(
            agent_name=spec.name,
            task=task.task,
            status=status,
            summary=summary,
            artifacts={
                "expected_output": task.expected_output,
                "allowed_skills": child_registry.list_skills(),
                "cost": cost,
            },
            steps_taken=state.current_step,
            error=error,
        )

    def _build_child_registry(self, task: SubagentTask, spec: SubagentSpec) -> SkillRegistry:
        child_registry = SkillRegistry()
        requested = tuple(task.allowed_skills) if task.allowed_skills else spec.allowed_skills
        skill_names = requested or tuple(self.parent_skill_registry.list_skills())
        for skill_name in skill_names:
            if skill_name == "subagent_task":
                continue
            skill = self.parent_skill_registry.get(skill_name)
            if skill is not None:
                child_registry.register(skill)
        return child_registry

    def _build_child_system_prompt(self, spec: SubagentSpec) -> str:
        return f"""You are a focused child research agent named `{spec.name}`.

Role:
{spec.description}

Specialization:
{spec.system_prompt}

Available skills:
{{skill_instructions}}

You run in an isolated child context. The parent agent will only receive your final summary.
Do not mention hidden chain-of-thought. Return concise evidence, artifacts, and uncertainty.
"""

    def _build_task_prompt(self, task: SubagentTask, spec: SubagentSpec) -> str:
        return f"""Delegated subagent task for {spec.name}:

Task:
{task.task}

Expected output:
{task.expected_output}

Return a final answer that the parent agent can use directly as a summary.
"""
