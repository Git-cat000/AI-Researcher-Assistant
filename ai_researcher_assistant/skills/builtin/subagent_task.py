"""Skill for delegating work to local child agents."""

from __future__ import annotations

from typing import Any

from ai_researcher_assistant.orchestration.subagent import SubagentTask
from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class SubagentTaskSkill(BaseSkill):
    """Delegate a focused task to a local child agent and return only its summary."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="subagent_task",
            description="Delegate a focused research task to a child agent with an isolated context.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="agent_name",
                    description=(
                        "Child agent role: literature_search_agent, paper_reading_agent, "
                        "rag_retrieval_agent, or writing_agent"
                    ),
                    type="string",
                    required=False,
                    default="literature_search_agent",
                ),
                SkillParameter(
                    name="task", description="Focused task for the child agent", type="string", required=True
                ),
                SkillParameter(
                    name="expected_output",
                    description="What the child agent should return to the parent",
                    type="string",
                    required=False,
                    default="Return a concise, evidence-grounded summary.",
                ),
                SkillParameter(
                    name="allowed_skills",
                    description="Optional explicit list of skills the child may use",
                    type="list",
                    required=False,
                    default=None,
                ),
                SkillParameter(
                    name="max_steps",
                    description="Maximum ReAct steps for the child agent",
                    type="integer",
                    required=False,
                    default=6,
                ),
            ],
            tags=["agent", "delegation", "subagent", "harness"],
            instructions=(
                "Use this skill when a task can be isolated into focused research work. "
                "The child agent uses a fresh context and returns only a summary to keep the parent context clean."
            ),
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "result": None,
            "error": "subagent_task requires the async harness. Use aexecute via an agent loop.",
        }

    async def aexecute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        runner = context.get("subagent_runner")
        if runner is None:
            return {"success": False, "result": None, "error": "No SubagentRunner is available in context"}

        allowed_skills = parameters.get("allowed_skills")
        if isinstance(allowed_skills, str):
            allowed_skills = [allowed_skills]

        task = SubagentTask(
            agent_name=parameters.get("agent_name", "literature_search_agent"),
            task=parameters["task"],
            expected_output=parameters.get("expected_output", "Return a concise, evidence-grounded summary."),
            allowed_skills=allowed_skills,
            max_steps=int(parameters.get("max_steps", 6)),
        )
        result = await runner.run(task)
        return {
            "success": result.status == "completed",
            "result": result.to_dict(),
            "error": result.error,
        }
