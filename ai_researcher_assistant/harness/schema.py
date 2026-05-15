"""Stable action, observation, permission, and budget contracts for the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_researcher_assistant.core.message import estimate_tokens, truncate_to_token_budget
from ai_researcher_assistant.llm.base import LLMResponse


@dataclass
class Action:
    """A normalized skill invocation requested by an LLM."""

    skill: str
    parameters: dict[str, Any] = field(default_factory=dict)
    thought: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], thought: str | None = None) -> Action:
        skill = data.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            raise ValueError("Action must include a non-empty string field: skill")
        parameters = data.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError("Action parameters must be an object")
        metadata = {key: value for key, value in data.items() if key not in {"skill", "parameters"}}
        return cls(skill=skill.strip(), parameters=parameters, thought=thought, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        payload = {"skill": self.skill, "parameters": self.parameters}
        if self.thought:
            payload["thought"] = self.thought
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class Observation:
    """Structured result returned to the LLM after executing an action."""

    skill: str
    success: bool
    content: str
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_skill_result(cls, skill: str, result: dict[str, Any], content: str) -> Observation:
        return cls(
            skill=skill,
            success=bool(result.get("success")),
            content=content,
            result=result.get("result"),
            error=result.get("error"),
            metadata={key: value for key, value in result.items() if key.startswith("_")},
        )

    def to_prompt(self, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            return self.content
        return truncate_to_token_budget(self.content, max_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "success": self.success,
            "content": self.content,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "tokens": estimate_tokens(self.content),
        }


@dataclass
class PermissionPolicy:
    """Simple allow/block policy for harness-controlled skill execution."""

    allowed_skills: set[str] | None = None
    blocked_skills: set[str] = field(default_factory=set)
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_script_execution: bool = False

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> PermissionPolicy:
        raw = context.get("permission_policy")
        if isinstance(raw, PermissionPolicy):
            return raw
        if isinstance(raw, dict):
            allowed = raw.get("allowed_skills")
            blocked = raw.get("blocked_skills", [])
            return cls(
                allowed_skills=set(allowed) if allowed else None,
                blocked_skills=set(blocked),
                allow_network=bool(raw.get("allow_network", True)),
                allow_filesystem=bool(raw.get("allow_filesystem", True)),
                allow_script_execution=bool(raw.get("allow_script_execution", False)),
            )
        return cls()

    def check_action(self, action: Action) -> tuple[bool, str | None]:
        if self.allowed_skills is not None and action.skill not in self.allowed_skills:
            return False, f"Skill '{action.skill}' is not in the allowed skill list"
        if action.skill in self.blocked_skills:
            return False, f"Skill '{action.skill}' is blocked by the permission policy"
        return True, None


@dataclass
class TokenBudget:
    """Prompt and observation token limits enforced by the harness."""

    max_context_tokens: int | None = None
    max_observation_tokens: int | None = 2000

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> TokenBudget:
        raw = context.get("token_budget")
        if isinstance(raw, TokenBudget):
            return raw
        if isinstance(raw, dict):
            return cls(
                max_context_tokens=raw.get("max_context_tokens"),
                max_observation_tokens=raw.get("max_observation_tokens", 2000),
            )
        return cls()


@dataclass
class CostTracker:
    """Accumulates token usage reported by model adapters."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: LLMResponse) -> None:
        self.calls += 1
        usage = response.usage or {}
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += usage.get("total_tokens", prompt_tokens + completion_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def get_cost_tracker(context: dict[str, Any]) -> CostTracker:
    tracker = context.get("cost_tracker")
    if isinstance(tracker, CostTracker):
        return tracker
    tracker = CostTracker()
    context["cost_tracker"] = tracker
    return tracker
