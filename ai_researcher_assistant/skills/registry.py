"""Skill registry.

Registries are ordinary instances so tests and agent instances can be isolated.
`get_skill_registry()` remains as a compatibility helper for older code.
"""

import logging
from typing import Any

from ai_researcher_assistant.core.exceptions import SkillError, ToolNotFoundError
from ai_researcher_assistant.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Register, inspect, and execute available skills."""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        name = skill.manifest.name
        if name in self._skills:
            logger.warning("Skill '%s' already registered, overwriting.", name)
        self._skills[name] = skill
        logger.info("Registered skill: %s", name)

    def register_many(self, skills: list[BaseSkill]) -> None:
        for skill in skills:
            self.register(skill)

    def unregister(self, name: str) -> bool:
        if name in self._skills:
            del self._skills[name]
            logger.info("Unregistered skill: %s", name)
            return True
        return False

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def get_all_manifests(self) -> dict[str, Any]:
        return {name: skill.manifest for name, skill in self._skills.items()}

    def get_instructions_for_all(self) -> str:
        if not self._skills:
            return "No skills available."
        return "\n\n".join(skill.get_instructions_for_llm() for skill in self._skills.values())

    def execute(self, skill_name: str, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        skill = self.get(skill_name)
        if skill is None:
            raise ToolNotFoundError(f"Skill '{skill_name}' not found")

        self._apply_defaults(skill, parameters, skill_name)
        try:
            return skill.execute(parameters, context)
        except Exception as exc:
            logger.exception("Error executing skill '%s'", skill_name)
            return {"success": False, "error": str(exc), "result": None}

    async def aexecute(self, skill_name: str, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        skill = self.get(skill_name)
        if skill is None:
            raise ToolNotFoundError(f"Skill '{skill_name}' not found")

        self._apply_defaults(skill, parameters, skill_name)
        try:
            return await skill.aexecute(parameters, context)
        except Exception as exc:
            logger.exception("Error executing skill '%s'", skill_name)
            return {"success": False, "error": str(exc), "result": None}

    def clear(self) -> None:
        self._skills.clear()
        logger.info("Cleared all registered skills")

    def _apply_defaults(self, skill: BaseSkill, parameters: dict[str, Any], skill_name: str) -> None:
        for param in skill.manifest.parameters:
            if param.required and param.name not in parameters:
                if param.default is not None:
                    parameters[param.name] = param.default
                else:
                    raise SkillError(f"Missing required parameter '{param.name}' for skill '{skill_name}'")


_default_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """Return the compatibility process-default skill registry."""

    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry
