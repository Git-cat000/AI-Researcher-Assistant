"""
技能注册表。
管理所有可用技能，支持注册、查询和调用。
"""
from typing import Dict, List, Optional, Any
import logging

from ai_researcher_assistant.skills.base import BaseSkill
from ai_researcher_assistant.core.exceptions import SkillError, ToolNotFoundError

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    技能注册表。
    单例模式，全局管理技能。
    """

    _instance: Optional["SkillRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: Dict[str, BaseSkill] = {}
            cls._instance._initialized = False
        return cls._instance

    def register(self, skill: BaseSkill) -> None:
        """
        注册一个技能。
        
        Args:
            skill: 技能实例
        """
        name = skill.manifest.name
        if name in self._skills:
            logger.warning(f"Skill '{name}' already registered, overwriting.")
        self._skills[name] = skill
        logger.info(f"Registered skill: {name}")

    def register_many(self, skills: List[BaseSkill]) -> None:
        """批量注册技能"""
        for skill in skills:
            self.register(skill)

    def unregister(self, name: str) -> bool:
        """注销技能"""
        if name in self._skills:
            del self._skills[name]
            logger.info(f"Unregistered skill: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[BaseSkill]:
        """获取指定名称的技能"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有已注册技能的名称"""
        return list(self._skills.keys())

    def get_all_manifests(self) -> Dict[str, Any]:
        """获取所有技能清单（用于 LLM 上下文）"""
        return {name: skill.manifest for name, skill in self._skills.items()}

    def get_instructions_for_all(self) -> str:
        """生成所有技能的指令文本，用于注入到系统提示中"""
        if not self._skills:
            return "No skills available."
        
        sections = []
        for skill in self._skills.values():
            sections.append(skill.get_instructions_for_llm())
            sections.append("")
        
        return "\n".join(sections)

    def execute(self, skill_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定的技能。
        
        Args:
            skill_name: 技能名称
            parameters: 参数
            context: 执行上下文
            
        Returns:
            执行结果
        """
        skill = self.get(skill_name)
        if skill is None:
            raise ToolNotFoundError(f"Skill '{skill_name}' not found")
        
        # 验证参数
        manifest = skill.manifest
        for param in manifest.parameters:
            if param.required and param.name not in parameters:
                if param.default is not None:
                    parameters[param.name] = param.default
                else:
                    raise SkillError(f"Missing required parameter '{param.name}' for skill '{skill_name}'")
        
        try:
            return skill.execute(parameters, context)
        except Exception as e:
            logger.exception(f"Error executing skill '{skill_name}'")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }

    async def aexecute(self, skill_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行技能"""
        skill = self.get(skill_name)
        if skill is None:
            raise ToolNotFoundError(f"Skill '{skill_name}' not found")
        
        manifest = skill.manifest
        for param in manifest.parameters:
            if param.required and param.name not in parameters:
                if param.default is not None:
                    parameters[param.name] = param.default
                else:
                    raise SkillError(f"Missing required parameter '{param.name}' for skill '{skill_name}'")
        
        try:
            return await skill.aexecute(parameters, context)
        except Exception as e:
            logger.exception(f"Error executing skill '{skill_name}'")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }

    def clear(self) -> None:
        """清空所有注册的技能"""
        self._skills.clear()
        logger.info("Cleared all registered skills")


def get_skill_registry() -> SkillRegistry:
    """获取全局技能注册表单例"""
    return SkillRegistry()
