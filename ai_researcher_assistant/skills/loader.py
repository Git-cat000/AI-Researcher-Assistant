"""
技能加载器。
支持从 Python 模块、Markdown 文件等加载技能。
"""
import os
import importlib
import importlib.util
from pathlib import Path
from typing import List, Optional, Type
import logging

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter
from ai_researcher_assistant.skills.markdown import MarkdownSkill
from ai_researcher_assistant.skills.registry import SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class SkillLoader:
    """技能加载器"""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or get_skill_registry()

    def load_from_class(self, skill_class: Type[BaseSkill]) -> BaseSkill:
        """从类加载技能并注册"""
        skill = skill_class()
        self.registry.register(skill)
        return skill

    def load_from_module(self, module_path: str) -> List[BaseSkill]:
        """
        从 Python 模块加载所有 BaseSkill 子类。
        
        Args:
            module_path: 模块路径，如 "ai_researcher_assistant.skills.buildin"
            
        Returns:
            加载的技能实例列表
        """
        module = importlib.import_module(module_path)
        skills = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseSkill) and attr is not BaseSkill:
                skill = self.load_from_class(attr)
                skills.append(skill)
        return skills

    def load_from_directory(self, directory: str) -> List[BaseSkill]:
        """
        从目录加载所有 Python 文件中的技能。
        
        Args:
            directory: 目录路径
            
        Returns:
            加载的技能实例列表
        """
        skills = []
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return skills

        if dir_path.is_file():
            if dir_path.suffix.lower() == ".md":
                return [self.load_from_markdown(dir_path)]
            logger.warning("Unsupported skill file: %s", directory)
            return skills

        root_skill = dir_path / "SKILL.md"
        if root_skill.exists():
            skill = self.load_from_markdown(root_skill)
            logger.info("Loaded Markdown skill from %s", root_skill)
            return [skill]

        seen_markdown: set[Path] = set()
        for skill_file in dir_path.rglob("SKILL.md"):
            seen_markdown.add(skill_file.resolve())
            skills.append(self.load_from_markdown(skill_file))

        for md_file in dir_path.glob("*.md"):
            if md_file.resolve() not in seen_markdown:
                skills.append(self.load_from_markdown(md_file))
        
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseSkill) and attr is not BaseSkill:
                        skill = self.load_from_class(attr)
                        skills.append(skill)
        
        logger.info(f"Loaded {len(skills)} skills from directory: {directory}")
        return skills

    def load_from_markdown(self, markdown_path: str | Path) -> BaseSkill:
        """Load a Claude Code / Codex compatible Markdown skill file."""

        skill = MarkdownSkill(markdown_path)
        self.registry.register(skill)
        return skill

    def load_builtin_skills(self) -> List[BaseSkill]:
        """加载所有内置技能"""
        skills = []
        builtin_classes = [
            "ai_researcher_assistant.skills.buildin.arxiv_fetcher.ArxivFetcherSkill",
            "ai_researcher_assistant.skills.buildin.paper_reader.PaperReaderSkill",
            "ai_researcher_assistant.skills.buildin.paper_writer.PaperWriterSkill",
        ]
        for dotted_path in builtin_classes:
            module_path, class_name = dotted_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                skill_class = getattr(module, class_name)
            except ImportError as exc:
                logger.warning("Skipping built-in skill %s: %s", dotted_path, exc)
                continue
            skills.append(self.load_from_class(skill_class))
        return skills
