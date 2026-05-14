"""
技能抽象基类。
定义了技能的生命周期和执行接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillParameter:
    """技能参数定义"""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None


@dataclass
class SkillManifest:
    """技能清单，描述技能的元信息"""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    parameters: list[SkillParameter] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    instructions: str = ""  # 给 LLM 看的详细使用说明
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """
    技能抽象基类。

    一个技能封装了完成特定任务的完整 SOP（标准操作流程）。
    它包含：
    - 元数据（名称、描述、参数）
    - 执行逻辑
    - 可选的资源文件
    """

    def __init__(self):
        self._manifest: SkillManifest | None = None

    @property
    def manifest(self) -> SkillManifest:
        """获取技能清单"""
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    @abstractmethod
    def _build_manifest(self) -> SkillManifest:
        """构建技能清单（子类必须实现）"""
        pass

    @abstractmethod
    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        执行技能。

        Args:
            parameters: 调用参数
            context: 执行上下文（可包含 LLM 实例、记忆系统等）

        Returns:
            执行结果字典，至少包含:
            - success: bool
            - result: Any (成功时的结果)
            - error: str (失败时的错误信息)
        """
        pass

    async def aexecute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """异步执行（默认调用同步方法）"""
        return self.execute(parameters, context)

    def get_instructions_for_llm(self) -> str:
        """
        生成给 LLM 看的指令文本。
        这部分内容会被注入到系统提示或上下文中，告诉 LLM 如何使用这个技能。
        """
        manifest = self.manifest
        lines = [
            f"## Skill: {manifest.name}",
            f"Description: {manifest.description}",
            f"Version: {manifest.version}",
            "",
            "### When to use this skill:",
            manifest.instructions or "Use when appropriate.",
            "",
            "### Parameters:",
        ]
        for param in manifest.parameters:
            required = "required" if param.required else "optional"
            default = f", default={param.default}" if param.default is not None else ""
            lines.append(f"- `{param.name}` ({param.type}, {required}{default}): {param.description}")

        lines.append("")
        lines.append("### How to invoke:")
        lines.append("Output a JSON action in the following format:")
        lines.append("```json")
        lines.append("{")
        lines.append(f'  "skill": "{manifest.name}",')
        lines.append('  "parameters": {')
        for param in manifest.parameters:
            lines.append(f'    "{param.name}": "<value>"')
        lines.append("  }")
        lines.append("}")
        lines.append("```")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Skill name='{self.manifest.name}' version='{self.manifest.version}'>"
