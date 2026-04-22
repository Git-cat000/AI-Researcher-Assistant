"""
执行循环模块。
实现 ReAct（Reasoning + Acting）循环和 Plan-and-Execute 模式。
"""
import json
import re
from typing import Dict, Any, Optional, Tuple, List
import logging

from ai_researcher_assistant.llm import BaseLLM
from ai_researcher_assistant.skills.registry import SkillRegistry, get_skill_registry
from ai_researcher_assistant.orchestration.state import ExecutionState, ExecutionStep, ExecutionStatus
from ai_researcher_assistant.orchestration.middleware import MiddlewareManager
from ai_researcher_assistant.core.message import Conversation, MessageRole
from ai_researcher_assistant.core.exceptions import SkillError

logger = logging.getLogger(__name__)


class ReActLoop:
    """
    ReAct 执行循环。
    实现经典的 Reason-Act-Observe 循环，让 Agent 能够自主推理和调用技能。
    """

    def __init__(
        self,
        llm: BaseLLM,
        skill_registry: Optional[SkillRegistry] = None,
        middleware_manager: Optional[MiddlewareManager] = None,
        max_steps: int = 20,
    ):
        self.llm = llm
        self.skill_registry = skill_registry or get_skill_registry()
        self.middleware = middleware_manager or MiddlewareManager()
        self.max_steps = max_steps

        # 系统提示词模板
        self.system_prompt_template = """You are an AI research assistant with the ability to use specialized skills.

{skill_instructions}

You operate in a ReAct loop: Thought -> Action -> Observation.
For each step, output in the following format:

Thought: <your reasoning about what to do next>
Action: <a JSON object specifying the skill and parameters>
```json
{
  "skill": "skill_name",
  "parameters": {
    "param1": "value1"
  }
}
