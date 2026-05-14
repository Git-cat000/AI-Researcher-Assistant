"""
执行循环模块。
实现多种 Agent 执行模式：
- ReActLoop: 经典的 Reason-Act-Observe 循环
- PlanAndExecuteLoop: 先规划后执行的模式
- LLMCompiler: 并行执行独立任务的模式
"""
import json
import re
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

from ai_researcher_assistant.llm import BaseLLM
from ai_researcher_assistant.skills.registry import SkillRegistry
from ai_researcher_assistant.orchestration.state import ExecutionState, ExecutionStep, ExecutionStatus
from ai_researcher_assistant.orchestration.middleware import MiddlewareManager
from ai_researcher_assistant.core.message import Conversation, MessageRole
from ai_researcher_assistant.core.exceptions import SkillError, LLMError
from ai_researcher_assistant.harness.parsing import (
    extract_action,
    extract_final_answer,
    extract_json_block,
    extract_thought,
)

logger = logging.getLogger(__name__)


class LoopType(str, Enum):
    """执行循环类型"""
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    LLM_COMPILER = "llm_compiler"


@dataclass
class LoopConfig:
    """执行循环配置"""
    loop_type: LoopType = LoopType.REACT
    max_steps: int = 20
    max_plan_steps: int = 10
    temperature: float = 0.0
    enable_parallel: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0


class BaseLoop(ABC):
    """执行循环抽象基类"""

    def __init__(
        self,
        llm: BaseLLM,
        skill_registry: Optional[SkillRegistry] = None,
        middleware_manager: Optional[MiddlewareManager] = None,
        config: Optional[LoopConfig] = None,
    ):
        self.llm = llm
        self.skill_registry = skill_registry or SkillRegistry()
        self.middleware = middleware_manager or MiddlewareManager()
        self.config = config or LoopConfig()

        self.base_system_prompt = """You are an AI research assistant with the ability to use specialized skills.

Available skills:
{skill_instructions}

You must reason step by step and use skills when necessary.
"""

    @abstractmethod
    async def run(self, task: str, context: Dict[str, Any]) -> ExecutionState:
        """运行执行循环"""
        pass

    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        skill_instructions = self.skill_registry.get_instructions_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions)

    async def _execute_skill(
        self, skill_name: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行单个技能（带重试）"""
        for attempt in range(self.config.max_retries):
            try:
                result = await self.skill_registry.aexecute(skill_name, parameters, context)
                result["_attempts"] = attempt + 1
                return result
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return {"success": False, "error": str(e), "_attempts": attempt + 1}
                await asyncio.sleep(self.config.retry_delay)
        return {"success": False, "error": "Max retries exceeded"}

    def _format_observation(self, skill_name: str, result: Dict[str, Any]) -> str:
        """格式化技能执行结果为观察文本"""
        if result.get("success"):
            data = result.get("result", {})
            if skill_name == "arxiv_fetcher" and "papers" in data:
                papers = data["papers"]
                lines = [f"Found {len(papers)} papers:"]
                for p in papers[:5]:
                    lines.append(f"- {p.get('title')} (arXiv:{p.get('arxiv_id')})")
                if len(papers) > 5:
                    lines.append(f"... and {len(papers)-5} more.")
                return "\n".join(lines)
            else:
                return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"


class ReActLoop(BaseLoop):
    """
    ReAct 执行循环。
    实现经典的 Reason-Act-Observe 循环。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.react_system_prompt = """You operate in a ReAct loop: Thought -> Action -> Observation.

For each step, output in the following format:

Thought: <your reasoning about what to do next>
Action: <a JSON object specifying the skill and parameters>
[//]: # (JSON format example)
{{
  "skill": "skill_name",
  "parameters": {{
    "param1": "value1"
  }}
}}

After receiving the observation, continue the loop.
When you have the final answer, output:

Thought: I now have enough information to answer the question.
Final Answer: <your comprehensive answer to the user>

Important rules:
- Always base your answer on information obtained from skills.
- For academic questions, cite sources (paper titles, arXiv IDs).
- If a skill fails, try an alternative approach or report the error.
- Do not invent information not obtained from observations.
"""

    def build_system_prompt(self) -> str:
        skill_instructions = self.skill_registry.get_instructions_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions) + "\n" + self.react_system_prompt

    async def run(self, task: str, context: Dict[str, Any]) -> ExecutionState:
        """运行 ReAct 循环"""
        state = ExecutionState(max_steps=self.config.max_steps)
        state.start_task(task)

        conversation = context.get("conversation") or Conversation()
        conversation.add_system(self.build_system_prompt())
        conversation.add(MessageRole.USER, task)

        try:
            while state.current_step < state.max_steps:
                # 1. Think 阶段
                await self.middleware.trigger_before_think(state, context)
                thought, action = await self._think(conversation, state, context)
                await self.middleware.trigger_after_think(state, thought, context)

                step = ExecutionStep(thought=thought)

                # 检查是否已完成
                final_answer = self._extract_final_answer(thought)
                if final_answer:
                    state.complete(final_answer)
                    break

                if action is None:
                    # 没有行动也没有最终答案，要求继续
                    conversation.add(MessageRole.ASSISTANT, thought)
                    conversation.add(MessageRole.USER, "Please provide an Action or Final Answer.")
                    continue

                step.action = action
                state.status = ExecutionStatus.ACTING

                # 2. Act 阶段
                await self.middleware.trigger_before_act(state, action, context)
                result = await self._execute_skill(action["skill"], action.get("parameters", {}), context)
                await self.middleware.trigger_after_act(state, result, context)

                observation = self._format_observation(action["skill"], result)
                step.observation = observation
                state.add_step(step)
                state.status = ExecutionStatus.OBSERVING

                # 添加到对话历史
                conversation.add(MessageRole.ASSISTANT, f"Thought: {thought}\nAction: {json.dumps(action)}")
                conversation.add(MessageRole.OBSERVATION, f"Observation: {observation}")

                # 检查是否失败过多
                if not result.get("success") and self._should_abort(state):
                    state.fail("Too many skill failures")
                    break

            # 超过最大步数，强制生成答案
            if state.status not in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                final = await self._force_final_answer(conversation)
                state.complete(final)

        except Exception as e:
            logger.exception("Error in ReAct loop")
            await self.middleware.trigger_on_error(state, e, context)
            state.fail(str(e))

        return state

    async def _think(
        self, conversation: Conversation, state: ExecutionState, context: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """调用 LLM 进行思考"""
        messages = conversation.get_context_window()
        response = await self.llm.agenerate(messages, temperature=self.config.temperature)
        content = response.content

        thought = self._extract_thought(content)
        final_answer = self._extract_final_answer(content)
        if final_answer:
            return f"Thought: {thought}\nFinal Answer: {final_answer}", None
        action = self._extract_action(content)

        return thought, action

    def _extract_thought(self, text: str) -> str:
        """提取 Thought 部分"""
        return extract_thought(text)

    def _extract_action(self, text: str) -> Optional[Dict[str, Any]]:
        """提取 JSON 格式的 Action"""
        return extract_action(text)

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """提取最终答案"""
        return extract_final_answer(text)

    async def _force_final_answer(self, conversation: Conversation) -> str:
        """强制 LLM 输出最终答案"""
        conversation.add(MessageRole.USER, "You have reached the step limit. Please provide your final answer now.")
        messages = conversation.get_context_window()
        response = await self.llm.agenerate(messages, temperature=0.0)
        return response.content

    def _should_abort(self, state: ExecutionState) -> bool:
        """判断是否应该中止（例如连续失败）"""
        if len(state.steps) < 3:
            return False
        recent = state.steps[-3:]
        failures = sum(1 for s in recent if s.observation and "Error:" in s.observation)
        return failures >= 3


class PlanAndExecuteLoop(BaseLoop):
    """
    先规划后执行模式。
    首先让 LLM 制定完整计划，然后按顺序执行各步骤。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_system_prompt = """You are a planning agent. Given a task, create a step-by-step plan to accomplish it.

Output your plan as a JSON list of steps. Each step should have:
- "step": step number
- "description": what to do in this step
- "skill": the skill to use (or null if no skill needed)
- "parameters": parameters for the skill (or empty object)

Example:
[//]: # (JSON array example)
[
  {"step": 1, "description": "Search for recent papers on quantum gravity", "skill": "arxiv_fetcher", "parameters": {"query": "quantum gravity", "max_results": 5}},
  {"step": 2, "description": "Read and summarize the most relevant paper", "skill": "paper_reader", "parameters": {"url": "from previous step"}},
  {"step": 3, "description": "Provide a summary to the user", "skill": null, "parameters": {}}
]
"""

    def build_system_prompt(self) -> str:
        skill_instructions = self.skill_registry.get_instructions_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions) + "\n" + self.plan_system_prompt

    async def run(self, task: str, context: Dict[str, Any]) -> ExecutionState:
        state = ExecutionState(max_steps=self.config.max_plan_steps)
        state.start_task(task)

        conversation = context.get("conversation") or Conversation()
        conversation.add_system(self.build_system_prompt())
        conversation.add(MessageRole.USER, f"Task: {task}\n\nCreate a plan:")

        try:
            # 1. 生成计划
            plan = await self._generate_plan(conversation)
            if not plan:
                state.fail("Failed to generate a valid plan")
                return state

            state.metadata["plan"] = plan
            logger.info(f"Generated plan with {len(plan)} steps")

            # 2. 执行计划
            results = {}
            for step_info in plan:
                step_num = step_info.get("step", state.current_step + 1)
                description = step_info.get("description", "")
                skill_name = step_info.get("skill")
                parameters = step_info.get("parameters", {})

                step = ExecutionStep(thought=f"Executing plan step {step_num}: {description}")
                state.status = ExecutionStatus.ACTING

                await self.middleware.trigger_before_think(state, context)

                if skill_name:
                    await self.middleware.trigger_before_act(state, {"skill": skill_name, "parameters": parameters}, context)
                    result = await self._execute_skill(skill_name, parameters, context)
                    await self.middleware.trigger_after_act(state, result, context)

                    observation = self._format_observation(skill_name, result)
                    step.action = {"skill": skill_name, "parameters": parameters}
                    step.observation = observation
                    results[f"step_{step_num}"] = result.get("result") if result.get("success") else None

                    if not result.get("success"):
                        logger.warning(f"Step {step_num} failed: {result.get('error')}")
                else:
                    # 无技能步骤，直接记录
                    step.observation = f"Completed: {description}"
                    results[f"step_{step_num}"] = description

                state.add_step(step)
                await self.middleware.trigger_after_think(state, description, context)

                # 更新上下文供后续步骤使用
                context["plan_results"] = results

            # 3. 生成最终答案
            final_answer = await self._synthesize_answer(task, plan, results, conversation)
            state.complete(final_answer)

        except Exception as e:
            logger.exception("Error in Plan-and-Execute loop")
            await self.middleware.trigger_on_error(state, e, context)
            state.fail(str(e))

        return state

    async def _generate_plan(self, conversation: Conversation) -> Optional[List[Dict[str, Any]]]:
        """生成执行计划"""
        messages = conversation.get_context_window()
        response = await self.llm.agenerate(messages, temperature=0.0)

        parsed = extract_json_block(response.content)
        return parsed if isinstance(parsed, list) else None

    async def _synthesize_answer(
        self, task: str, plan: List[Dict], results: Dict, conversation: Conversation
    ) -> str:
        """综合计划执行结果生成最终答案"""
        prompt = f"""Based on the execution of the plan for task: "{task}"

Plan steps and results:
{json.dumps({"plan": plan, "results": results}, indent=2)}

Please provide a comprehensive final answer to the user's original task.
Include citations and references where appropriate."""
        
        conversation.add(MessageRole.USER, prompt)
        messages = conversation.get_context_window()
        response = await self.llm.agenerate(messages, temperature=0.0)
        return response.content


class LLMCompilerLoop(BaseLoop):
    """
    LLM Compiler 模式。
    借鉴 LLMCompiler 论文思想，先识别可并行执行的独立任务，然后并行调用技能。
    """

    async def run(self, task: str, context: Dict[str, Any]) -> ExecutionState:
        state = ExecutionState(max_steps=self.config.max_steps)
        state.start_task(task)

        try:
            # 1. 解析任务，生成 DAG
            dag = await self._parse_dag(task, context)
            state.metadata["dag"] = dag

            # 2. 并行执行无依赖的任务
            results = await self._execute_dag(dag, state, context)

            # 3. 生成最终答案
            final = await self._synthesize_dag_results(task, dag, results, context)
            state.complete(final)

        except Exception as e:
            logger.exception("Error in LLMCompiler loop")
            state.fail(str(e))

        return state

    async def _parse_dag(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """解析任务生成依赖图"""
        prompt = f"""Analyze the following task and break it into independent subtasks that can be executed in parallel where possible.

Task: {task}

Output a JSON object with:
- "subtasks": list of subtasks, each with "id", "description", "skill", "parameters", "depends_on" (list of subtask IDs that must complete first)
- "parallel_groups": list of groups of subtask IDs that can run in parallel

Example:
{{
  "subtasks": [
    {{"id": "1", "description": "Search arXiv for paper A", "skill": "arxiv_fetcher", "parameters": {{"query": "paper A"}}, "depends_on": []}},
    {{"id": "2", "description": "Search arXiv for paper B", "skill": "arxiv_fetcher", "parameters": {{"query": "paper B"}}, "depends_on": []}},
    {{"id": "3", "description": "Compare results", "skill": null, "parameters": {{}}, "depends_on": ["1", "2"]}}
  ],
  "parallel_groups": [["1", "2"]]
}}
"""
        response = await self.llm.agenerate([{"role": "user", "content": prompt}], temperature=0.0)
        parsed = extract_json_block(response.content)
        return parsed if isinstance(parsed, dict) else {"subtasks": [], "parallel_groups": []}

    async def _execute_dag(
        self, dag: Dict[str, Any], state: ExecutionState, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行 DAG"""
        subtasks = {t["id"]: t for t in dag.get("subtasks", [])}
        completed = set()
        results = {}
        step_counter = 0

        while len(completed) < len(subtasks):
            # 找出所有依赖已满足且未执行的子任务
            ready = []
            for tid, task in subtasks.items():
                if tid in completed:
                    continue
                deps = task.get("depends_on", [])
                if all(d in completed for d in deps):
                    ready.append((tid, task))

            if not ready:
                # 死锁或无任务
                break

            # 并行执行
            async def exec_one(tid, task):
                if task.get("skill"):
                    result = await self._execute_skill(task["skill"], task.get("parameters", {}), context)
                    return tid, result
                else:
                    return tid, {"success": True, "result": task.get("description")}

            tasks = [exec_one(tid, task) for tid, task in ready]
            batch_results = await asyncio.gather(*tasks)

            for tid, res in batch_results:
                results[tid] = res
                completed.add(tid)
                step_counter += 1
                step = ExecutionStep(
                    thought=f"Executed subtask {tid}: {subtasks[tid]['description']}",
                    action={"skill": subtasks[tid].get("skill"), "parameters": subtasks[tid].get("parameters")} if subtasks[tid].get("skill") else None,
                    observation=self._format_observation(subtasks[tid].get("skill", ""), res) if subtasks[tid].get("skill") else str(res.get("result")),
                )
                state.add_step(step)

        return results

    async def _synthesize_dag_results(
        self, task: str, dag: Dict, results: Dict, context: Dict
    ) -> str:
        """综合 DAG 结果"""
        prompt = f"""Task: {task}

Execution results from subtasks:
{json.dumps(results, indent=2)}

Please provide a comprehensive final answer."""
        response = await self.llm.agenerate([{"role": "user", "content": prompt}], temperature=0.0)
        return response.content


def create_loop(
    loop_type: Union[str, LoopType],
    llm: BaseLLM,
    skill_registry: Optional[SkillRegistry] = None,
    middleware_manager: Optional[MiddlewareManager] = None,
    config: Optional[LoopConfig] = None,
) -> BaseLoop:
    """工厂函数：根据类型创建执行循环实例"""
    if isinstance(loop_type, str):
        loop_type = LoopType(loop_type)

    if loop_type == LoopType.REACT:
        return ReActLoop(llm, skill_registry, middleware_manager, config)
    elif loop_type == LoopType.PLAN_EXECUTE:
        return PlanAndExecuteLoop(llm, skill_registry, middleware_manager, config)
    elif loop_type == LoopType.LLM_COMPILER:
        return LLMCompilerLoop(llm, skill_registry, middleware_manager, config)
    else:
        raise ValueError(f"Unknown loop type: {loop_type}")
