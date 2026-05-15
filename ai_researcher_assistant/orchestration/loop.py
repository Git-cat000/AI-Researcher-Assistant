"""
鎵ц寰幆妯″潡銆?
瀹炵幇澶氱 Agent 鎵ц妯″紡锛?
- ReActLoop: 缁忓吀鐨?Reason-Act-Observe 寰幆
- PlanAndExecuteLoop: 鍏堣鍒掑悗鎵ц鐨勬ā寮?
- LLMCompiler: 骞惰鎵ц鐙珛浠诲姟鐨勬ā寮?
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai_researcher_assistant.core.message import Conversation, MessageRole
from ai_researcher_assistant.harness.parsing import (
    extract_action,
    extract_final_answer,
    extract_json_block,
    extract_thought,
)
from ai_researcher_assistant.harness.schema import (
    Action,
    Observation,
    PermissionPolicy,
    TokenBudget,
    get_cost_tracker,
)
from ai_researcher_assistant.llm import BaseLLM, LLMResponse
from ai_researcher_assistant.orchestration.middleware import MiddlewareManager
from ai_researcher_assistant.orchestration.state import ExecutionState, ExecutionStatus, ExecutionStep
from ai_researcher_assistant.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class LoopType(str, Enum):
    """鎵ц寰幆绫诲瀷"""

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    LLM_COMPILER = "llm_compiler"


@dataclass
class LoopConfig:
    """鎵ц寰幆閰嶇疆"""

    loop_type: LoopType = LoopType.REACT
    max_steps: int = 20
    max_plan_steps: int = 10
    temperature: float = 0.0
    enable_parallel: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0


class BaseLoop(ABC):
    """鎵ц寰幆鎶借薄鍩虹被"""

    def __init__(
        self,
        llm: BaseLLM,
        skill_registry: SkillRegistry | None = None,
        middleware_manager: MiddlewareManager | None = None,
        config: LoopConfig | None = None,
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
    async def run(self, task: str, context: dict[str, Any]) -> ExecutionState:
        """杩愯鎵ц寰幆"""
        pass

    def build_system_prompt(self) -> str:
        """Build the system prompt."""
        skill_instructions = self.skill_registry.get_catalog_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions)

    def _init_conversation(self, task: str, context: dict[str, Any], task_prefix: str = "") -> Conversation:
        """Set up a conversation with system prompt and initial user task."""
        conversation = context.get("conversation") or Conversation()
        conversation.add_system(self.build_system_prompt())
        conversation.add(MessageRole.USER, f"{task_prefix}{task}" if task_prefix else task)
        return conversation

    async def _handle_loop_error(
        self, state: ExecutionState, error: Exception, context: dict[str, Any], loop_name: str
    ) -> None:
        """Wrap a loop-level exception with logging, middleware notification, and state update."""
        logger.exception("Error in %s loop", loop_name)
        await self.middleware.trigger_on_error(state, error, context)
        state.fail(str(error))

    async def _synthesize_final_answer(
        self, task: str, summary_data: dict, conversation: Conversation, context: dict[str, Any]
    ) -> str:
        """Prompt the LLM to synthesize a final answer from structured execution results."""
        prompt = f"""Based on the execution results for task: "{task}"

Summary:
{json.dumps(summary_data, indent=2)}

Please provide a comprehensive final answer to the user's original task.
Include citations and references where appropriate."""

        conversation.add(MessageRole.USER, prompt)
        response = await self._call_llm(conversation, context, temperature=0.0)
        return response.content

    async def _call_llm(
        self, conversation: Conversation, context: dict[str, Any], temperature: float = 0.0
    ) -> LLMResponse:
        budget = TokenBudget.from_context(context)
        messages = conversation.get_context_window(max_tokens=budget.max_context_tokens)
        response = await self.llm.agenerate(messages, temperature=temperature)
        get_cost_tracker(context).add_response(response)
        return response

    async def _execute_action(self, action: Action, context: dict[str, Any]) -> dict[str, Any]:
        allowed, reason = PermissionPolicy.from_context(context).check_action(action)
        if not allowed:
            return {"success": False, "result": None, "error": reason or "Action blocked by permission policy"}
        return await self._execute_skill(action.skill, action.parameters, context)

    async def _execute_skill(
        self, skill_name: str, parameters: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """鎵ц鍗曚釜鎶€鑳斤紙甯﹂噸璇曪級"""
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

    def _format_observation(self, skill_name: str, result: dict[str, Any]) -> str:
        """鏍煎紡鍖栨妧鑳芥墽琛岀粨鏋滀负瑙傚療鏂囨湰"""
        if result.get("success"):
            data = result.get("result", {})
            if skill_name == "arxiv_fetcher" and "papers" in data:
                papers = data["papers"]
                lines = [f"Found {len(papers)} papers:"]
                for p in papers[:5]:
                    lines.append(f"- {p.get('title')} (arXiv:{p.get('arxiv_id')})")
                if len(papers) > 5:
                    lines.append(f"... and {len(papers) - 5} more.")
                return "\n".join(lines)
            else:
                return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"


class ReActLoop(BaseLoop):
    """
    ReAct 鎵ц寰幆銆?
    瀹炵幇缁忓吀鐨?Reason-Act-Observe 寰幆銆?
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
        skill_instructions = self.skill_registry.get_catalog_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions) + "\n" + self.react_system_prompt

    async def run(self, task: str, context: dict[str, Any]) -> ExecutionState:
        """杩愯 ReAct 寰幆"""
        state = ExecutionState(max_steps=self.config.max_steps)
        state.start_task(task)

        conversation = self._init_conversation(task, context)

        try:
            while state.current_step < state.max_steps:
                # 1. Think 闃舵
                await self.middleware.trigger_before_think(state, context)
                thought, action = await self._think(conversation, state, context)
                await self.middleware.trigger_after_think(state, thought, context)

                step = ExecutionStep(thought=thought)

                final_answer = extract_final_answer(thought)
                if final_answer:
                    state.complete(final_answer)
                    break

                if action is None:
                    conversation.add(MessageRole.ASSISTANT, thought)
                    conversation.add(MessageRole.USER, "Please provide an Action or Final Answer.")
                    continue

                step.action = action
                state.status = ExecutionStatus.ACTING

                # 2. Act 闃舵
                action_model = Action.from_mapping(action, thought=thought)
                await self.middleware.trigger_before_act(state, action_model.to_dict(), context)
                result = await self._execute_action(action_model, context)
                await self.middleware.trigger_after_act(state, result, context)

                observation = Observation.from_skill_result(
                    action_model.skill,
                    result,
                    self._format_observation(action_model.skill, result),
                )
                step.observation = observation.to_prompt(TokenBudget.from_context(context).max_observation_tokens)
                state.add_step(step)
                state.status = ExecutionStatus.OBSERVING

                conversation.add(MessageRole.ASSISTANT, f"Thought: {thought}\nAction: {json.dumps(action)}")
                conversation.add(MessageRole.OBSERVATION, f"Observation: {step.observation}")

                if not result.get("success") and self._should_abort(state):
                    state.fail("Too many skill failures")
                    break

            if state.status not in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                final = await self._force_final_answer(conversation, context)
                state.complete(final)

        except Exception as e:
            await self._handle_loop_error(state, e, context, "ReAct")

        return state

    async def _think(
        self, conversation: Conversation, state: ExecutionState, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        """Call the LLM for the next ReAct thought/action."""
        response = await self._call_llm(conversation, context, temperature=self.config.temperature)
        content = response.content

        thought = extract_thought(content)
        final_answer = extract_final_answer(content)
        if final_answer:
            return f"Thought: {thought}\nFinal Answer: {final_answer}", None
        action = extract_action(content)

        return thought, action

    async def _force_final_answer(self, conversation: Conversation, context: dict[str, Any]) -> str:
        """Ask the LLM to produce a final answer at the step limit."""
        conversation.add(MessageRole.USER, "You have reached the step limit. Please provide your final answer now.")
        response = await self._call_llm(conversation, context, temperature=0.0)
        return response.content

    def _should_abort(self, state: ExecutionState) -> bool:
        """鍒ゆ柇鏄惁搴旇涓锛堜緥濡傝繛缁け璐ワ級"""
        if len(state.steps) < 3:
            return False
        recent = state.steps[-3:]
        failures = sum(1 for s in recent if s.observation and "Error:" in s.observation)
        return failures >= 3


class PlanAndExecuteLoop(BaseLoop):
    """
    鍏堣鍒掑悗鎵ц妯″紡銆?
    棣栧厛璁?LLM 鍒跺畾瀹屾暣璁″垝锛岀劧鍚庢寜椤哄簭鎵ц鍚勬楠ゃ€?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_system_prompt = """You are a planning agent.
Given a task, create a step-by-step plan to accomplish it.

Output your plan as a JSON list of steps. Each step should have:
- "step": step number
- "description": what to do in this step
- "skill": the skill to use (or null if no skill needed)
- "parameters": parameters for the skill (or empty object)

Example:
[//]: # (JSON array example)
[
  {
    "step": 1,
    "description": "Search for recent papers on quantum gravity",
    "skill": "arxiv_fetcher",
    "parameters": {"query": "quantum gravity", "max_results": 5}
  },
  {
    "step": 2,
    "description": "Read and summarize the most relevant paper",
    "skill": "paper_reader",
    "parameters": {"url": "from previous step"}
  },
  {"step": 3, "description": "Provide a summary to the user", "skill": null, "parameters": {}}
]
"""

    def build_system_prompt(self) -> str:
        skill_instructions = self.skill_registry.get_catalog_for_all()
        return self.base_system_prompt.format(skill_instructions=skill_instructions) + "\n" + self.plan_system_prompt

    async def run(self, task: str, context: dict[str, Any]) -> ExecutionState:
        state = ExecutionState(max_steps=self.config.max_plan_steps)
        state.start_task(task)

        conversation = self._init_conversation(task, context, task_prefix="Task: ")

        try:
            conversation.add(MessageRole.USER, "Create a plan:")
            plan = await self._generate_plan(conversation, context)
            if not plan:
                state.fail("Failed to generate a valid plan")
                return state

            state.metadata["plan"] = plan
            logger.info("Generated plan with %d steps", len(plan))

            results: dict[str, Any] = {}
            for step_info in plan:
                step_num = step_info.get("step", state.current_step + 1)
                description = step_info.get("description", "")
                skill_name = step_info.get("skill")
                parameters = step_info.get("parameters", {})

                step = ExecutionStep(thought=f"Executing plan step {step_num}: {description}")
                state.status = ExecutionStatus.ACTING

                await self.middleware.trigger_before_think(state, context)

                if skill_name:
                    await self.middleware.trigger_before_act(
                        state, {"skill": skill_name, "parameters": parameters}, context
                    )
                    result = await self._execute_action(Action(skill=skill_name, parameters=parameters), context)
                    await self.middleware.trigger_after_act(state, result, context)

                    observation = self._format_observation(skill_name, result)
                    step.action = {"skill": skill_name, "parameters": parameters}
                    step.observation = observation
                    results[f"step_{step_num}"] = result.get("result") if result.get("success") else None

                    if not result.get("success"):
                        logger.warning("Step %d failed: %s", step_num, result.get("error"))
                else:
                    step.observation = f"Completed: {description}"
                    results[f"step_{step_num}"] = description

                state.add_step(step)
                await self.middleware.trigger_after_think(state, description, context)

                context["plan_results"] = results

            final_answer = await self._synthesize_final_answer(
                task, {"plan": plan, "results": results}, conversation, context
            )
            state.complete(final_answer)

        except Exception as e:
            await self._handle_loop_error(state, e, context, "Plan-and-Execute")

        return state

    async def _generate_plan(self, conversation: Conversation, context: dict[str, Any]) -> list[dict[str, Any]] | None:
        """鐢熸垚鎵ц璁″垝"""
        response = await self._call_llm(conversation, context, temperature=0.0)

        parsed = extract_json_block(response.content)
        return parsed if isinstance(parsed, list) else None


class LLMCompilerLoop(BaseLoop):
    """
    LLM Compiler 妯″紡銆?
    鍊熼壌 LLMCompiler 璁烘枃鎬濇兂锛屽厛璇嗗埆鍙苟琛屾墽琛岀殑鐙珛浠诲姟锛岀劧鍚庡苟琛岃皟鐢ㄦ妧鑳姐€?
    """

    async def run(self, task: str, context: dict[str, Any]) -> ExecutionState:
        state = ExecutionState(max_steps=self.config.max_steps)
        state.start_task(task)

        try:
            dag = await self._parse_dag(task, context)
            state.metadata["dag"] = dag

            results = await self._execute_dag(dag, state, context)

            conversation = self._init_conversation(task, context)
            final = await self._synthesize_final_answer(task, {"dag": dag, "results": results}, conversation, context)
            state.complete(final)

        except Exception as e:
            await self._handle_loop_error(state, e, context, "LLMCompiler")

        return state

    async def _parse_dag(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        """Parse a task into an executable dependency graph."""
        prompt = f"""Analyze the following task.
Break it into independent subtasks that can be executed in parallel where possible.

Task: {task}

Output a JSON object with:
- "subtasks": list of subtasks.
  Each subtask has "id", "description", "skill", "parameters", and "depends_on".
- "parallel_groups": list of groups of subtask IDs that can run in parallel

Example:
{{
  "subtasks": [
    {{
      "id": "1",
      "description": "Search arXiv for paper A",
      "skill": "arxiv_fetcher",
      "parameters": {{"query": "paper A"}},
      "depends_on": []
    }},
    {{
      "id": "2",
      "description": "Search arXiv for paper B",
      "skill": "arxiv_fetcher",
      "parameters": {{"query": "paper B"}},
      "depends_on": []
    }},
    {{"id": "3", "description": "Compare results", "skill": null, "parameters": {{}}, "depends_on": ["1", "2"]}}
  ],
  "parallel_groups": [["1", "2"]]
}}
"""
        conversation = Conversation()
        conversation.add(MessageRole.USER, prompt)
        response = await self._call_llm(conversation, context, temperature=0.0)
        parsed = extract_json_block(response.content)
        return parsed if isinstance(parsed, dict) else {"subtasks": [], "parallel_groups": []}

    async def _execute_dag(self, dag: dict[str, Any], state: ExecutionState, context: dict[str, Any]) -> dict[str, Any]:
        """鎵ц DAG"""
        subtasks = {t["id"]: t for t in dag.get("subtasks", [])}
        completed: set[str] = set()
        results: dict[str, Any] = {}
        step_counter = 0

        while len(completed) < len(subtasks):
            # 鎵惧嚭鎵€鏈変緷璧栧凡婊¤冻涓旀湭鎵ц鐨勫瓙浠诲姟
            ready: list[tuple[str, dict[str, Any]]] = []
            for tid, task in subtasks.items():
                if tid in completed:
                    continue
                deps = task.get("depends_on", [])
                if all(d in completed for d in deps):
                    ready.append((tid, task))

            if not ready:
                # 姝婚攣鎴栨棤浠诲姟
                break

            # 骞惰鎵ц
            async def exec_one(tid: str, task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                if task.get("skill"):
                    result = await self._execute_action(
                        Action(skill=task["skill"], parameters=task.get("parameters", {})), context
                    )
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
                    action={"skill": subtasks[tid].get("skill"), "parameters": subtasks[tid].get("parameters")}
                    if subtasks[tid].get("skill")
                    else None,
                    observation=self._format_observation(subtasks[tid].get("skill", ""), res)
                    if subtasks[tid].get("skill")
                    else str(res.get("result")),
                )
                state.add_step(step)

        return results


def create_loop(
    loop_type: str | LoopType,
    llm: BaseLLM,
    skill_registry: SkillRegistry | None = None,
    middleware_manager: MiddlewareManager | None = None,
    config: LoopConfig | None = None,
) -> BaseLoop:
    """Create an execution loop instance by type."""
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
