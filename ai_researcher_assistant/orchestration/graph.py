"""
基于图的编排模块。
提供类似 LangGraph 的状态图编排能力。
"""
from typing import (
    Dict, Any, Optional, List, Callable, Awaitable, Union,
    Set, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging

from ai_researcher_assistant.llm import BaseLLM
from ai_researcher_assistant.skills.registry import SkillRegistry
from ai_researcher_assistant.orchestration.state import ExecutionState, ExecutionStep
from ai_researcher_assistant.core.message import Conversation, MessageRole

logger = logging.getLogger(__name__)


class GraphState(dict):
    """图状态字典，在节点间传递"""
    pass


NodeFunc = Callable[[GraphState], Awaitable[GraphState]]
ConditionalFunc = Callable[[GraphState], str]


class NodeType(str, Enum):
    """节点类型"""
    LLM = "llm"
    SKILL = "skill"
    FUNCTION = "function"
    CONDITION = "condition"
    START = "start"
    END = "end"


@dataclass
class Node:
    """图节点"""
    name: str
    func: NodeFunc
    node_type: NodeType = NodeType.FUNCTION
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """图边"""
    source: str
    target: str
    condition: Optional[ConditionalFunc] = None
    label: str = ""


class StateGraph:
    """状态图构建器与执行器"""

    def __init__(self, name: str = "AgentGraph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.conditional_edges: Dict[str, List[Edge]] = {}
        self.entry_point: Optional[str] = None
        self.finish_points: Set[str] = set()

    def add_node(
        self,
        name: str,
        func: NodeFunc,
        node_type: NodeType = NodeType.FUNCTION,
        description: str = "",
        **metadata
    ) -> "StateGraph":
        self.nodes[name] = Node(
            name=name,
            func=func,
            node_type=node_type,
            description=description,
            metadata=metadata,
        )
        return self

    def add_llm_node(
        self,
        name: str,
        llm: BaseLLM,
        prompt_template: str,
        **metadata
    ) -> "StateGraph":
        async def llm_node(state: GraphState) -> GraphState:
            prompt = prompt_template.format(**state)
            messages = [{"role": "user", "content": prompt}]
            response = await llm.agenerate(messages)
            state["llm_response"] = response.content
            return state

        return self.add_node(name, llm_node, NodeType.LLM, **metadata)

    def add_skill_node(
        self,
        name: str,
        skill_name: str,
        skill_registry: Optional[SkillRegistry] = None,
        param_mapping: Optional[Callable[[GraphState], Dict[str, Any]]] = None,
        **metadata
    ) -> "StateGraph":
        registry = skill_registry or SkillRegistry()

        async def skill_node(state: GraphState) -> GraphState:
            if param_mapping:
                params = param_mapping(state)
            else:
                params = state.get("skill_parameters", {})
            result = await registry.aexecute(skill_name, params, {"state": state})
            state["skill_result"] = result
            if not result.get("success"):
                state["error"] = result.get("error")
            return state

        return self.add_node(name, skill_node, NodeType.SKILL, **metadata)

    def set_entry_point(self, node_name: str) -> "StateGraph":
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.entry_point = node_name
        return self

    def set_finish_point(self, node_name: str) -> "StateGraph":
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.finish_points.add(node_name)
        return self

    def add_edge(self, source: str, target: str, label: str = "") -> "StateGraph":
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        self.edges.append(Edge(source=source, target=target, label=label))
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: ConditionalFunc,
        targets: Dict[str, str],
        label: str = ""
    ) -> "StateGraph":
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        for t in targets.values():
            if t not in self.nodes:
                raise ValueError(f"Target node '{t}' not found")
        self.conditional_edges[source] = [
            Edge(source=source, target=targets[k], condition=condition, label=f"{label}:{k}")
            for k in targets
        ]
        return self

    def compile(self) -> "CompiledGraph":
        if self.entry_point is None:
            raise ValueError("Entry point not set")
        return CompiledGraph(
            name=self.name,
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            finish_points=self.finish_points,
        )


class CompiledGraph:
    """编译后的可执行图"""

    def __init__(
        self,
        name: str,
        nodes: Dict[str, Node],
        edges: List[Edge],
        conditional_edges: Dict[str, List[Edge]],
        entry_point: str,
        finish_points: Set[str],
    ):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.finish_points = finish_points

        self._build_adjacency()

    def _build_adjacency(self):
        self.adjacency: Dict[str, List[Edge]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            self.adjacency[edge.source].append(edge)
        for source, edges in self.conditional_edges.items():
            self.adjacency[source].extend(edges)

    async def run(self, initial_state: GraphState, max_steps: int = 50) -> GraphState:
        current_node = self.entry_point
        state = initial_state.copy()
        step_count = 0

        if "execution_state" in state:
            exec_state = state["execution_state"]
        else:
            exec_state = ExecutionState()
            exec_state.start_task(state.get("task", "Graph execution"))
            state["execution_state"] = exec_state

        logger.info(f"Starting graph '{self.name}' execution from node '{current_node}'")

        while current_node and step_count < max_steps:
            if current_node in self.finish_points:
                logger.info(f"Reached finish node '{current_node}'")
                break

            if current_node not in self.nodes:
                logger.error(f"Node '{current_node}' not found")
                state["error"] = f"Node '{current_node}' not found"
                break

            node = self.nodes[current_node]
            logger.debug(f"Executing node '{current_node}' (type: {node.node_type})")

            try:
                step = ExecutionStep(thought=f"Executing graph node: {current_node}")
                exec_state.add_step(step)

                state = await node.func(state)

                step.observation = f"Node '{current_node}' completed"
                if "error" in state and state["error"]:
                    step.error = state["error"]

                next_node = await self._get_next_node(current_node, state)
                current_node = next_node

            except Exception as e:
                logger.exception(f"Error executing node '{current_node}'")
                state["error"] = str(e)
                step.error = str(e)
                break

            step_count += 1

        if step_count >= max_steps:
            state["error"] = "Max steps exceeded"
            exec_state.fail("Max steps exceeded")
        elif "error" not in state or not state["error"]:
            exec_state.complete("Graph execution completed")

        return state

    async def _get_next_node(self, current: str, state: GraphState) -> Optional[str]:
        edges = self.adjacency.get(current, [])

        for edge in edges:
            if edge.condition:
                target_key = edge.condition(state)
                for e in self.conditional_edges.get(current, []):
                    if e.label.endswith(f":{target_key}") or e.label == target_key:
                        return e.target
            else:
                return edge.target

        return None


class AgentGraphBuilder:
    """针对 ResearcherAgent 的图构建器"""

    def __init__(self, llm: BaseLLM, skill_registry: Optional[SkillRegistry] = None):
        self.llm = llm
        self.skill_registry = skill_registry or SkillRegistry()

    def build_react_graph(self) -> StateGraph:
        graph = StateGraph("ReActGraph")

        async def think_node(state: GraphState) -> GraphState:
            conv = state.get("conversation")
            if not conv:
                conv = Conversation()
                conv.add_system("You are a ReAct agent...")
                state["conversation"] = conv

            messages = conv.get_context_window()
            response = await self.llm.agenerate(messages, temperature=0.0)
            state["llm_response"] = response.content

            import re
            text = response.content
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
            state["thought"] = thought_match.group(1).strip() if thought_match else text

            action_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if action_match:
                import json
                try:
                    state["action"] = json.loads(action_match.group(1))
                except (json.JSONDecodeError, TypeError):
                    state["action"] = None
            else:
                state["action"] = None

            if "Final Answer:" in text:
                state["is_final"] = True
                final_match = re.search(r"Final Answer:\s*(.+)$", text, re.DOTALL)
                state["final_answer"] = final_match.group(1).strip() if final_match else text
            else:
                state["is_final"] = False

            return state

        def route_after_think(state: GraphState) -> str:
            if state.get("is_final"):
                return "final"
            elif state.get("action"):
                return "act"
            else:
                return "think"

        async def act_node(state: GraphState) -> GraphState:
            action = state.get("action", {})
            skill_name = action.get("skill")
            parameters = action.get("parameters", {})

            if skill_name:
                result = await self.skill_registry.aexecute(skill_name, parameters, {"state": state})
                state["skill_result"] = result
                if result.get("success"):
                    state["observation"] = f"Skill '{skill_name}' executed successfully."
                else:
                    state["observation"] = f"Error: {result.get('error')}"
            else:
                state["observation"] = "No skill specified."

            conv = state["conversation"]
            conv.add(MessageRole.ASSISTANT, f"Thought: {state.get('thought')}\nAction: {json.dumps(action)}")
            conv.add(MessageRole.OBSERVATION, f"Observation: {state.get('observation')}")
            return state

        async def final_node(state: GraphState) -> GraphState:
            return state

        graph.add_node("think", think_node, NodeType.LLM)
        graph.add_node("act", act_node, NodeType.SKILL)
        graph.add_node("final", final_node, NodeType.END)

        graph.set_entry_point("think")
        graph.set_finish_point("final")

        graph.add_conditional_edges("think", route_after_think, {"act": "act", "final": "final"})
        graph.add_edge("act", "think")

        return graph
