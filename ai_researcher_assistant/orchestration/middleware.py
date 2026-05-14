"""
中间件系统。
允许在 Agent 执行流程的各个阶段插入自定义逻辑（日志、监控、验证等）。
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ai_researcher_assistant.orchestration.state import ExecutionState

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """中间件抽象基类"""

    @abstractmethod
    async def before_think(self, state: ExecutionState, context: dict[str, Any]) -> None:
        """在 LLM 思考之前调用"""
        pass

    @abstractmethod
    async def after_think(self, state: ExecutionState, thought: str, context: dict[str, Any]) -> None:
        """在 LLM 思考之后调用"""
        pass

    @abstractmethod
    async def before_act(self, state: ExecutionState, action: dict[str, Any], context: dict[str, Any]) -> None:
        """在执行技能之前调用"""
        pass

    @abstractmethod
    async def after_act(self, state: ExecutionState, result: dict[str, Any], context: dict[str, Any]) -> None:
        """在执行技能之后调用"""
        pass

    @abstractmethod
    async def on_error(self, state: ExecutionState, error: Exception, context: dict[str, Any]) -> None:
        """发生错误时调用"""
        pass


class LoggingMiddleware(Middleware):
    """日志中间件"""

    async def before_think(self, state: ExecutionState, context: dict[str, Any]) -> None:
        logger.info(f"Step {state.current_step}: Starting think phase...")

    async def after_think(self, state: ExecutionState, thought: str, context: dict[str, Any]) -> None:
        preview = thought[:200] + "..." if len(thought) > 200 else thought
        logger.debug(f"Thought: {preview}")

    async def before_act(self, state: ExecutionState, action: dict[str, Any], context: dict[str, Any]) -> None:
        logger.info(
            f"Step {state.current_step}: Executing skill '{action.get('skill')}' with params {action.get('parameters')}"
        )

    async def after_act(self, state: ExecutionState, result: dict[str, Any], context: dict[str, Any]) -> None:
        if result.get("success"):
            logger.info(f"Step {state.current_step}: Skill executed successfully")
        else:
            logger.warning(f"Step {state.current_step}: Skill execution failed: {result.get('error')}")

    async def on_error(self, state: ExecutionState, error: Exception, context: dict[str, Any]) -> None:
        logger.error(f"Error in step {state.current_step}: {error}")


class TelemetryMiddleware(Middleware):
    """可观测性中间件（记录耗时、Token 消耗等）"""

    def __init__(self):
        self.think_times: list[float] = []
        self.act_times: list[float] = []
        self._current_start: float | None = None

    async def before_think(self, state: ExecutionState, context: dict[str, Any]) -> None:
        self._current_start = time.time()

    async def after_think(self, state: ExecutionState, thought: str, context: dict[str, Any]) -> None:
        if self._current_start:
            self.think_times.append(time.time() - self._current_start)
            self._current_start = None

    async def before_act(self, state: ExecutionState, action: dict[str, Any], context: dict[str, Any]) -> None:
        self._current_start = time.time()

    async def after_act(self, state: ExecutionState, result: dict[str, Any], context: dict[str, Any]) -> None:
        if self._current_start:
            self.act_times.append(time.time() - self._current_start)
            self._current_start = None

    async def on_error(self, state: ExecutionState, error: Exception, context: dict[str, Any]) -> None:
        self._current_start = None

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_think_time": sum(self.think_times),
            "avg_think_time": sum(self.think_times) / len(self.think_times) if self.think_times else 0,
            "total_act_time": sum(self.act_times),
            "avg_act_time": sum(self.act_times) / len(self.act_times) if self.act_times else 0,
        }


class MiddlewareManager:
    """中间件管理器"""

    def __init__(self):
        self.middlewares: list[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        self.middlewares.append(middleware)

    def remove(self, middleware: Middleware) -> None:
        self.middlewares.remove(middleware)

    async def trigger_before_think(self, state: ExecutionState, context: dict[str, Any]) -> None:
        for mw in self.middlewares:
            await mw.before_think(state, context)

    async def trigger_after_think(self, state: ExecutionState, thought: str, context: dict[str, Any]) -> None:
        for mw in self.middlewares:
            await mw.after_think(state, thought, context)

    async def trigger_before_act(self, state: ExecutionState, action: dict[str, Any], context: dict[str, Any]) -> None:
        for mw in self.middlewares:
            await mw.before_act(state, action, context)

    async def trigger_after_act(self, state: ExecutionState, result: dict[str, Any], context: dict[str, Any]) -> None:
        for mw in self.middlewares:
            await mw.after_act(state, result, context)

    async def trigger_on_error(self, state: ExecutionState, error: Exception, context: dict[str, Any]) -> None:
        for mw in self.middlewares:
            await mw.on_error(state, error, context)
