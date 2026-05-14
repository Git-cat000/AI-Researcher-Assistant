"""
状态管理模块。
定义 Agent 执行过程中的状态结构。
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """执行状态枚举"""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ExecutionStep:
    """单步执行记录"""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thought: str | None = None
    action: dict[str, Any] | None = None
    observation: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class ExecutionState:
    """
    Agent 执行状态。
    包含任务目标、当前步骤、历史记录等。
    """

    task: str = ""
    status: ExecutionStatus = ExecutionStatus.IDLE
    current_step: int = 0
    max_steps: int = 20
    steps: list[ExecutionStep] = field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_task(self, task: str) -> None:
        """开始新任务"""
        self.task = task
        self.status = ExecutionStatus.THINKING
        self.current_step = 0
        self.steps = []
        self.final_answer = None

    def add_step(self, step: ExecutionStep) -> None:
        """添加执行步骤"""
        self.steps.append(step)
        self.current_step += 1

    def complete(self, answer: str) -> None:
        """标记任务完成"""
        self.status = ExecutionStatus.COMPLETED
        self.final_answer = answer

    def fail(self, error: str) -> None:
        """标记任务失败"""
        self.status = ExecutionStatus.FAILED
        self.final_answer = f"Task failed: {error}"

    def get_last_step(self) -> ExecutionStep | None:
        """获取最后一步"""
        return self.steps[-1] if self.steps else None

    def get_context_for_llm(self) -> str:
        """生成供 LLM 查看的执行历史上下文"""
        if not self.steps:
            return "No previous steps."

        lines = ["Previous steps:"]
        for i, step in enumerate(self.steps):
            lines.append(f"Step {i + 1}:")
            if step.thought:
                lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
            if step.observation:
                lines.append(f"  Observation: {step.observation}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status.value,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }
