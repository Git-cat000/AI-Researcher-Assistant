"""
消息类型定义。
用于 Agent 内部各模块间通信，以及 LLM 对话历史管理。
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Any, Dict, List
from enum import Enum
from datetime import datetime
import uuid


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"           # 工具执行结果
    SKILL = "skill"         # 技能调用
    OBSERVATION = "observation"  # 环境观察（ReAct 循环）


@dataclass
class Message:
    """基础消息类"""
    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于序列化和 LLM API 调用"""
        return {
            "role": self.role.value,
            "content": self.content,
            # 可选添加 metadata（某些 LLM API 可能不支持）
        }

    def to_llm_format(self) -> Dict[str, str]:
        """转换为标准的 LLM 消息格式"""
        # 将 TOOL/SKILL/OBSERVATION 映射为 user 或 assistant，取决于上下文
        if self.role == MessageRole.TOOL:
            return {"role": "user", "content": f"[工具结果] {self.content}"}
        elif self.role == MessageRole.SKILL:
            return {"role": "user", "content": f"[技能输出] {self.content}"}
        elif self.role == MessageRole.OBSERVATION:
            return {"role": "user", "content": f"[观察] {self.content}"}
        else:
            return {"role": self.role.value, "content": self.content}


@dataclass
class Conversation:
    """对话历史管理器"""
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def add(self, role: MessageRole, content: str, **metadata):
        """添加一条消息"""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_system(self, content: str):
        """设置或更新系统提示"""
        self.system_prompt = content

    def get_context_window(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """获取用于 LLM 调用的消息列表"""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            result.append(msg.to_llm_format())
        # TODO: 实现 token 截断逻辑
        return result

    def clear(self):
        """清空历史（保留系统提示）"""
        self.messages.clear()

    def last(self) -> Optional[Message]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
