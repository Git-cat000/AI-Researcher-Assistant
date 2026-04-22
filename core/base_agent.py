"""
Agent 基类。
所有具体 Agent 实现都应继承此类。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, AsyncIterator
from AI-Researcher-Assistant.core.message import Conversation, MessageRole
from AI-Researcher-Assistant.core.config import AgentConfig, get_config


class BaseAgent(ABC):
    """
    Agent 抽象基类。
    
    定义了 Agent 的基本生命周期：
    1. 初始化配置
    2. 处理消息（同步/异步/流式）
    3. 管理对话历史
    """

    def __init__(self, config: Optional[AgentConfig] = None, name: str = "Agent"):
        self.config = config or get_config()
        self.name = name
        self.conversation = Conversation()
        self._initialized = False

    def initialize(self):
        """初始化 Agent（加载资源、建立连接等）"""
        self._initialized = True

    def shutdown(self):
        """清理资源"""
        pass

    @abstractmethod
    def process(self, user_input: str) -> str:
        """
        同步处理用户输入，返回最终回复。
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            Agent 的最终回复
        """
        pass

    async def aprocess(self, user_input: str) -> str:
        """
        异步处理用户输入（默认调用同步方法，子类可覆盖）。
        """
        return self.process(user_input)

    async def stream_process(self, user_input: str) -> AsyncIterator[str]:
        """
        流式处理用户输入，逐步返回生成的内容。
        
        Args:
            user_input: 用户输入
            
        Yields:
            逐步生成的文本片段
        """
        # 默认实现：一次性返回
        result = await self.aprocess(user_input)
        yield result

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation.clear()

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.conversation.add_system(prompt)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
