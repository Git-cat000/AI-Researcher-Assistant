"""自定义异常类"""


class AgentError(Exception):
    """Agent 基础异常"""

    pass


class LLMError(AgentError):
    """LLM 调用相关错误"""

    pass


class SkillError(AgentError):
    """技能执行错误"""

    pass


class MemoryError(AgentError):
    """记忆系统错误"""

    pass


class ConfigurationError(AgentError):
    """配置错误"""

    pass


class ToolNotFoundError(SkillError):
    """请求的技能/工具不存在"""

    pass


class SkillExecutionError(SkillError):
    """技能执行过程中出错"""

    pass


class SandboxError(AgentError):
    """沙箱执行错误"""

    pass
