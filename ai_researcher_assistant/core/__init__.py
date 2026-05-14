"""ai_researcher_assistant Framework - Core Module"""

from ai_researcher_assistant.core.base_agent import BaseAgent
from ai_researcher_assistant.core.config import (
    AgentConfig,
    LLMConfig,
    MemoryConfig,
    SkillsConfig,
    get_config,
    load_config_from_env,
    update_config,
)
from ai_researcher_assistant.core.exceptions import (
    AgentError,
    ConfigurationError,
    LLMError,
    MemoryError,
    SkillError,
)
from ai_researcher_assistant.core.message import (
    Conversation,
    Message,
    MessageRole,
)

__all__ = [
    # Config
    "AgentConfig",
    "LLMConfig",
    "MemoryConfig",
    "SkillsConfig",
    "get_config",
    "load_config_from_env",
    "update_config",
    # Exceptions
    "AgentError",
    "LLMError",
    "SkillError",
    "MemoryError",
    "ConfigurationError",
    # Message
    "Message",
    "Conversation",
    "MessageRole",
    # BaseAgent
    "BaseAgent",
]
