"""Message types and conversation window management."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Roles used inside the agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SKILL = "skill"
    OBSERVATION = "observation"


@dataclass
class Message:
    """A single conversation message."""

    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""

        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_llm_format(self) -> dict[str, str]:
        """Convert to the common chat-completions message format."""

        if self.role == MessageRole.TOOL:
            return {"role": "user", "content": f"[Tool result] {self.content}"}
        if self.role == MessageRole.SKILL:
            return {"role": "user", "content": f"[Skill output] {self.content}"}
        if self.role == MessageRole.OBSERVATION:
            return {"role": "user", "content": f"[Observation] {self.content}"}
        return {"role": self.role.value, "content": self.content}


@dataclass
class Conversation:
    """Conversation history with a simple token-budgeted context window."""

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None

    def add(self, role: MessageRole, content: str, **metadata: Any) -> Message:
        """Append a message to the conversation."""

        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def add_system(self, content: str) -> None:
        """Set or replace the system prompt."""

        self.system_prompt = content

    def get_context_window(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Return messages for an LLM call, keeping the newest history within budget."""

        if max_tokens is None:
            result = []
            if self.system_prompt:
                result.append({"role": "system", "content": self.system_prompt})
            result.extend(msg.to_llm_format() for msg in self.messages)
            return result

        system_message = {"role": "system", "content": self.system_prompt} if self.system_prompt else None
        system_tokens = estimate_tokens(self.system_prompt or "") if system_message else 0
        budget = max(0, max_tokens - system_tokens)

        selected: list[dict[str, str]] = []
        used_tokens = 0
        for msg in reversed(self.messages):
            formatted = msg.to_llm_format()
            tokens = estimate_tokens(formatted["content"])
            if selected and used_tokens + tokens > budget:
                break
            if not selected and tokens > budget:
                formatted = {
                    "role": formatted["role"],
                    "content": truncate_to_token_budget(formatted["content"], budget),
                }
                tokens = estimate_tokens(formatted["content"])
            selected.insert(0, formatted)
            used_tokens += tokens

        if system_message:
            return [system_message, *selected]
        return selected

    def clear(self) -> None:
        """Clear conversation history while preserving the system prompt."""

        self.messages.clear()

    def last(self) -> Message | None:
        """Return the most recent message."""

        return self.messages[-1] if self.messages else None


def estimate_tokens(text: str) -> int:
    """Estimate tokens without provider-specific tokenizers."""

    if not text:
        return 0
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_chars = len(text) - cjk_chars
    return max(1, cjk_chars + other_chars // 4)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to a rough token budget while preserving a useful suffix."""

    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    approximate_chars = max_tokens * 4
    suffix = text[-approximate_chars:]
    return f"[truncated]\n{suffix}"
