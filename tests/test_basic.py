"""
Basic unit tests for AI Researcher Assistant.
Run with: pytest tests/
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ai_researcher_assistant.core import AgentConfig, Message, Conversation, MessageRole
from ai_researcher_assistant.llm import BaseLLM, LLMResponse
from ai_researcher_assistant.memory import ShortTermMemory
from ai_researcher_assistant.skills import SkillRegistry
from ai_researcher_assistant.orchestration import ReActLoop


class MockLLM(BaseLLM):
    """Mock LLM for testing without API calls"""
    
    def __init__(self):
        super().__init__(model="mock", temperature=0.0)
    
    def generate(self, messages, **kwargs):
        return LLMResponse(content="Mock response", model="mock")
    
    async def agenerate(self, messages, **kwargs):
        return LLMResponse(content="Mock response", model="mock")
    
    async def stream_generate(self, messages, **kwargs):
        yield "Mock"
        yield " response"


def test_short_term_memory():
    """Test short-term memory operations"""
    mem = ShortTermMemory(max_tokens=1000, max_messages=10)
    
    msg_id = mem.add("Hello world", metadata={"role": "user"})
    assert mem.count() == 1
    
    item = mem.get(msg_id)
    assert item.content == "Hello world"
    
    mem.clear()
    assert mem.count() == 0


def test_message_conversation():
    """Test message and conversation structures"""
    conv = Conversation()
    conv.add_system("You are a helpful assistant")
    conv.add(MessageRole.USER, "Hi")
    conv.add(MessageRole.ASSISTANT, "Hello")
    
    context = conv.get_context_window()
    assert len(context) == 3
    assert context[0]["role"] == "system"
    assert context[1]["role"] == "user"


@pytest.mark.asyncio
async def test_react_loop_mock():
    """Test ReAct loop with mock LLM and skill registry"""
    llm = MockLLM()
    registry = SkillRegistry()
    
    loop = ReActLoop(llm=llm, skill_registry=registry)
    
    # This would normally call LLM and skills; with mocks it should complete
    # For a real test, we'd need to mock the skill execution as well
    # For now, we just test initialization
    assert loop.llm is not None
    assert loop.skill_registry is not None


def test_config():
    """Test configuration loading"""
    from ai_researcher_assistant.core.config import get_config, LLMConfig
    
    config = get_config()
    assert config.llm.provider == "openai"
    assert config.memory.chunk_size == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
