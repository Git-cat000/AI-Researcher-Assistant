"""Basic unit tests for AI Researcher Assistant."""

from ai_researcher_assistant.core import Conversation, MessageRole
from ai_researcher_assistant.llm import BaseLLM, LLMResponse
from ai_researcher_assistant.memory import ShortTermMemory
from ai_researcher_assistant.orchestration import ReActLoop
from ai_researcher_assistant.skills import SkillRegistry


class ScriptedLLM(BaseLLM):
    """Mock LLM that returns pre-configured responses for testing loops."""

    def __init__(self, responses=None):
        super().__init__(model="mock", temperature=0.0)
        self.responses = list(responses or [])

    def generate(self, messages, **kwargs):
        content = self.responses.pop(0) if self.responses else "Final Answer: Done."
        return LLMResponse(content=content, model="mock")

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)

    async def stream_generate(self, messages, **kwargs):
        yield self.generate(messages, **kwargs).content


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


async def test_react_loop_completes_with_final_answer():
    """ReAct loop returns execution state with a final answer."""
    llm = ScriptedLLM(["Final Answer: Test completed."])
    registry = SkillRegistry()
    loop = ReActLoop(llm=llm, skill_registry=registry)

    state = await loop.run("Test task", context={})

    assert state.final_answer == "Test completed."
    assert state.status.value == "completed"


async def test_react_loop_detects_skill_action():
    """ReAct loop parses an action and attempts skill execution."""
    action_json = (
        "Thought: Use skill.\n"
        "Action:\n"
        "```json\n"
        '{"skill": "paper_writer", "parameters": {"action": "summarize", "text": "test"}}\n'
        "```"
    )
    llm = ScriptedLLM([action_json, "Final Answer: Skill called."])
    registry = SkillRegistry()
    loop = ReActLoop(llm=llm, skill_registry=registry)

    state = await loop.run("Test task", context={})

    assert state.final_answer is not None


def test_config():
    """Test configuration loading"""
    from ai_researcher_assistant.core.config import get_config

    config = get_config()
    assert config.llm.provider == "openai"
    assert config.memory.chunk_size == 1000
