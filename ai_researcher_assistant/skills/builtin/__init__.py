"""Built-in skills.

External skill dependencies are imported lazily so one built-in skill can be
tested without installing every integration package.
"""


def __getattr__(name: str):
    if name == "ArxivFetcherSkill":
        from ai_researcher_assistant.skills.builtin.arxiv_fetcher import ArxivFetcherSkill

        return ArxivFetcherSkill
    if name == "PaperReaderSkill":
        from ai_researcher_assistant.skills.builtin.paper_reader import PaperReaderSkill

        return PaperReaderSkill
    if name == "PaperWriterSkill":
        from ai_researcher_assistant.skills.builtin.paper_writer import PaperWriterSkill

        return PaperWriterSkill
    if name == "HarnessCoordinationSkill":
        from ai_researcher_assistant.skills.builtin.harness_coordination import HarnessCoordinationSkill

        return HarnessCoordinationSkill
    if name == "RagSearchSkill":
        from ai_researcher_assistant.skills.builtin.rag_search import RagSearchSkill

        return RagSearchSkill
    if name == "SubagentTaskSkill":
        from ai_researcher_assistant.skills.builtin.subagent_task import SubagentTaskSkill

        return SubagentTaskSkill
    raise AttributeError(name)


__all__ = [
    "ArxivFetcherSkill",
    "PaperReaderSkill",
    "PaperWriterSkill",
    "HarnessCoordinationSkill",
    "RagSearchSkill",
    "SubagentTaskSkill",
]
