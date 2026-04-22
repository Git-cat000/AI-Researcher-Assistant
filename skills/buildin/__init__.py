"""AI Researcher Assistant - Built-in Skills"""

from ai_researcher_assistant.skills.builtin.arxiv_fetcher import ArxivFetcherSkill
from ai_researcher_assistant.skills.builtin.paper_reader import PaperReaderSkill
from ai_researcher_assistant.skills.builtin.paper_writer import PaperWriterSkill

__all__ = [
    "ArxivFetcherSkill",
    "PaperReaderSkill",
    "PaperWriterSkill",
]
