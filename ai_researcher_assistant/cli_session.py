"""Stateful command-line session helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_researcher_assistant.core.config import AgentConfig, load_config_from_env
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.orchestration import ResearcherAgent
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry
from ai_researcher_assistant.skills.builtin.paper_reader import PaperReaderSkill


@dataclass
class SessionCommandResult:
    """Result returned by a slash command inside the interactive CLI."""

    output: str
    should_exit: bool = False
    markdown: bool = False


class AgentCliSession:
    """Long-running agent session scoped to one working directory."""

    def __init__(
        self,
        cwd: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        skills_dir: list[Path] | None = None,
        no_builtin: bool = False,
        auto_rag: bool = True,
        config: AgentConfig | None = None,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.auto_rag = auto_rag
        self.config = config or load_config_from_env()
        self._apply_model_overrides(provider=provider, model=model, base_url=base_url)

        self.registry = SkillRegistry()
        self.loader = SkillLoader(self.registry)
        if not no_builtin:
            self.loader.load_builtin_skills()
        for path in skills_dir or []:
            self.loader.load_from_directory(str(self.resolve_path(path)))

        self.rag = AcademicRAG()
        self.agent = ResearcherAgent(config=self.config, enable_builtin_skills=False, skill_registry=self.registry)
        self.agent.rag = self.rag

    async def ask(self, task: str) -> str:
        """Run one task through the agent, optionally injecting local RAG context."""

        prompt = task
        if self.auto_rag and self.rag.count_papers() > 0:
            rag_context = self.rag.build_context(task, top_k=5, include_full_text=False)
            if rag_context.strip():
                prompt = f"{task}\n\nRelevant local paper context:\n{rag_context}"
        self.agent.rag = self.rag
        return await self.agent.aprocess(prompt)

    def handle_command(self, raw_command: str) -> SessionCommandResult:
        """Handle one slash command without invoking the LLM."""

        parts = shlex.split(raw_command.strip())
        if not parts:
            return SessionCommandResult("")

        command = parts[0].lower()
        args = parts[1:]
        if command in {"/exit", "/quit", "/q"}:
            return SessionCommandResult("Bye.", should_exit=True)
        if command == "/help":
            return SessionCommandResult(self.help_text(), markdown=True)
        if command == "/pwd":
            return SessionCommandResult(str(self.cwd))
        if command == "/clear":
            self.agent.reset_conversation()
            return SessionCommandResult("Conversation cleared.")
        if command == "/skills":
            return SessionCommandResult("\n".join(self.registry.list_skills()) or "No skills loaded.")
        if command == "/stats":
            stats = self.agent.get_stats()
            stats["rag_papers"] = self.rag.count_papers()
            return SessionCommandResult(json.dumps(stats, ensure_ascii=False, indent=2))
        if command == "/model":
            return self._handle_model_command(args)
        if command == "/rag":
            return self._handle_rag_command(args)
        return SessionCommandResult(f"Unknown command: {command}. Use /help to list commands.")

    def help_text(self) -> str:
        return """# AI Researcher Assistant Session

Commands:

- `/help` show this help
- `/pwd` show the session working directory
- `/skills` list loaded skills
- `/stats` show conversation and RAG stats
- `/model` show the current provider/model
- `/model PROVIDER MODEL` switch provider/model for later turns
- `/rag load PATH` load a JSONL paper library into this session
- `/rag ingest-pdf PATH_OR_URL` parse a local PDF or PDF URL into this session RAG
- `/rag search QUERY` search the session RAG
- `/clear` clear short-term conversation memory
- `/exit` leave the session

Normal input is sent to the agent. If local papers are loaded, compact RAG context is attached automatically.
"""

    def load_papers_jsonl(self, path: str | Path) -> int:
        count = 0
        with self.resolve_path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                add_record_to_rag(self.rag, record)
                count += 1
        self.agent.rag = self.rag
        return count

    def ingest_pdf(self, source: str | Path, **metadata: Any) -> dict[str, Any]:
        record = paper_record_from_pdf(source, cwd=self.cwd, **metadata)
        add_record_to_rag(self.rag, record)
        self.agent.rag = self.rag
        return record

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.cwd / candidate

    def _handle_model_command(self, args: list[str]) -> SessionCommandResult:
        if not args:
            return SessionCommandResult(f"provider={self.config.llm.provider}, model={self.config.llm.model}")
        if len(args) < 2:
            return SessionCommandResult("Usage: /model PROVIDER MODEL")
        self._apply_model_overrides(provider=args[0], model=args[1], base_url=None)
        self.agent.shutdown()
        self.agent = ResearcherAgent(config=self.config, enable_builtin_skills=False, skill_registry=self.registry)
        self.agent.rag = self.rag
        return SessionCommandResult(f"Switched to provider={self.config.llm.provider}, model={self.config.llm.model}")

    def _handle_rag_command(self, args: list[str]) -> SessionCommandResult:
        if not args:
            return SessionCommandResult("Usage: /rag load PATH | /rag search QUERY | /rag ingest-pdf PATH_OR_URL")
        subcommand = args[0].lower()
        if subcommand == "load" and len(args) >= 2:
            count = self.load_papers_jsonl(args[1])
            return SessionCommandResult(
                f"Loaded {count} paper records. Session RAG now has {self.rag.count_papers()} papers."
            )
        if subcommand == "search" and len(args) >= 2:
            query = " ".join(args[1:])
            results = self.rag.search_papers(query, top_k=5)
            return SessionCommandResult(format_paper_results(results), markdown=True)
        if subcommand == "ingest-pdf" and len(args) >= 2:
            record = self.ingest_pdf(args[1])
            return SessionCommandResult(
                f"Ingested `{record['title']}`. Session RAG now has {self.rag.count_papers()} papers.",
                markdown=True,
            )
        return SessionCommandResult("Usage: /rag load PATH | /rag search QUERY | /rag ingest-pdf PATH_OR_URL")

    def _apply_model_overrides(self, provider: str | None, model: str | None, base_url: str | None) -> None:
        if provider:
            self.config.llm.provider = provider  # type: ignore[assignment]
        if model:
            self.config.llm.model = model
        if base_url:
            self.config.llm.base_url = base_url


def paper_record_from_pdf(
    source: str | Path,
    cwd: str | Path | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    arxiv_id: str | None = None,
    categories: list[str] | None = None,
    published_date: str | None = None,
    max_pages: int = 0,
) -> dict[str, Any]:
    """Extract one normalized paper record from a local PDF path or PDF URL."""

    source_text = str(source)
    parameters: dict[str, Any] = {"extract_sections": True, "max_pages": max_pages}
    if source_text.startswith(("http://", "https://")):
        parameters["url"] = source_text
        fallback_title = Path(source_text.rstrip("/")).stem or "Remote PDF"
    else:
        path = Path(source_text)
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        parameters["file_path"] = str(path)
        fallback_title = path.stem

    result = PaperReaderSkill().execute(parameters, {})
    if not result.get("success"):
        raise ValueError(result.get("error") or "Failed to parse PDF")

    parsed = result["result"]
    sections = parsed.get("sections", {})
    full_text = parsed.get("full_text", "")
    extracted_title = parsed.get("metadata", {}).get("Title")
    abstract = sections.get("abstract") or full_text[:2000]

    return {
        "title": title or extracted_title or fallback_title,
        "abstract": abstract,
        "full_text": full_text,
        "authors": authors or [],
        "arxiv_id": arxiv_id,
        "categories": categories or [],
        "published_date": published_date,
        "citations": [],
        "metadata": {
            "source": source_text,
            "pdf_metadata": parsed.get("metadata", {}),
            "total_pages": parsed.get("total_pages"),
            "pages_read": parsed.get("pages_read"),
        },
    }


def add_record_to_rag(rag: AcademicRAG, record: dict[str, Any]) -> list[str]:
    return rag.add_paper(
        title=record["title"],
        abstract=record.get("abstract", ""),
        full_text=record.get("full_text"),
        authors=record.get("authors"),
        arxiv_id=record.get("arxiv_id"),
        categories=record.get("categories"),
        published_date=record.get("published_date"),
        citations=record.get("citations"),
        metadata=record.get("metadata"),
    )


def append_record_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_paper_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No papers found."
    lines = []
    for index, paper in enumerate(results, start=1):
        title = paper.get("title") or "Untitled"
        lines.append(f"{index}. **{title}**")
        if paper.get("authors"):
            lines.append(f"   Authors: {', '.join(paper['authors'])}")
        if paper.get("arxiv_id"):
            lines.append(f"   arXiv: {paper['arxiv_id']}")
        if paper.get("categories"):
            lines.append(f"   Categories: {', '.join(paper['categories'])}")
        if paper.get("abstract"):
            abstract = str(paper["abstract"]).replace("\n", " ")
            lines.append(f"   Abstract: {abstract[:300]}")
    return "\n".join(lines)
