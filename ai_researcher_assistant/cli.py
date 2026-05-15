"""Command line interface for AI Researcher Assistant."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ai_researcher_assistant import __version__
from ai_researcher_assistant.cli_session import (
    AgentCliSession,
    append_record_jsonl,
    paper_record_from_pdf,
)
from ai_researcher_assistant.core.config import load_config_from_env
from ai_researcher_assistant.llm import list_model_capabilities
from ai_researcher_assistant.memory import AcademicRAG
from ai_researcher_assistant.orchestration import ResearcherAgent
from ai_researcher_assistant.skills import SkillLoader, SkillRegistry

app = typer.Typer(help="AI Researcher Assistant command line tools.")
skills_app = typer.Typer(help="Inspect and load Python or Markdown skills.")
rag_app = typer.Typer(help="Run local RAG indexing, search, and citation graph utilities.")
app.add_typer(skills_app, name="skills")
app.add_typer(rag_app, name="rag")


@app.command()
def version(
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """Print package version."""

    payload = {"version": __version__}
    _emit(payload, json_output)


@app.command()
def providers(
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """List supported model providers and their coarse capabilities."""

    payload = [capability.to_dict() for capability in list_model_capabilities()]
    _emit(payload, json_output)


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """Run a lightweight local environment check."""

    registry = SkillRegistry()
    loaded = SkillLoader(registry).load_builtin_skills()
    payload = {
        "version": __version__,
        "builtin_skills": [skill.manifest.name for skill in loaded],
        "skill_count": len(loaded),
        "providers": [capability.provider for capability in list_model_capabilities()],
    }
    _emit(payload, json_output)


@app.command()
def ask(
    task: Annotated[str, typer.Argument(help="Research task to send to the agent.")],
    provider: Annotated[str | None, typer.Option("--provider", help="Override configured LLM provider.")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Override configured LLM model.")] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override OpenAI-compatible or Ollama base URL."),
    ] = None,
    skills_dir: Annotated[
        list[Path] | None,
        typer.Option("--skills-dir", help="Directory or SKILL.md file to load."),
    ] = None,
    no_builtin: Annotated[bool, typer.Option("--no-builtin", help="Do not load built-in skills.")] = False,
) -> None:
    """Run one agent task. Requires credentials for the selected provider."""

    config = load_config_from_env()
    if provider:
        config.llm.provider = provider  # type: ignore[assignment]
    if model:
        config.llm.model = model
    if base_url:
        config.llm.base_url = base_url

    registry = SkillRegistry()
    loader = SkillLoader(registry)
    if not no_builtin:
        loader.load_builtin_skills()
    for path in skills_dir or []:
        loader.load_from_directory(str(path))

    agent = ResearcherAgent(config=config, enable_builtin_skills=False, skill_registry=registry)
    answer = asyncio.run(agent.aprocess(task))
    typer.echo(answer)


@app.command()
def chat(
    cwd: Annotated[Path, typer.Option("--cwd", help="Working directory for this long-running session.")] = Path("."),
    provider: Annotated[str | None, typer.Option("--provider", help="Override configured LLM provider.")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Override configured LLM model.")] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override OpenAI-compatible or Ollama base URL."),
    ] = None,
    skills_dir: Annotated[
        list[Path] | None,
        typer.Option("--skills-dir", help="Directory or SKILL.md file to load."),
    ] = None,
    no_builtin: Annotated[bool, typer.Option("--no-builtin", help="Do not load built-in skills.")] = False,
    no_auto_rag: Annotated[
        bool,
        typer.Option("--no-auto-rag", help="Do not attach local RAG context automatically to normal prompts."),
    ] = False,
) -> None:
    """Start a persistent Claude Code-like terminal session in one working directory."""

    asyncio.run(
        _run_chat(
            AgentCliSession(
                cwd=cwd,
                provider=provider,
                model=model,
                base_url=base_url,
                skills_dir=skills_dir,
                no_builtin=no_builtin,
                auto_rag=not no_auto_rag,
            )
        )
    )


@skills_app.command("list")
def list_skills(
    skills_dir: Annotated[
        list[Path] | None,
        typer.Option("--skills-dir", help="Directory or SKILL.md file to load."),
    ] = None,
    no_builtin: Annotated[bool, typer.Option("--no-builtin", help="Do not load built-in skills.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """List built-in and user-provided skills."""

    registry = SkillRegistry()
    loader = SkillLoader(registry)
    if not no_builtin:
        loader.load_builtin_skills()
    for path in skills_dir or []:
        loader.load_from_directory(str(path))
    payload = [
        {
            "name": manifest.name,
            "description": manifest.description,
            "parameters": [parameter.__dict__ for parameter in manifest.parameters],
            "tags": manifest.tags,
        }
        for manifest in registry.get_all_manifests().values()
    ]
    _emit(payload, json_output)


@skills_app.command("inspect")
def inspect_skill(
    path: Annotated[Path, typer.Argument(help="Path to a SKILL.md or standalone Markdown skill.")],
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """Inspect one Markdown skill file without executing it."""

    registry = SkillRegistry()
    skill = SkillLoader(registry).load_from_markdown(path)
    manifest = skill.manifest
    payload = {
        "name": manifest.name,
        "description": manifest.description,
        "parameters": [parameter.__dict__ for parameter in manifest.parameters],
        "tags": manifest.tags,
        "metadata": manifest.metadata,
    }
    _emit(payload, json_output)


@rag_app.command("search")
def rag_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    paper_jsonl: Annotated[Path, typer.Option("--paper-jsonl", help="JSONL file containing paper records.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="Number of papers to return.")] = 5,
    category: Annotated[
        list[str] | None,
        typer.Option("--category", help="Category filter; can be repeated."),
    ] = None,
    author: Annotated[list[str] | None, typer.Option("--author", help="Author filter; can be repeated.")] = None,
    year: Annotated[str | None, typer.Option("--year", help="Published year filter.")] = None,
    mode: Annotated[str, typer.Option("--mode", help="Retrieval mode: hybrid, vector, keyword, or bm25.")] = "hybrid",
    include_full_text: Annotated[
        bool,
        typer.Option("--include-full-text", help="Include excerpts in context output."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """Build a local in-memory RAG index from JSONL and search it."""

    rag = _load_rag_from_jsonl(paper_jsonl)
    results = rag.search_papers(
        query,
        top_k=top_k,
        categories=category,
        authors=author,
        published_year=year,
        retrieval_mode=mode,
    )
    payload = {
        "results": results,
        "context": rag.build_context(
            query,
            top_k=top_k,
            include_full_text=include_full_text,
            categories=category,
            authors=author,
            published_year=year,
            retrieval_mode=mode,
        ),
    }
    _emit(payload, json_output)


@rag_app.command("graph")
def rag_graph(
    paper_jsonl: Annotated[Path, typer.Option("--paper-jsonl", help="JSONL file containing paper records.")],
    json_output: Annotated[bool, typer.Option("--json/--text", help="Output JSON or a compact text summary.")] = True,
) -> None:
    """Build a citation graph from paper metadata."""

    rag = _load_rag_from_jsonl(paper_jsonl)
    _emit(rag.citation_graph(), json_output)


@rag_app.command("ingest-pdf")
def rag_ingest_pdf(
    source: Annotated[str, typer.Argument(help="Local PDF path or direct PDF URL.")],
    output: Annotated[Path | None, typer.Option("--output", help="Append normalized paper record to JSONL.")] = None,
    title: Annotated[str | None, typer.Option("--title", help="Override detected paper title.")] = None,
    author: Annotated[list[str] | None, typer.Option("--author", help="Author name; can be repeated.")] = None,
    arxiv_id: Annotated[str | None, typer.Option("--arxiv-id", help="Optional arXiv ID or stable paper ID.")] = None,
    category: Annotated[list[str] | None, typer.Option("--category", help="Category/tag; can be repeated.")] = None,
    published_date: Annotated[str | None, typer.Option("--published-date", help="Publication date string.")] = None,
    max_pages: Annotated[int, typer.Option("--max-pages", help="Maximum pages to read, or 0 for all.")] = 0,
    json_output: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON.")] = False,
) -> None:
    """Parse a local PDF or direct PDF URL into a normalized JSONL paper record."""

    record = paper_record_from_pdf(
        source,
        title=title,
        authors=author,
        arxiv_id=arxiv_id,
        categories=category,
        published_date=published_date,
        max_pages=max_pages,
    )
    if output:
        append_record_jsonl(output, record)
    payload = {"record": record, "output": str(output) if output else None}
    _emit(payload, json_output)


def _load_rag_from_jsonl(path: Path) -> AcademicRAG:
    rag = AcademicRAG()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rag.add_paper(
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
    return rag


def _emit(payload: Any, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if isinstance(payload, list):
        for item in payload:
            typer.echo(json.dumps(item, ensure_ascii=False))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")
    else:
        typer.echo(str(payload))


async def _run_chat(session: AgentCliSession) -> None:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.patch_stdout import patch_stdout
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError as exc:
        raise typer.BadParameter("Interactive chat requires prompt-toolkit and rich to be installed.") from exc

    state_dir = session.cwd / ".ai-researcher"
    state_dir.mkdir(exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(state_dir / "history.txt")),
        completer=WordCompleter(
            ["/help", "/pwd", "/skills", "/stats", "/model", "/rag", "/clear", "/exit", "/quit"],
            ignore_case=True,
        ),
    )
    console = Console()
    console.print(f"[bold]AI Researcher Assistant[/bold] session: {session.cwd}")
    console.print("Type /help for commands. Type /exit to quit.")

    with patch_stdout():
        while True:
            try:
                user_input = (await prompt_session.prompt_async("research> ")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("Bye.")
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                result = session.handle_command(user_input)
                if result.output:
                    console.print(Markdown(result.output) if result.markdown else result.output)
                if result.should_exit:
                    break
                continue
            try:
                with console.status("Thinking...", spinner="dots"):
                    answer = await session.ask(user_input)
                console.print(Markdown(answer))
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
