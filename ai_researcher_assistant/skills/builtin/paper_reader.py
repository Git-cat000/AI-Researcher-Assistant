"""PDF paper reader skill."""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any

import requests

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class PaperReaderSkill(BaseSkill):
    """Parse PDFs and extract text, metadata, and heuristic sections."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="paper_reader",
            description="Parse and extract text, metadata, and references from PDF papers.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="file_path",
                    description="Path to local PDF file",
                    type="string",
                    required=False,
                ),
                SkillParameter(
                    name="url",
                    description="URL of the PDF file, for example an arXiv PDF link",
                    type="string",
                    required=False,
                ),
                SkillParameter(
                    name="extract_sections",
                    description="Whether to extract sections such as abstract, introduction, and conclusion",
                    type="boolean",
                    required=False,
                    default=True,
                ),
                SkillParameter(
                    name="max_pages",
                    description="Maximum number of pages to process, or 0 for all pages",
                    type="integer",
                    required=False,
                    default=50,
                ),
            ],
            tags=["academic", "pdf", "parser", "reader"],
            instructions="""
Use this skill to read and extract content from PDF papers.
Provide either a local file path or a URL to the PDF.
The skill returns full text and optionally structured sections.
            """,
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        file_path = parameters.get("file_path")
        url = parameters.get("url")
        extract_sections = parameters.get("extract_sections", True)
        max_pages = parameters.get("max_pages", 50)

        if not file_path and not url:
            return {"success": False, "result": None, "error": "Either file_path or url must be provided"}

        temp_file = None
        if url:
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    temp_file = tmp.name
                file_path = temp_file
            except Exception as exc:
                return {"success": False, "result": None, "error": f"Failed to download PDF: {exc}"}

        try:
            pdf_reader_class = self._load_pdf_reader()
            if pdf_reader_class is None:
                return {
                    "success": False,
                    "result": None,
                    "error": "PDF parsing library not available. Install pypdf or PyPDF2.",
                }

            reader = pdf_reader_class(file_path)
            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, max_pages) if max_pages > 0 else total_pages

            full_text = ""
            for index in range(pages_to_read):
                page = reader.pages[index]
                full_text += (page.extract_text() or "") + "\n"

            metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    if value:
                        metadata[str(key).strip("/")] = str(value)

            result = {
                "full_text": full_text,
                "metadata": metadata,
                "total_pages": total_pages,
                "pages_read": pages_to_read,
            }
            if extract_sections:
                result["sections"] = self._extract_sections(full_text)

            return {"success": True, "result": result, "error": None}
        except Exception as exc:
            return {"success": False, "result": None, "error": f"PDF parsing error: {exc}"}
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _load_pdf_reader(self) -> Any | None:
        try:
            from pypdf import PdfReader

            return PdfReader
        except ImportError:
            try:
                import PyPDF2

                return PyPDF2.PdfReader
            except ImportError:
                return None

    def _extract_sections(self, text: str) -> dict[str, str]:
        """Extract common paper sections with simple heading heuristics."""

        sections: dict[str, str] = {}
        patterns = {
            "abstract": r"(?i)(abstract|summary)",
            "introduction": r"(?i)(introduction|background)",
            "method": r"(?i)(method|approach|model|framework)",
            "results": r"(?i)(results?|findings|experiments?)",
            "discussion": r"(?i)(discussion|analysis)",
            "conclusion": r"(?i)(conclusion|summary|future work)",
            "references": r"(?i)(references?|bibliography)",
        }

        current_section: str | None = None
        section_content: list[str] = []
        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for section_name, pattern in patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    if current_section and section_content:
                        sections[current_section] = "\n".join(section_content)
                    current_section = section_name
                    section_content = []
                    break
            else:
                if current_section:
                    section_content.append(line)

        if current_section and section_content:
            sections[current_section] = "\n".join(section_content)

        return sections
