"""
论文阅读技能。
支持解析 PDF 文件，提取文本、元数据和参考文献。
"""
import os
import re
from typing import Dict, Any, List, Optional
import tempfile
import requests

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class PaperReaderSkill(BaseSkill):
    """解析和阅读 PDF 论文"""

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
                    description="URL of the PDF file (e.g., arXiv PDF link)",
                    type="string",
                    required=False,
                ),
                SkillParameter(
                    name="extract_sections",
                    description="Whether to extract sections (abstract, introduction, conclusion, etc.)",
                    type="boolean",
                    required=False,
                    default=True,
                ),
                SkillParameter(
                    name="max_pages",
                    description="Maximum number of pages to process (0 for all)",
                    type="integer",
                    required=False,
                    default=50,
                ),
            ],
            tags=["academic", "pdf", "parser", "reader"],
            instructions="""
Use this skill to read and extract content from PDF papers.
Provide either a local file path or a URL to the PDF.
The skill returns the full text and optionally structured sections.
You can then use this content for summarization, question answering, or adding to RAG.
            """,
        )

    def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        file_path = parameters.get("file_path")
        url = parameters.get("url")
        extract_sections = parameters.get("extract_sections", True)
        max_pages = parameters.get("max_pages", 50)

        if not file_path and not url:
            return {
                "success": False,
                "result": None,
                "error": "Either file_path or url must be provided",
            }

        # 如果是 URL，先下载到临时文件
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
            except Exception as e:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Failed to download PDF: {str(e)}",
                }

        try:
            # 尝试导入 PDF 解析库
            try:
                import pypdf
                from pypdf import PdfReader
            except ImportError:
                try:
                    import PyPDF2
                    PdfReader = PyPDF2.PdfReader
                except ImportError:
                    return {
                        "success": False,
                        "result": None,
                        "error": "PDF parsing library not available. Install pypdf or PyPDF2.",
                    }

            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, max_pages) if max_pages > 0 else total_pages

            full_text = ""
            for i in range(pages_to_read):
                page = reader.pages[i]
                text = page.extract_text()
                full_text += text + "\n"

            # 提取元数据
            metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    if value:
                        metadata[key.strip("/")] = str(value)

            result = {
                "full_text": full_text,
                "metadata": metadata,
                "total_pages": total_pages,
                "pages_read": pages_to_read,
            }

            # 提取章节（简单启发式）
            if extract_sections:
                sections = self._extract_sections(full_text)
                result["sections"] = sections

            return {
                "success": True,
                "result": result,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"PDF parsing error: {str(e)}",
            }
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """启发式提取论文章节"""
        sections = {}
        
        # 常见章节标题模式
        patterns = {
            "abstract": r"(?i)(abstract|summary)",
            "introduction": r"(?i)(introduction|background)",
            "method": r"(?i)(method|approach|model|framework)",
            "results": r"(?i)(results?|findings|experiments?)",
            "discussion": r"(?i)(discussion|analysis)",
            "conclusion": r"(?i)(conclusion|summary|future work)",
            "references": r"(?i)(references?|bibliography)",
        }

        lines = text.split("\n")
        current_section = None
        section_content = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 检查是否是章节标题
            for section_name, pattern in patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # 保存前一章节
                    if current_section and section_content:
                        sections[current_section] = "\n".join(section_content)
                    current_section = section_name
                    section_content = []
                    break
            else:
                if current_section:
                    section_content.append(line)

        # 保存最后一个章节
        if current_section and section_content:
            sections[current_section] = "\n".join(section_content)

        return sections
