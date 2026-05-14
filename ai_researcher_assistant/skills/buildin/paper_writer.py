"""Academic writing tool.

This skill does not call an LLM. It builds a structured writing request that
the harness loop can send to the model. That keeps model decisions centralized
in the agent loop.
"""

from typing import Any

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class PaperWriterSkill(BaseSkill):
    """Prepare academic writing prompts for the harness."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="paper_writer",
            description="Prepare academic writing, polishing, summarization, LaTeX, and abstract-generation prompts.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="action",
                    description="Action: polish, summarize, expand, latex_format, or generate_abstract",
                    type="string",
                    required=True,
                ),
                SkillParameter(
                    name="text",
                    description="Input text to process",
                    type="string",
                    required=True,
                ),
                SkillParameter(
                    name="style",
                    description="Writing style: formal, concise, or elaborate",
                    type="string",
                    required=False,
                    default="formal",
                ),
                SkillParameter(
                    name="field",
                    description="Academic field, for example physics or machine learning",
                    type="string",
                    required=False,
                    default="physics",
                ),
                SkillParameter(
                    name="additional_instructions",
                    description="Extra instructions for the model",
                    type="string",
                    required=False,
                    default="",
                ),
            ],
            tags=["writing", "academic", "polish", "latex"],
            instructions="""
Use this skill for academic writing tasks. The skill returns a model-ready prompt
and metadata; the harness must send the prompt to the LLM and synthesize the
final answer.
            """,
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        action = parameters["action"]
        text = parameters["text"]
        style = parameters.get("style", "formal")
        field = parameters.get("field", "physics")
        additional = parameters.get("additional_instructions", "")

        prompts = {
            "polish": f"""You are an expert academic editor in the field of {field}.
Polish the following text to improve clarity, grammar, and academic tone while preserving the original meaning and technical accuracy.
Use a {style} style. Do not add new content or change the scientific claims.

Text to polish:
---
{text}
---

Polished version:""",
            "summarize": f"""You are an expert in {field}.
Provide a concise summary of the following academic text. Capture the main contributions, methods, and conclusions.
Keep the summary under 250 words.

Text:
---
{text}
---

Summary:""",
            "expand": f"""You are an expert academic writer in {field}.
Expand the following brief idea or outline into a well-structured paragraph suitable for a research paper.
Use a {style} tone and include appropriate academic phrasing.

Idea:
---
{text}
---

Expanded paragraph:""",
            "latex_format": f"""You are an expert in LaTeX typesetting for {field} papers.
Convert the following text into proper LaTeX format. Ensure mathematical expressions, citations, and sections are formatted appropriately.

Text:
---
{text}
---

LaTeX output:""",
            "generate_abstract": f"""You are writing an abstract for a research paper in {field}.
Based on the following content, write a concise abstract of 150-250 words summarizing the problem, methods, key results, and implications.
Use a {style} academic style.

Content:
---
{text}
---

Abstract:""",
        }

        if action not in prompts:
            return {
                "success": False,
                "result": None,
                "error": f"Unknown action: {action}. Valid actions: {list(prompts.keys())}",
            }

        prompt = prompts[action]
        if additional:
            prompt = f"{prompt}\n\nAdditional instructions: {additional}"

        return {
            "success": True,
            "result": {
                "action": action,
                "style": style,
                "field": field,
                "original": text,
                "prompt": prompt,
                "requires_llm": True,
            },
            "error": None,
        }
