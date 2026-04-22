"""
论文写作与润色技能。
支持学术写作辅助、润色、格式化。
"""
from typing import Dict, Any, List, Optional

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter
from ai_researcher_assistant.llm import get_llm
from ai_researcher_assistant.core.exceptions import SkillError


class PaperWriterSkill(BaseSkill):
    """学术论文写作与润色助手"""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="paper_writer",
            description="Assist with academic writing: polish text, generate LaTeX, format citations, etc.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="action",
                    description="Action to perform: 'polish', 'summarize', 'expand', 'latex_format', 'generate_abstract'",
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
                    description="Writing style: 'formal', 'concise', 'elaborate'",
                    type="string",
                    required=False,
                    default="formal",
                ),
                SkillParameter(
                    name="field",
                    description="Academic field, e.g., 'physics', 'theoretical physics'",
                    type="string",
                    required=False,
                    default="physics",
                ),
                SkillParameter(
                    name="additional_instructions",
                    description="Any extra instructions for the LLM",
                    type="string",
                    required=False,
                    default="",
                ),
            ],
            tags=["writing", "academic", "polish", "latex"],
            instructions="""
Use this skill for academic writing tasks:
- 'polish': Improve clarity, grammar, and academic tone.
- 'summarize': Condense a long text into a concise summary.
- 'expand': Elaborate on a brief idea into a full paragraph.
- 'latex_format': Convert plain text to proper LaTeX with math formatting.
- 'generate_abstract': Create an abstract from a full paper or outline.

Provide the text and specify the desired action. You can also set the writing style.
            """,
        )

    def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        action = parameters["action"]
        text = parameters["text"]
        style = parameters.get("style", "formal")
        field = parameters.get("field", "physics")
        additional = parameters.get("additional_instructions", "")

        # 获取 LLM 实例（从上下文或全局）
        llm = context.get("llm") or get_llm()

        # 根据 action 构建不同的提示词
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

            "latex_format": f"""You are an expert in LaTeX typesetting for physics papers.
Convert the following text into proper LaTeX format. Ensure mathematical expressions are enclosed in $...$ or $$...$$, 
citations are formatted as \\cite{{...}}, and sections use appropriate commands.

Text:
---
{text}
---

LaTeX output:""",

            "generate_abstract": f"""You are a physicist writing an abstract for a research paper.
Based on the following content, write a concise abstract (150-250 words) summarizing the problem, methods, key results, and implications. 
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
            prompt += f"\n\nAdditional instructions: {additional}"

        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm.generate(messages, temperature=0.3)  # 学术写作低温度
            return {
                "success": True,
                "result": {
                    "original": text,
                    "processed": response.content,
                    "action": action,
                    "style": style,
                },
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"LLM error: {str(e)}",
            }
