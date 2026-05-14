---
name: custom-skill-name
description: Describe what this skill does and when an agent should use it.
---

# Custom Skill Template

Use this template to document a custom skill before or alongside its Python implementation.

## When To Use

Describe the tasks where the agent should choose this skill.

Example:

> Use this skill when the user asks to search a local bibliography by keyword, author, or paper title.

## Parameters

| Name | Type | Required | Default | Description |
|---|---|---:|---|---|
| `query` | string | yes | | Search query or task input. |
| `top_k` | integer | no | `5` | Maximum number of results to return. |

## Output

Skills should return a structured result:

```python
{"success": True, "result": {...}, "error": None}
{"success": False, "result": None, "error": "Describe the failure"}
```

## Constraints

- Do not call an LLM from inside the skill.
- Do not mutate global state.
- Do not read secrets from files directly; use configuration passed by the harness.
- Keep network and filesystem behavior explicit and testable.

## Python Skeleton

```python
from ai_researcher_assistant.skills import BaseSkill, SkillManifest, SkillParameter


class CustomSkill(BaseSkill):
    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="custom_skill_name",
            description="One short sentence describing the skill.",
            parameters=[
                SkillParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Search query or task input.",
                ),
                SkillParameter(
                    name="top_k",
                    type="integer",
                    required=False,
                    default=5,
                    description="Maximum number of results to return.",
                ),
            ],
        )

    def execute(self, parameters, context):
        query = parameters["query"]
        top_k = parameters.get("top_k", 5)

        return {
            "success": True,
            "result": {"query": query, "top_k": top_k, "items": []},
            "error": None,
        }
```

## Implementation Location

Built-in examples currently live in:

```text
ai_researcher_assistant/skills/builtin/
```

The legacy `skills/buildin/` package is still available as a compatibility alias.
