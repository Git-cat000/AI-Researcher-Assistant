"""Parsing helpers for model outputs produced by harness loops."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_block(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract the first JSON object or array from plain text or a code block."""

    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = _loads_json(match.group(1))
            if parsed is not None:
                return parsed

    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            parsed = _loads_json(text[start : end + 1])
            if parsed is not None:
                return parsed

    return None


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract a ReAct action object with a `skill` key."""

    parsed = extract_json_block(text)
    if isinstance(parsed, dict) and "skill" in parsed:
        return parsed

    match = re.search(r"Action:\s*(\{[\s\S]*\})", text, re.IGNORECASE)
    if match:
        parsed = _loads_json(match.group(1))
        if isinstance(parsed, dict) and "skill" in parsed:
            return parsed

    return None


def extract_thought(text: str) -> str:
    """Extract the Thought section from a model response."""

    match = re.search(
        r"Thought:\s*(.+?)(?=\n\s*(?:Action:|Final Answer:)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text.strip()


def extract_final_answer(text: str) -> str | None:
    """Extract a final answer section from a model response."""

    match = re.search(r"Final Answer:\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _loads_json(raw: str) -> dict[str, Any] | list[Any] | None:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None
