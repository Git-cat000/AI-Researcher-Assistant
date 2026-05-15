"""Markdown skill compatibility for Claude Code and Codex SKILL.md files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class MarkdownSkill(BaseSkill):
    """A skill backed by a Markdown file with optional YAML frontmatter."""

    def __init__(self, markdown_path: str | Path):
        super().__init__()
        self.path = Path(markdown_path)
        self.root = self.path.parent
        self.frontmatter, self.body = parse_skill_markdown(self.path.read_text(encoding="utf-8"))

    def _build_manifest(self) -> SkillManifest:
        name = str(self.frontmatter.get("name") or self._fallback_name())
        description = str(self.frontmatter.get("description") or self._fallback_description())
        tags = self.frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        return SkillManifest(
            name=normalize_skill_name(name),
            description=description.strip(),
            version=str(self.frontmatter.get("version", "1.0.0")),
            author=str(self.frontmatter.get("author", "")),
            parameters=build_parameters_from_frontmatter(self.frontmatter),
            tags=list(tags) if isinstance(tags, list) else [],
            instructions=self.body.strip(),
            metadata={
                k: v
                for k, v in self.frontmatter.items()
                if k not in {"name", "description", "version", "author", "tags"}
            },
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        manifest = self.manifest
        requested_resources = parameters.get("read_resources") or parameters.get("resource_paths") or []
        if isinstance(requested_resources, str):
            requested_resources = [requested_resources]
        return {
            "success": True,
            "result": {
                "skill": manifest.name,
                "description": manifest.description,
                "instructions": manifest.instructions,
                "frontmatter": self.frontmatter,
                "parameters": parameters,
                "skill_file": str(self.path),
                "skill_root": str(self.root),
                "resources": self._discover_resources(),
                "resource_contents": self._read_requested_resources(list(requested_resources)),
                "script_policy": {
                    "can_execute": False,
                    "reason": (
                        "Markdown skills expose script paths only; the harness does not execute scripts by default."
                    ),
                },
                "requires_llm": True,
            },
            "error": None,
        }

    def _fallback_name(self) -> str:
        if self.path.name.upper() == "SKILL.MD":
            return self.root.name
        return self.path.stem

    def _fallback_description(self) -> str:
        heading = re.search(r"^#\s+(.+)$", self.body, re.MULTILINE)
        if heading:
            return heading.group(1).strip()
        first_line = next((line.strip() for line in self.body.splitlines() if line.strip()), "")
        return first_line[:200] or f"Markdown skill loaded from {self.path.name}"

    def _discover_resources(self) -> dict[str, list[dict[str, Any]]]:
        resources: dict[str, list[dict[str, Any]]] = {}
        for folder_name in ("references", "scripts", "assets"):
            folder = self.root / folder_name
            if folder.exists() and folder.is_dir():
                entries = []
                for path in folder.rglob("*"):
                    if not path.is_file():
                        continue
                    entries.append(
                        {
                            "path": str(path),
                            "name": path.name,
                            "size": path.stat().st_size,
                            "readable": folder_name in {"references", "assets"},
                            "executable": False,
                        }
                    )
                resources[folder_name] = entries
        return resources

    def _read_requested_resources(self, resource_paths: list[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        root = self.root.resolve()
        for requested in resource_paths:
            path = (self.root / requested).resolve()
            if not path.is_file() or not path.is_relative_to(root):
                continue
            if "scripts" in path.relative_to(root).parts:
                continue
            try:
                contents[str(path)] = path.read_text(encoding="utf-8")[:20000]
            except UnicodeDecodeError:
                continue
        return contents


def parse_skill_markdown(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-like frontmatter and return metadata plus Markdown body."""

    if not content.startswith("---"):
        return {}, content

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        return {}, content

    return parse_simple_yaml(match.group(1)), match.group(2)


def parse_simple_yaml(raw: str) -> dict[str, Any]:
    """Parse the small YAML subset commonly used in SKILL.md frontmatter."""

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        loaded = yaml.safe_load(raw) or {}
        return loaded if isinstance(loaded, dict) else {}

    data: dict[str, Any] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        key_match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if key_match:
            key, value = key_match.groups()
            current_key = key
            data[key] = _parse_yaml_scalar(value)
            continue

        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(_parse_yaml_scalar(stripped[2:]))
        elif current_key and isinstance(data.get(current_key), str):
            data[current_key] = f"{data[current_key]} {stripped}".strip()

    return data


def normalize_skill_name(name: str) -> str:
    """Normalize names to the portable kebab-case style used by SKILL.md tools."""

    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-_").lower()
    normalized = normalized.replace("_", "-")
    return normalized or "markdown-skill"


def build_parameters_from_frontmatter(frontmatter: dict[str, Any]) -> list[SkillParameter]:
    """Build parameters from common SKILL.md frontmatter schema shapes."""

    schema = (
        frontmatter.get("parameters")
        or frontmatter.get("params")
        or frontmatter.get("input_schema")
        or frontmatter.get("schema")
    )
    if not schema:
        return []

    if isinstance(schema, list):
        return [_parameter_from_spec(spec) for spec in schema if isinstance(spec, dict)]

    if isinstance(schema, dict) and "properties" in schema:
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            return [
                _parameter_from_spec({"name": name, "required": name in required, **spec})
                for name, spec in properties.items()
                if isinstance(spec, dict)
            ]

    if isinstance(schema, dict):
        parameters = []
        for name, spec in schema.items():
            if isinstance(spec, dict):
                parameters.append(_parameter_from_spec({"name": name, **spec}))
            else:
                parameters.append(SkillParameter(name=str(name), description=str(spec), required=False))
        return parameters

    return []


def _parameter_from_spec(spec: dict[str, Any]) -> SkillParameter:
    name = str(spec.get("name", "")).strip() or "input"
    return SkillParameter(
        name=name,
        description=str(spec.get("description", spec.get("title", name))),
        type=str(spec.get("type", "string")),
        required=bool(spec.get("required", False)),
        default=spec.get("default"),
    )


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return _strip_quotes(value)


def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value
