"""Persistent coordination primitives for multi-agent harness workflows."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return slug or uuid.uuid4().hex[:8]


@dataclass
class TaskRecord:
    """One durable task in the harness task board."""

    id: int
    subject: str
    description: str = ""
    status: str = "pending"
    blocked_by: list[int] = field(default_factory=list)
    owner: str = ""
    worktree: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskRecord:
        return cls(
            id=int(payload["id"]),
            subject=str(payload.get("subject", "")),
            description=str(payload.get("description", "")),
            status=str(payload.get("status", "pending")),
            blocked_by=[int(item) for item in payload.get("blocked_by", payload.get("blockedBy", []))],
            owner=str(payload.get("owner", "")),
            worktree=str(payload.get("worktree", "")),
            metadata=dict(payload.get("metadata", {})),
            created_at=str(payload.get("created_at", _now())),
            updated_at=str(payload.get("updated_at", _now())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskBoard:
    """File-backed task DAG with dependency clearing and ownership claims."""

    VALID_STATUSES = {"pending", "in_progress", "completed", "blocked", "failed"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[int] | None = None,
        owner: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        task = TaskRecord(
            id=self._next_id(),
            subject=subject,
            description=description,
            blocked_by=blocked_by or [],
            owner=owner,
            metadata=metadata or {},
        )
        self.save(task)
        return task

    def get(self, task_id: int) -> TaskRecord:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return TaskRecord.from_dict(_read_json(path, {}))

    def save(self, task: TaskRecord) -> None:
        task.updated_at = _now()
        _write_json(self._path(task.id), task.to_dict())

    def update(
        self,
        task_id: int,
        status: str | None = None,
        owner: str | None = None,
        worktree: str | None = None,
        add_blocked_by: list[int] | None = None,
        remove_blocked_by: list[int] | None = None,
    ) -> TaskRecord:
        task = self.get(task_id)
        if status:
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Invalid task status: {status}")
            task.status = status
        if owner is not None:
            task.owner = owner
        if worktree is not None:
            task.worktree = worktree
        if add_blocked_by:
            task.blocked_by = sorted(set(task.blocked_by + [int(item) for item in add_blocked_by]))
        if remove_blocked_by:
            removals = {int(item) for item in remove_blocked_by}
            task.blocked_by = [item for item in task.blocked_by if item not in removals]
        self.save(task)
        if status == "completed":
            self._clear_dependency(task_id)
        return task

    def claim_next(self, owner: str) -> TaskRecord | None:
        for task in self.ready_tasks():
            if task.owner:
                continue
            task.owner = owner
            task.status = "in_progress"
            self.save(task)
            return task
        return None

    def ready_tasks(self) -> list[TaskRecord]:
        return [task for task in self.list_all() if task.status == "pending" and not task.blocked_by]

    def list_all(self) -> list[TaskRecord]:
        tasks = [TaskRecord.from_dict(_read_json(path, {})) for path in sorted(self.root.glob("task_*.json"))]
        return sorted(tasks, key=lambda task: task.id)

    def summary(self) -> dict[str, Any]:
        tasks = self.list_all()
        return {
            "tasks": [task.to_dict() for task in tasks],
            "ready": [task.id for task in self.ready_tasks()],
            "counts": {status: sum(1 for task in tasks if task.status == status) for status in self.VALID_STATUSES},
        }

    def _next_id(self) -> int:
        ids = [int(path.stem.split("_")[1]) for path in self.root.glob("task_*.json") if "_" in path.stem]
        return max(ids, default=0) + 1

    def _path(self, task_id: int) -> Path:
        return self.root / f"task_{int(task_id)}.json"

    def _clear_dependency(self, completed_id: int) -> None:
        for task in self.list_all():
            if completed_id in task.blocked_by:
                task.blocked_by = [item for item in task.blocked_by if item != completed_id]
                self.save(task)


@dataclass
class TeamMessage:
    """Append-only team mailbox message."""

    sender: str
    recipient: str
    content: str
    message_type: str = "message"
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TeamMailbox:
    """JSONL mailbox bus for local team-style coordination."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "message",
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TeamMessage:
        message = TeamMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            request_id=request_id,
            metadata=metadata or {},
        )
        path = self._path(recipient)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        return message

    def read(self, recipient: str, drain: bool = False) -> list[dict[str, Any]]:
        path = self._path(recipient)
        if not path.exists():
            return []
        messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if drain:
            path.write_text("", encoding="utf-8")
        return messages

    def _path(self, recipient: str) -> Path:
        return self.root / f"{_slug(recipient)}.jsonl"


@dataclass
class ProtocolRequest:
    """Request-response FSM record for team protocols."""

    id: str
    request_type: str
    sender: str
    target: str
    payload: dict[str, Any]
    status: str = "pending"
    response: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProtocolStore:
    """Persistent request-response protocol store."""

    VALID_STATUSES = {"pending", "approved", "rejected"}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def create(self, request_type: str, sender: str, target: str, payload: dict[str, Any]) -> ProtocolRequest:
        requests = self._load()
        request = ProtocolRequest(
            id=uuid.uuid4().hex[:8],
            request_type=request_type,
            sender=sender,
            target=target,
            payload=payload,
        )
        requests[request.id] = request.to_dict()
        self._save(requests)
        return request

    def respond(self, request_id: str, approve: bool, responder: str, feedback: str = "") -> ProtocolRequest:
        requests = self._load()
        if request_id not in requests:
            raise ValueError(f"Request {request_id} not found")
        payload = requests[request_id]
        payload["status"] = "approved" if approve else "rejected"
        payload["response"] = {
            "approve": approve,
            "responder": responder,
            "feedback": feedback,
            "created_at": _now(),
        }
        payload["updated_at"] = _now()
        requests[request_id] = payload
        self._save(requests)
        return ProtocolRequest(**payload)

    def list_all(self) -> list[ProtocolRequest]:
        return [ProtocolRequest(**payload) for payload in self._load().values()]

    def _load(self) -> dict[str, dict[str, Any]]:
        return _read_json(self.path, {})

    def _save(self, requests: dict[str, dict[str, Any]]) -> None:
        _write_json(self.path, requests)


class WorktreeIndex:
    """Persistent task-to-worktree binding index.

    This class records isolation intent and lifecycle events. It does not run
    git commands; callers can choose whether to materialize a git worktree.
    """

    def __init__(self, root: str | Path, task_board: TaskBoard | None = None) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.events_path = self.root / "events.jsonl"
        self.task_board = task_board

    def bind(self, name: str, task_id: int | None = None, path: str | None = None) -> dict[str, Any]:
        index = self._load()
        slug = _slug(name)
        record = {
            "name": slug,
            "path": path or str(self.root / slug),
            "task_id": task_id,
            "status": "active",
            "updated_at": _now(),
        }
        index[slug] = record
        self._save(index)
        if task_id is not None and self.task_board is not None:
            self.task_board.update(task_id, status="in_progress", worktree=slug)
        self._event("worktree.bind", record)
        return record

    def mark(self, name: str, status: str, complete_task: bool = False) -> dict[str, Any]:
        index = self._load()
        slug = _slug(name)
        if slug not in index:
            raise ValueError(f"Worktree binding {name} not found")
        record = index[slug]
        record["status"] = status
        record["updated_at"] = _now()
        self._save(index)
        if complete_task and record.get("task_id") is not None and self.task_board is not None:
            self.task_board.update(int(record["task_id"]), status="completed", worktree="")
        self._event(f"worktree.{status}", record)
        return record

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._load().values())

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load(self) -> dict[str, dict[str, Any]]:
        return _read_json(self.index_path, {})

    def _save(self, index: dict[str, dict[str, Any]]) -> None:
        _write_json(self.index_path, index)

    def _event(self, event: str, record: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, "record": record, "created_at": _now()}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ContextCompactor:
    """Deterministic context compaction helpers."""

    def __init__(self, keep_recent: int = 3, max_observation_chars: int = 600) -> None:
        self.keep_recent = keep_recent
        self.max_observation_chars = max_observation_chars

    def compact_steps(self, steps: list[Any]) -> list[dict[str, Any]]:
        compacted = []
        cutoff = max(0, len(steps) - self.keep_recent)
        for index, step in enumerate(steps):
            payload = step.to_dict() if hasattr(step, "to_dict") else dict(step)
            observation = payload.get("observation")
            if index < cutoff and isinstance(observation, str) and len(observation) > self.max_observation_chars:
                action = payload.get("action") or {}
                skill_name = action.get("skill") if isinstance(action, dict) else "unknown"
                payload["observation"] = f"[compacted: previous observation from {skill_name}]"
            compacted.append(payload)
        return compacted


def default_coordination_root(cwd: str | Path | None = None) -> Path:
    """Return the default local coordination root for a workspace."""

    return Path(cwd or Path.cwd()) / ".ai-researcher"
