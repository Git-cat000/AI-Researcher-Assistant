"""Coordination skill for durable task, team, protocol, and worktree state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_researcher_assistant.harness.coordination import (
    ProtocolStore,
    TaskBoard,
    TeamMailbox,
    WorktreeIndex,
    default_coordination_root,
)
from ai_researcher_assistant.skills.base import BaseSkill, SkillManifest, SkillParameter


class HarnessCoordinationSkill(BaseSkill):
    """Expose persistent coordination primitives as one deterministic harness skill."""

    def _build_manifest(self) -> SkillManifest:
        return SkillManifest(
            name="harness_coordination",
            description="Manage persistent task-board, team mailbox, request protocol, and worktree-binding state.",
            version="1.0.0",
            author="AI Researcher Assistant",
            parameters=[
                SkillParameter(
                    name="action",
                    description=(
                        "One of: task_create, task_update, task_get, task_list, task_claim, "
                        "mail_send, mail_read, protocol_create, protocol_respond, protocol_list, "
                        "worktree_bind, worktree_mark, worktree_list"
                    ),
                    type="string",
                    required=True,
                ),
                SkillParameter(
                    name="cwd",
                    description="Workspace root for .ai-researcher coordination state",
                    type="string",
                    required=False,
                    default=None,
                ),
                SkillParameter(name="task_id", description="Task id", type="integer", required=False, default=None),
                SkillParameter(name="subject", description="Task subject", type="string", required=False, default=None),
                SkillParameter(
                    name="description",
                    description="Task description or work notes",
                    type="string",
                    required=False,
                    default="",
                ),
                SkillParameter(name="status", description="Task or worktree status", type="string", required=False),
                SkillParameter(name="owner", description="Task owner or claiming agent", type="string", required=False),
                SkillParameter(
                    name="blocked_by",
                    description="Task ids blocking this task",
                    type="list",
                    required=False,
                    default=None,
                ),
                SkillParameter(name="sender", description="Message or request sender", type="string", required=False),
                SkillParameter(name="recipient", description="Message recipient", type="string", required=False),
                SkillParameter(name="content", description="Message content", type="string", required=False),
                SkillParameter(name="message_type", description="Mailbox message type", type="string", required=False),
                SkillParameter(name="drain", description="Drain mailbox after reading", type="boolean", required=False),
                SkillParameter(
                    name="request_type",
                    description="Protocol request type, such as plan_approval or shutdown",
                    type="string",
                    required=False,
                ),
                SkillParameter(name="request_id", description="Protocol request id", type="string", required=False),
                SkillParameter(
                    name="approve",
                    description="Whether a protocol response approves",
                    type="boolean",
                    required=False,
                ),
                SkillParameter(
                    name="feedback", description="Protocol response feedback", type="string", required=False
                ),
                SkillParameter(name="worktree", description="Worktree binding name", type="string", required=False),
                SkillParameter(name="path", description="Optional isolated path", type="string", required=False),
                SkillParameter(
                    name="complete_task",
                    description="Mark the bound task completed when marking a worktree",
                    type="boolean",
                    required=False,
                    default=False,
                ),
            ],
            tags=["harness", "tasks", "team", "protocol", "worktree"],
            instructions=(
                "Use this skill for durable coordination that should survive context compression. "
                "It stores local JSON/JSONL state under .ai-researcher and never calls an LLM."
            ),
        )

    def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        try:
            root = self._root(parameters, context)
            task_board = TaskBoard(root / "tasks")
            mailbox = TeamMailbox(root / "team" / "inbox")
            protocols = ProtocolStore(root / "team" / "protocols.json")
            worktrees = WorktreeIndex(root / "worktrees", task_board=task_board)
            result = self._dispatch(parameters, task_board, mailbox, protocols, worktrees)
            return {"success": True, "result": result, "error": None}
        except Exception as exc:
            return {"success": False, "result": None, "error": str(exc)}

    def _root(self, parameters: dict[str, Any], context: dict[str, Any]) -> Path:
        cwd = parameters.get("cwd") or context.get("cwd") or context.get("workspace_dir")
        return default_coordination_root(cwd)

    def _dispatch(
        self,
        parameters: dict[str, Any],
        task_board: TaskBoard,
        mailbox: TeamMailbox,
        protocols: ProtocolStore,
        worktrees: WorktreeIndex,
    ) -> dict[str, Any]:
        action = parameters["action"]
        if action == "task_create":
            task = task_board.create(
                subject=parameters["subject"],
                description=parameters.get("description", ""),
                blocked_by=self._int_list(parameters.get("blocked_by")),
                owner=parameters.get("owner") or "",
            )
            return {"task": task.to_dict(), "summary": task_board.summary()}
        if action == "task_update":
            task = task_board.update(
                task_id=int(parameters["task_id"]),
                status=parameters.get("status"),
                owner=parameters.get("owner"),
                add_blocked_by=self._int_list(parameters.get("blocked_by")),
            )
            return {"task": task.to_dict(), "summary": task_board.summary()}
        if action == "task_get":
            return {"task": task_board.get(int(parameters["task_id"])).to_dict()}
        if action == "task_list":
            return task_board.summary()
        if action == "task_claim":
            claimed = task_board.claim_next(parameters.get("owner") or "agent")
            return {"task": claimed.to_dict() if claimed else None, "summary": task_board.summary()}
        if action == "mail_send":
            message = mailbox.send(
                sender=parameters.get("sender") or "agent",
                recipient=parameters["recipient"],
                content=parameters["content"],
                message_type=parameters.get("message_type", "message"),
                request_id=parameters.get("request_id"),
            )
            return {"message": message.to_dict()}
        if action == "mail_read":
            return {
                "messages": mailbox.read(
                    recipient=parameters.get("recipient") or parameters.get("owner") or "agent",
                    drain=bool(parameters.get("drain", False)),
                )
            }
        if action == "protocol_create":
            request = protocols.create(
                request_type=parameters["request_type"],
                sender=parameters.get("sender") or "agent",
                target=parameters["recipient"],
                payload={"content": parameters.get("content", ""), "task_id": parameters.get("task_id")},
            )
            mailbox.send(
                request.sender, request.target, parameters.get("content", ""), request.request_type, request.id
            )
            return {"request": request.to_dict()}
        if action == "protocol_respond":
            request = protocols.respond(
                request_id=parameters["request_id"],
                approve=bool(parameters.get("approve", False)),
                responder=parameters.get("sender") or "agent",
                feedback=parameters.get("feedback", ""),
            )
            mailbox.send(
                sender=parameters.get("sender") or "agent",
                recipient=request.sender,
                content=parameters.get("feedback", ""),
                message_type=f"{request.request_type}_response",
                request_id=request.id,
            )
            return {"request": request.to_dict()}
        if action == "protocol_list":
            return {"requests": [request.to_dict() for request in protocols.list_all()]}
        if action == "worktree_bind":
            return {
                "worktree": worktrees.bind(parameters["worktree"], parameters.get("task_id"), parameters.get("path"))
            }
        if action == "worktree_mark":
            return {
                "worktree": worktrees.mark(
                    parameters["worktree"],
                    parameters.get("status") or "kept",
                    complete_task=bool(parameters.get("complete_task", False)),
                )
            }
        if action == "worktree_list":
            return {"worktrees": worktrees.list_all(), "events": worktrees.events()}
        raise ValueError(f"Unknown coordination action: {action}")

    def _int_list(self, value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in value]
