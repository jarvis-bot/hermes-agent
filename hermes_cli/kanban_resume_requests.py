"""Durable, narrowly-scoped supervisor requests consumed only by default gateway.

The request table is an outbox: observers may append one immutable request, while the
mutation-authorized default gateway revalidates authoritative task and workspace state
before changing a task.  Delegated children remain unable to append or consume.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

from hermes_cli import kanban_db as kb
from hermes_cli.workspace_digest import canonical_logical_workspace_digest

PRODUCER_KIND = "host-no-agent"
SUPPORTED_ACTION = "resume_iteration_budget"
SUPPORTED_BLOCK_KIND = "needs_input"
CANDIDATE_DIGEST_DEADLINE_SECONDS = 600.0
MAX_CANDIDATE_NODES = 50_000
MAX_CANDIDATE_FILES = 40_000
MAX_CANDIDATE_FILE_BYTES = 256 * 1024 * 1024
MAX_CANDIDATE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_CONSUME_BATCH = 8


@dataclass(frozen=True)
class ResumeRequestSpec:
    board: str
    task_id: str
    action: str
    expected_status: str
    expected_state_version: int
    expected_workspace_path: str
    expected_branch: str
    expected_sha: str
    expected_candidate_fingerprint: str
    expected_block_kind: str
    expected_block_reason_sha256: str
    expected_workspace_kind: str = "dir"


@dataclass(frozen=True)
class ResumePolicy:
    board: str
    task_id: str
    action: str
    workspace_path: str
    branch: str
    sha: str
    candidate_fingerprint: str
    block_reason_sha256: str
    workspace_kind: str = "dir"
    bind_legacy_metadata: bool = False


@dataclass(frozen=True)
class ResumeRequest:
    request_id: str
    state: str
    result_code: Optional[str] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class AcceptedDispatchBinding:
    task_id: str
    request_id: str
    accepted_event_version: int
    accepted_event_payload: str
    workspace_path: str
    workspace_kind: str
    branch: str
    sha: str
    candidate_fingerprint: str
    capability_identity: tuple[object, ...]


def _capability_identity(capability: object) -> tuple[object, ...]:
    """Freeze every prepared runtime/capability field consumed at spawn."""
    return tuple(
        getattr(capability, name, None)
        for name in (
            "available",
            "profile",
            "workspace",
            "reason",
            "read_only",
            "runtime_path",
            "device",
            "inode",
            "content_sha256",
            "reviewer_isolated",
        )
    )


def _canonical_spec(spec: ResumeRequestSpec) -> bytes:
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def request_identity(spec: ResumeRequestSpec) -> str:
    return (
        "rr_"
        + hashlib.sha256(
            b"hermes-kanban-resume-v1\0" + _canonical_spec(spec)
        ).hexdigest()[:32]
    )


def _request_row_matches_binding(
    row: sqlite3.Row, binding: AcceptedDispatchBinding
) -> bool:
    """Verify the complete immutable policy payload behind a request id."""
    try:
        spec = ResumeRequestSpec(
            board=str(row["board_slug"]),
            task_id=str(row["task_id"]),
            action=str(row["action"]),
            expected_status=str(row["expected_status"]),
            expected_state_version=int(row["expected_state_version"]),
            expected_workspace_path=str(row["expected_workspace_path"]),
            expected_branch=str(row["expected_branch"]),
            expected_sha=str(row["expected_sha"]),
            expected_candidate_fingerprint=str(row["expected_candidate_fingerprint"]),
            expected_block_kind=str(row["expected_block_kind"]),
            expected_block_reason_sha256=str(row["expected_block_reason_sha256"]),
            expected_workspace_kind=binding.workspace_kind,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        row["producer"] == PRODUCER_KIND
        and row["request_id"] == binding.request_id
        and spec.task_id == binding.task_id
        and request_identity(spec) == binding.request_id
    )


def active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _is_delegated_child() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return bool(is_delegated_child_process_context())
    except Exception:
        return bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))


def _request_from_row(row: sqlite3.Row) -> ResumeRequest:
    return ResumeRequest(
        request_id=str(row["request_id"]),
        state=str(row["state"]),
        result_code=row["result_code"],
        detail=row["detail"],
    )


def _append_verified_request(
    conn: sqlite3.Connection, spec: ResumeRequestSpec
) -> ResumeRequest:
    """Persist a request already authenticated by :func:`ingest_resume_outbox`."""
    if spec.action != SUPPORTED_ACTION:
        raise ValueError(f"unsupported resume action: {spec.action}")
    if spec.expected_workspace_kind != "dir":
        raise ValueError("resume requests support only authenticated dir workspaces")
    request_id = request_identity(spec)
    now = int(time.time())
    # Deliberately does not use kb.write_txn: this is not a task/board mutation,
    # but it has an even narrower producer guard above. Delegated children fail
    # before BEGIN and cannot turn this outbox into a mutation bypass.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_resume_requests (
                request_id, board_slug, task_id, action, producer,
                expected_status, expected_state_version, expected_workspace_path,
                expected_branch, expected_sha, expected_candidate_fingerprint,
                expected_block_kind, expected_block_reason_sha256, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                request_id,
                spec.board,
                spec.task_id,
                spec.action,
                PRODUCER_KIND,
                spec.expected_status,
                int(spec.expected_state_version),
                str(Path(spec.expected_workspace_path).resolve()),
                spec.expected_branch,
                spec.expected_sha,
                spec.expected_candidate_fingerprint,
                spec.expected_block_kind,
                spec.expected_block_reason_sha256,
                now,
            ),
        )
        row = conn.execute(
            "SELECT request_id, state, result_code, detail FROM kanban_resume_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return _request_from_row(row)


def _spec_from_policy(policy: ResumePolicy, state_version: int) -> ResumeRequestSpec:
    """Build identity only from gateway policy plus the observer's event fence."""
    return ResumeRequestSpec(
        board=policy.board,
        task_id=policy.task_id,
        action=policy.action,
        expected_status="blocked",
        expected_state_version=state_version,
        expected_workspace_path=policy.workspace_path,
        expected_branch=policy.branch,
        expected_sha=policy.sha,
        expected_candidate_fingerprint=policy.candidate_fingerprint,
        expected_block_kind=SUPPORTED_BLOCK_KIND,
        expected_block_reason_sha256=policy.block_reason_sha256,
        expected_workspace_kind=policy.workspace_kind,
    )


def ingest_resume_outbox(
    conn: sqlite3.Connection,
    *,
    board: str,
    outbox_dir: Path,
    producer_uid: int,
    policies: Iterable[ResumePolicy],
) -> list[ResumeRequest]:
    """Authenticate fixed requests by filesystem ownership and ingest them.

    The producer must be a dedicated OS identity, distinct from the gateway.
    Files contain a policy index and observed event fence; authority-bearing
    values come only from the gateway's operator-owned configuration.
    """
    if _is_delegated_child():
        raise PermissionError("delegated children cannot ingest resume requests")
    gateway_uid = os.getuid()
    producer_uid = int(producer_uid)
    if producer_uid < 0 or producer_uid == gateway_uid:
        raise PermissionError("resume outbox requires a distinct producer UID")
    configured_root = outbox_dir.expanduser()
    if configured_root.is_symlink():
        raise PermissionError("resume outbox must not be a symlink")
    root = configured_root.resolve(strict=True)
    root_info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid not in {
        0,
        gateway_uid,
        producer_uid,
    }:
        raise PermissionError("resume outbox has unsafe ownership")
    if root_info.st_mode & stat.S_IWOTH:
        raise PermissionError("resume outbox must not be world writable")
    policy_list = tuple(policies)
    results: list[ResumeRequest] = []
    for path in sorted(root.glob("*.json")):
        try:
            before = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != producer_uid
                or before.st_nlink != 1
                or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                continue
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                    raise PermissionError("resume request changed while opening")
                payload = os.read(fd, 4097)
            finally:
                os.close(fd)
            if len(payload) > 4096:
                raise ValueError("resume request exceeds 4 KiB")
            data = json.loads(payload)
            if not isinstance(data, dict) or set(data) != {
                "policy_index",
                "state_version",
            }:
                raise ValueError("resume request has unexpected fields")
            index = data["policy_index"]
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("policy_index must be an integer")
            state_version = data["state_version"]
            if (
                not isinstance(state_version, int)
                or isinstance(state_version, bool)
                or state_version < 0
            ):
                raise ValueError("state_version must be a non-negative integer")
            if index < 0 or index >= len(policy_list):
                raise ValueError("policy_index is outside the configured allowlist")
            policy = policy_list[index]
            if policy.board != board:
                raise ValueError("request policy belongs to another board")
            request = _append_verified_request(
                conn, _spec_from_policy(policy, state_version)
            )
            results.append(request)
            # Preserve the OS-authenticated witness across append→consume crashes.
            if request.state != "pending":
                path.unlink()
        except (IndexError, KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return results


def inspect_task_read_only(db_path: Path, task_id: str) -> dict:
    """Read authoritative task state without schema init, recompute, or writes."""
    with contextlib.closing(kb.connect_read_only(db_path)) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        version = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        run = conn.execute(
            "SELECT summary FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        reason = str(run["summary"] or "") if run is not None else ""
        return {
            "task_id": task_id,
            "status": row["status"],
            "block_kind": row["block_kind"],
            "workspace_path": row["workspace_path"],
            "branch": row["branch_name"],
            "expected_sha": row["expected_workspace_sha"],
            "claim_lock": row["claim_lock"],
            "worker_pid": row["worker_pid"],
            "current_run_id": row["current_run_id"],
            "state_version": int(version),
            "block_reason_sha256": "sha256:"
            + hashlib.sha256(reason.encode("utf-8")).hexdigest(),
        }


def latest_resume_request_read_only(db_path: Path, task_id: str) -> Optional[dict]:
    """Return the newest request outcome without opening the board writable."""
    with contextlib.closing(kb.connect_read_only(db_path)) as conn:
        try:
            row = conn.execute(
                "SELECT request_id, state, result_code, detail FROM "
                "kanban_resume_requests WHERE task_id=? ORDER BY created_at DESC, "
                "request_id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return None
            raise
        return dict(row) if row is not None else None


def candidate_fingerprint(
    workspace: Path, expected_sha: str, *, deadline: Optional[float] = None
) -> str:
    """Authenticate logical candidate bytes without interpreting candidate Git data.

    ``expected_sha`` is an identity supplied by trusted board/project policy.  It is
    validated here but deliberately never resolved through the candidate's ``.git``;
    the separately policy-pinned logical digest authenticates the actual bytes.
    """
    if not isinstance(expected_sha, str) or len(expected_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("expected workspace SHA must be 40 lowercase hex characters")
    return "sha256:" + canonical_logical_workspace_digest(
        workspace,
        deadline=(
            deadline
            if deadline is not None
            else time.monotonic() + CANDIDATE_DIGEST_DEADLINE_SECONDS
        ),
        max_nodes=MAX_CANDIDATE_NODES,
        max_files=MAX_CANDIDATE_FILES,
        max_file_bytes=MAX_CANDIDATE_FILE_BYTES,
        max_total_bytes=MAX_CANDIDATE_TOTAL_BYTES,
        allow_symlinks=True,
        require_single_link=False,
    )


def _matching_policy(
    row: sqlite3.Row, policies: Iterable[ResumePolicy]
) -> Optional[ResumePolicy]:
    for policy in policies:
        if (
            policy.board == row["board_slug"]
            and policy.task_id == row["task_id"]
            and policy.action == row["action"]
            and str(Path(policy.workspace_path).resolve())
            == row["expected_workspace_path"]
            and policy.branch == row["expected_branch"]
            and policy.sha == row["expected_sha"]
            and policy.candidate_fingerprint == row["expected_candidate_fingerprint"]
            and policy.block_reason_sha256 == row["expected_block_reason_sha256"]
        ):
            return policy
    return None


def _reject(
    conn: sqlite3.Connection, row: sqlite3.Row, code: str, detail: str, now: int
) -> ResumeRequest:
    conn.execute(
        "UPDATE kanban_resume_requests SET state='rejected', result_code=?, detail=?, "
        "finished_at=?, lease_owner=NULL, lease_expires=NULL WHERE request_id=? AND state='leased'",
        (code, detail[:500], now, row["request_id"]),
    )
    return ResumeRequest(row["request_id"], "rejected", code, detail[:500])


def _task_validation_snapshot(
    conn: sqlite3.Connection, task_id: str
) -> tuple[Optional[dict], int, str]:
    task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    version = int(
        conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()[0]
    )
    run = conn.execute(
        "SELECT summary FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    reason = str(run["summary"] or "") if run is not None else ""
    return (dict(task) if task is not None else None), version, reason


def _validation_failure(
    row: sqlite3.Row,
    task: Optional[dict],
    version: int,
    reason: str,
    *,
    board: str,
    policy: Optional[ResumePolicy],
) -> Optional[tuple[str, str]]:
    if policy is None:
        return "policy_mismatch", "request is not in the fixed gateway allowlist"
    if task is None:
        return "missing_task", "task no longer exists"
    reason_hash = "sha256:" + hashlib.sha256(reason.encode("utf-8")).hexdigest()
    metadata_matches = (
        task["branch_name"] == row["expected_branch"]
        and task["expected_workspace_sha"] == row["expected_sha"]
    )
    legacy_shape = (
        policy.bind_legacy_metadata
        and policy.workspace_kind == "dir"
        and task["workspace_kind"] == "dir"
        and task["branch_name"] is None
        and task["expected_workspace_sha"] is None
    )
    checks = (
        (row["board_slug"] == board, "board_mismatch", "request belongs to another board"),
        (row["action"] == SUPPORTED_ACTION, "unsupported_action", "action is not recoverable"),
        (
            row["expected_block_kind"] == SUPPORTED_BLOCK_KIND
            and task["block_kind"] == SUPPORTED_BLOCK_KIND,
            "unsupported_block_kind",
            "block kind requires human handling",
        ),
        (reason_hash == row["expected_block_reason_sha256"], "stale_block_reason", "block reason changed"),
        (task["status"] == row["expected_status"] == "blocked", "stale_status", "task status changed"),
        (
            str(Path(task["workspace_path"] or "").resolve()) == row["expected_workspace_path"],
            "stale_path",
            "workspace path changed",
        ),
        (task["workspace_kind"] == policy.workspace_kind, "stale_workspace_kind", "workspace kind changed"),
        (
            metadata_matches or legacy_shape,
            "stale_provenance",
            "task branch/SHA provenance changed",
        ),
        (
            task["claim_lock"] is None
            and task["worker_pid"] is None
            and task["current_run_id"] is None,
            "active_worker",
            "task has a live claim or run",
        ),
        (int(version) == int(row["expected_state_version"]), "stale_version", "task event version changed"),
    )
    return next(((code, detail) for ok, code, detail in checks if not ok), None)


def consume_resume_requests(
    conn: sqlite3.Connection,
    *,
    board: str,
    gateway_profile: str,
    policies: Iterable[ResumePolicy],
    lease_seconds: int = 610,
    batch_size: int = 8,
    validation_seconds: float = CANDIDATE_DIGEST_DEADLINE_SECONDS,
) -> list[ResumeRequest]:
    """Lease a bounded batch, authenticate outside the writer lock, then CAS."""
    if (
        _is_delegated_child()
        or gateway_profile != "default"
        or active_profile_name() != "default"
    ):
        raise PermissionError(
            "only the mutation-authorized default gateway may consume resume requests"
        )
    now = int(time.time())
    owner = f"default:{os.getpid()}:{time.monotonic_ns()}"
    policy_list = tuple(policies)
    leased: list[sqlite3.Row] = []
    limit = max(1, min(int(batch_size), MAX_CONSUME_BATCH))
    with kb.write_txn(conn):
        candidates = conn.execute(
            "SELECT request_id FROM kanban_resume_requests WHERE state='pending' "
            "OR (state='leased' AND lease_expires < ?) "
            "ORDER BY created_at, request_id LIMIT ?",
            (now, limit),
        ).fetchall()
        for candidate in candidates:
            updated = conn.execute(
                "UPDATE kanban_resume_requests SET state='leased', lease_owner=?, "
                "lease_expires=?, fence=fence+1 WHERE request_id=? AND "
                "(state='pending' OR (state='leased' AND lease_expires < ?))",
                (owner, now + max(1, int(lease_seconds)), candidate["request_id"], now),
            )
            if updated.rowcount == 1:
                leased.append(
                    conn.execute(
                        "SELECT * FROM kanban_resume_requests WHERE request_id=?",
                        (candidate["request_id"],),
                    ).fetchone()
                )

    validations = []
    deadline = time.monotonic() + max(0.001, float(validation_seconds))
    for row in leased:
        policy = _matching_policy(row, policy_list)
        task, version, reason = _task_validation_snapshot(conn, row["task_id"])
        failure = _validation_failure(
            row, task, version, reason, board=board, policy=policy
        )
        if failure is None:
            try:
                actual_fp = candidate_fingerprint(
                    Path(row["expected_workspace_path"]),
                    row["expected_sha"],
                    deadline=deadline,
                )
            except Exception as exc:
                failure = ("candidate_unreadable", str(exc))
            else:
                if actual_fp != row["expected_candidate_fingerprint"]:
                    failure = ("stale_fingerprint", "candidate bytes changed")
        validations.append((row, policy, task, version, reason, failure))

    results: list[ResumeRequest] = []
    for row, policy, task, version, reason, failure in validations:
        finished = int(time.time())
        with kb.write_txn(conn):
            lease = conn.execute(
                "SELECT * FROM kanban_resume_requests WHERE request_id=? "
                "AND state='leased' AND lease_owner=? AND fence=? AND lease_expires>=?",
                (row["request_id"], owner, row["fence"], finished),
            ).fetchone()
            if lease is None:
                continue
            current_task, current_version, current_reason = _task_validation_snapshot(
                conn, row["task_id"]
            )
            if (
                current_task != task
                or current_version != version
                or current_reason != reason
                or _matching_policy(lease, policy_list) != policy
            ):
                failure = (
                    "cas_failed",
                    "task, request, or policy changed during validation",
                )
            if failure is not None:
                results.append(_reject(conn, lease, failure[0], failure[1], finished))
                continue
            assert task is not None and policy is not None
            updated = conn.execute(
                "UPDATE tasks SET status='ready', branch_name=?, expected_workspace_sha=? "
                "WHERE id=? AND status='blocked' AND block_kind=? "
                "AND workspace_kind=? AND workspace_path=? AND branch_name IS ? "
                "AND expected_workspace_sha IS ? AND claim_lock IS NULL "
                "AND worker_pid IS NULL AND current_run_id IS NULL",
                (
                    row["expected_branch"],
                    row["expected_sha"],
                    row["task_id"],
                    SUPPORTED_BLOCK_KIND,
                    policy.workspace_kind,
                    task["workspace_path"],
                    task["branch_name"],
                    task["expected_workspace_sha"],
                ),
            )
            if updated.rowcount != 1:
                results.append(
                    _reject(conn, lease, "cas_failed", "task changed during consume", finished)
                )
                continue
            kb._append_event(
                conn,
                row["task_id"],
                "resume_request_accepted",
                {
                    "request_id": row["request_id"],
                    "action": row["action"],
                    "legacy_metadata_bound": bool(
                        task["branch_name"] is None
                        and task["expected_workspace_sha"] is None
                    ),
                },
            )
            accepted_version = int(
                conn.execute(
                    "SELECT MAX(id) FROM task_events WHERE task_id=?", (row["task_id"],)
                ).fetchone()[0]
            )
            cur = conn.execute(
                "UPDATE kanban_resume_requests SET state='accepted', result_code='accepted', "
                "detail=?, finished_at=?, lease_owner=NULL, lease_expires=NULL "
                "WHERE request_id=? AND state='leased' AND lease_owner=? AND fence=?",
                (
                    f"validated; accepted task event {accepted_version}",
                    finished,
                    row["request_id"],
                    owner,
                    row["fence"],
                ),
            )
            if cur.rowcount != 1:
                raise RuntimeError("resume request lease CAS failed after task transition")
            results.append(
                ResumeRequest(row["request_id"], "accepted", "accepted", "validated")
            )
    return results


def revalidate_accepted_request_before_dispatch(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    task_snapshot: object = None,
    workspace_capability: Any = None,
) -> bool | AcceptedDispatchBinding:
    """Authenticate and bind an accepted request to the exact prepared task."""
    row = conn.execute(
        "SELECT * FROM kanban_resume_requests WHERE task_id=? AND state='accepted' "
        "AND result_code='accepted' "
        "ORDER BY finished_at DESC, request_id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return True
    task_row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task_row is None or task_row["status"] != "ready" or task_row["claim_lock"] is not None:
        return False
    task = dict(task_row)
    event = conn.execute(
        "SELECT id, kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    failure: Optional[tuple[str, str]] = None
    expected_path = row["expected_workspace_path"]
    actual_path = str(Path(task["workspace_path"] or "").resolve())
    if actual_path != expected_path:
        failure = ("dispatch_stale_path", "task workspace path changed after acceptance")
    elif task["workspace_kind"] != "dir":
        failure = ("dispatch_stale_workspace_kind", "task workspace kind changed after acceptance")
    elif task["branch_name"] != row["expected_branch"]:
        failure = ("dispatch_stale_branch", "task branch changed after acceptance")
    elif task["expected_workspace_sha"] != row["expected_sha"]:
        failure = ("dispatch_stale_sha", "task expected SHA changed after acceptance")
    elif event is None or event["kind"] != "resume_request_accepted":
        failure = ("dispatch_stale_version", "task state changed after acceptance")
    else:
        try:
            payload = json.loads(event["payload"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if payload.get("request_id") != row["request_id"]:
            failure = ("dispatch_stale_version", "accepted request is not the latest task state")
    if failure is None and task_snapshot is not None:
        for attr, expected in (
            ("workspace_path", task["workspace_path"]),
            ("workspace_kind", task["workspace_kind"]),
            ("branch_name", task["branch_name"]),
            ("expected_workspace_sha", task["expected_workspace_sha"]),
            ("status", task["status"]),
        ):
            if getattr(task_snapshot, attr, None) != expected:
                failure = ("dispatch_stale_preflight", "prepared task snapshot changed")
                break
    if failure is None and workspace_capability is not None:
        from hermes_cli.kanban_workspace_preflight import workspace_capability_matches

        if not workspace_capability_matches(workspace_capability, expected_path):
            failure = ("dispatch_stale_capability", "prepared workspace capability changed")
        elif (
            getattr(workspace_capability, "content_sha256", None) is not None
            and "sha256:" + workspace_capability.content_sha256
            != row["expected_candidate_fingerprint"]
        ):
            failure = ("dispatch_stale_capability", "prepared capability authenticated different bytes")
    if failure is None:
        try:
            actual = candidate_fingerprint(Path(expected_path), row["expected_sha"])
        except Exception as exc:
            failure = ("dispatch_candidate_unreadable", str(exc))
        else:
            if actual != row["expected_candidate_fingerprint"]:
                failure = ("dispatch_stale_fingerprint", "candidate bytes changed before dispatch")
    if failure is None:
        assert event is not None
        return AcceptedDispatchBinding(
            task_id=task_id,
            request_id=row["request_id"],
            accepted_event_version=int(event["id"]),
            accepted_event_payload=str(event["payload"] or ""),
            workspace_path=expected_path,
            workspace_kind="dir",
            branch=row["expected_branch"],
            sha=row["expected_sha"],
            candidate_fingerprint=row["expected_candidate_fingerprint"],
            capability_identity=_capability_identity(workspace_capability),
        )

    now = int(time.time())
    with kb.write_txn(conn):
        current = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        current_request = conn.execute(
            "SELECT state FROM kanban_resume_requests WHERE request_id=?",
            (row["request_id"],),
        ).fetchone()
        if current is None or dict(current) != task or current_request is None or current_request["state"] != "accepted":
            return False
        updated = conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='needs_input' "
            "WHERE id=? AND status='ready' AND claim_lock IS NULL",
            (task_id,),
        )
        if updated.rowcount != 1:
            return False
        conn.execute(
            "UPDATE kanban_resume_requests SET state='rejected', result_code=?, "
            "detail=?, finished_at=? WHERE request_id=? AND state='accepted'",
            (failure[0], failure[1][:500], now, row["request_id"]),
        )
        kb._append_event(
            conn,
            task_id,
            "resume_request_dispatch_rejected",
            {"request_id": row["request_id"], "code": failure[0]},
        )
    return False


def validate_claimed_resume_dispatch(
    conn: sqlite3.Connection,
    task: object,
    binding: AcceptedDispatchBinding,
    workspace_capability: Any,
) -> bool:
    """CAS the complete accepted provenance after claim and before spawn."""
    failure: Optional[tuple[str, str]] = None
    try:
        from hermes_cli.kanban_workspace_preflight import workspace_capability_matches

        if _capability_identity(workspace_capability) != binding.capability_identity:
            failure = (
                "dispatch_stale_capability",
                "prepared workspace capability changed after claim",
            )
        elif not workspace_capability_matches(workspace_capability, binding.workspace_path):
            failure = (
                "dispatch_stale_capability",
                "workspace capability changed after claim",
            )
        elif (
            candidate_fingerprint(Path(binding.workspace_path), binding.sha)
            != binding.candidate_fingerprint
        ):
            failure = (
                "dispatch_stale_fingerprint",
                "candidate bytes changed after claim",
            )
    except Exception as exc:
        failure = ("dispatch_candidate_unreadable", str(exc))

    now = int(time.time())
    task_id = str(getattr(task, "id"))
    claim_lock = getattr(task, "claim_lock")
    run_id = getattr(task, "current_run_id")
    with kb.write_txn(conn):
        current = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        request = conn.execute(
            "SELECT * FROM kanban_resume_requests WHERE request_id=? AND task_id=?",
            (binding.request_id, task_id),
        ).fetchone()
        accepted_event = conn.execute(
            "SELECT id, kind, payload FROM task_events WHERE id=? AND task_id=?",
            (binding.accepted_event_version, task_id),
        ).fetchone()
        latest_event = conn.execute(
            "SELECT id, run_id, kind, payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        claimed_run = conn.execute(
            "SELECT task_id, profile, status, claim_lock, claim_expires, ended_at "
            "FROM task_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        try:
            accepted_payload = (
                json.loads(accepted_event["payload"] or "{}") if accepted_event else {}
            )
            latest_payload = (
                json.loads(latest_event["payload"] or "{}") if latest_event else {}
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            accepted_payload = latest_payload = {}

        if task_id != binding.task_id:
            failure = ("dispatch_stale_claim", "claimed task identity changed")
        elif (
            request is None
            or request["state"] != "accepted"
            or request["result_code"] != "accepted"
            or request["lease_owner"] is not None
            or request["lease_expires"] is not None
        ):
            failure = ("dispatch_stale_request", "accepted request state changed after claim")
        elif current is None or current["status"] != "running":
            failure = ("dispatch_stale_claim", "claimed task state changed after claim")
        elif (
            current["claim_lock"] != claim_lock
            or current["current_run_id"] != run_id
            or current["claim_expires"] != getattr(task, "claim_expires")
            or current["assignee"] != getattr(task, "assignee")
        ):
            failure = ("dispatch_stale_claim", "task claim or run changed after claim")
        elif (
            claimed_run is None
            or claimed_run["task_id"] != task_id
            or claimed_run["profile"] != getattr(task, "assignee")
            or claimed_run["status"] != "running"
            or claimed_run["claim_lock"] != claim_lock
            or claimed_run["claim_expires"] != getattr(task, "claim_expires")
            or claimed_run["ended_at"] is not None
        ):
            failure = ("dispatch_stale_claim", "claimed run record changed after claim")
        elif current["workspace_path"] != binding.workspace_path:
            failure = ("dispatch_stale_path", "task workspace path changed after claim")
        elif current["workspace_kind"] != binding.workspace_kind:
            failure = (
                "dispatch_stale_workspace_kind",
                "task workspace kind changed after claim",
            )
        elif current["branch_name"] != binding.branch:
            failure = ("dispatch_stale_branch", "task branch changed after claim")
        elif current["expected_workspace_sha"] != binding.sha:
            failure = ("dispatch_stale_sha", "task expected SHA changed after claim")
        elif (
            accepted_event is None
            or accepted_event["kind"] != "resume_request_accepted"
            or str(accepted_event["payload"] or "") != binding.accepted_event_payload
            or accepted_payload.get("request_id") != binding.request_id
            or latest_event is None
            or latest_event["kind"] != "claimed"
            or latest_event["run_id"] != run_id
            or latest_payload
            != {
                "lock": claim_lock,
                "expires": getattr(task, "claim_expires"),
                "run_id": run_id,
            }
        ):
            failure = ("dispatch_stale_version", "task event version changed after claim")
        elif not _request_row_matches_binding(request, binding):
            failure = ("dispatch_stale_request", "accepted policy changed after claim")

        if failure is None:
            return True

        # Release only the exact claim/run this dispatcher owns.  If another
        # winner replaced either fence, its task and run remain untouched.
        updated = conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='needs_input', "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, current_run_id=NULL "
            "WHERE id=? AND status='running' AND claim_lock IS ? AND current_run_id IS ?",
            (task_id, claim_lock, run_id),
        )
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET status='workspace_changed', outcome='workspace_changed', "
                "summary=?, ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND task_id=? AND claim_lock IS ? AND ended_at IS NULL",
                (failure[1][:500], now, run_id, task_id, claim_lock),
            )
        conn.execute(
            "UPDATE kanban_resume_requests SET state='rejected', result_code=?, detail=?, "
            "finished_at=? WHERE request_id=? AND state='accepted'",
            (failure[0], failure[1][:500], now, binding.request_id),
        )
        if updated.rowcount == 1:
            kb._append_event(
                conn,
                task_id,
                "resume_request_dispatch_rejected",
                {"request_id": binding.request_id, "code": failure[0]},
                run_id=run_id,
            )
    return False


def finalize_resume_dispatch_spawned(
    conn: sqlite3.Connection,
    binding: AcceptedDispatchBinding,
    task: object,
    workspace_capability: Any,
    spawn: Callable[[], Optional[int]],
) -> tuple[bool, Optional[int]]:
    """Atomically fence the final control-plane snapshot through process launch.

    The expensive candidate digest is computed by
    :func:`validate_claimed_resume_dispatch` before this transaction.  This final
    short write transaction re-reads every mutable database fence, records the
    durable launch intent, and holds SQLite's writer lock until ``spawn`` returns.
    Consequently no database provenance writer can fit between the final CAS and
    process creation; ``_default_spawn`` receives the already-bound immutable task
    and capability identities.
    """
    task_id = str(getattr(task, "id"))
    claim_lock = getattr(task, "claim_lock")
    run_id = getattr(task, "current_run_id")
    now = int(time.time())
    with kb.write_txn(conn):
        current = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        request = conn.execute(
            "SELECT * FROM kanban_resume_requests WHERE request_id=? AND task_id=?",
            (binding.request_id, task_id),
        ).fetchone()
        accepted_event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE id=? AND task_id=?",
            (binding.accepted_event_version, task_id),
        ).fetchone()
        latest_event = conn.execute(
            "SELECT run_id, kind, payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        claimed_run = conn.execute(
            "SELECT task_id, profile, status, claim_lock, claim_expires, ended_at "
            "FROM task_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        try:
            accepted_payload = (
                json.loads(accepted_event["payload"] or "{}") if accepted_event else {}
            )
            latest_payload = (
                json.loads(latest_event["payload"] or "{}") if latest_event else {}
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            accepted_payload = latest_payload = {}

        failure: Optional[tuple[str, str]] = None
        if task_id != binding.task_id:
            failure = ("dispatch_stale_claim", "claimed task identity changed")
        elif (
            request is None
            or request["state"] != "accepted"
            or request["result_code"] != "accepted"
            or request["lease_owner"] is not None
            or request["lease_expires"] is not None
        ):
            failure = ("dispatch_stale_request", "accepted request state changed before spawn")
        elif current is None or current["status"] != "running":
            failure = ("dispatch_stale_claim", "claimed task state changed before spawn")
        elif (
            current["claim_lock"] != claim_lock
            or current["current_run_id"] != run_id
            or current["claim_expires"] != getattr(task, "claim_expires")
            or current["assignee"] != getattr(task, "assignee")
        ):
            failure = ("dispatch_stale_claim", "task claim or run changed before spawn")
        elif (
            claimed_run is None
            or claimed_run["task_id"] != task_id
            or claimed_run["profile"] != getattr(task, "assignee")
            or claimed_run["status"] != "running"
            or claimed_run["claim_lock"] != claim_lock
            or claimed_run["claim_expires"] != getattr(task, "claim_expires")
            or claimed_run["ended_at"] is not None
        ):
            failure = ("dispatch_stale_claim", "claimed run record changed before spawn")
        elif current["workspace_path"] != binding.workspace_path:
            failure = ("dispatch_stale_path", "task workspace path changed before spawn")
        elif current["workspace_kind"] != binding.workspace_kind:
            failure = ("dispatch_stale_workspace_kind", "task workspace kind changed before spawn")
        elif current["branch_name"] != binding.branch:
            failure = ("dispatch_stale_branch", "task branch changed before spawn")
        elif current["expected_workspace_sha"] != binding.sha:
            failure = ("dispatch_stale_sha", "task expected SHA changed before spawn")
        elif (
            _capability_identity(workspace_capability) != binding.capability_identity
        ):
            failure = ("dispatch_stale_capability", "prepared capability changed before spawn")
        elif (
            accepted_event is None
            or accepted_event["kind"] != "resume_request_accepted"
            or str(accepted_event["payload"] or "") != binding.accepted_event_payload
            or accepted_payload.get("request_id") != binding.request_id
            or latest_event is None
            or latest_event["kind"] != "claimed"
            or latest_event["run_id"] != run_id
            or latest_payload
            != {
                "lock": claim_lock,
                "expires": getattr(task, "claim_expires"),
                "run_id": run_id,
            }
        ):
            failure = ("dispatch_stale_version", "task event version changed before spawn")
        elif not _request_row_matches_binding(request, binding):
            failure = ("dispatch_stale_request", "accepted policy changed before spawn")

        if failure is not None:
            updated = conn.execute(
                "UPDATE tasks SET status='blocked', block_kind='needs_input', "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, current_run_id=NULL "
                "WHERE id=? AND status='running' AND claim_lock IS ? "
                "AND current_run_id IS ?",
                (task_id, claim_lock, run_id),
            )
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET status='workspace_changed', "
                    "outcome='workspace_changed', summary=?, ended_at=?, "
                    "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL "
                    "WHERE id=? AND task_id=? AND claim_lock IS ? AND ended_at IS NULL",
                    (failure[1][:500], now, run_id, task_id, claim_lock),
                )
            conn.execute(
                "UPDATE kanban_resume_requests SET state='rejected', result_code=?, "
                "detail=?, finished_at=? WHERE request_id=? AND state='accepted'",
                (failure[0], failure[1][:500], now, binding.request_id),
            )
            if updated.rowcount == 1:
                kb._append_event(
                    conn,
                    task_id,
                    "resume_request_dispatch_rejected",
                    {"request_id": binding.request_id, "code": failure[0]},
                    run_id=run_id,
                )
            return False, None

        updated = conn.execute(
            "UPDATE kanban_resume_requests SET result_code='dispatched', "
            "detail='accepted candidate bound to dispatcher spawn attempt' "
            "WHERE request_id=? AND task_id=? AND state='accepted' "
            "AND result_code='accepted'",
            (binding.request_id, task_id),
        )
        if updated.rowcount != 1:
            raise RuntimeError("accepted resume request changed before spawn finalization")
        return True, spawn()


def rearm_resume_dispatch_after_spawn_failure(
    conn: sqlite3.Connection, task_id: str, binding: AcceptedDispatchBinding
) -> None:
    """Re-arm a pre-exec failure so a retry must authenticate the request again."""
    with kb.write_txn(conn):
        task = conn.execute(
            "SELECT status, claim_lock, current_run_id, workspace_path, workspace_kind, "
            "branch_name, expected_workspace_sha FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        request = conn.execute(
            "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
            (binding.request_id,),
        ).fetchone()
        if request is None or request["state"] != "accepted":
            return
        retryable = bool(
            task is not None
            and task["status"] == "ready"
            and task["claim_lock"] is None
            and task["current_run_id"] is None
            and task["workspace_path"] == binding.workspace_path
            and task["workspace_kind"] == binding.workspace_kind
            and task["branch_name"] == binding.branch
            and task["expected_workspace_sha"] == binding.sha
        )
        if retryable:
            conn.execute(
                "UPDATE kanban_resume_requests SET result_code='accepted', "
                "detail='pre-exec failure; candidate must be revalidated' "
                "WHERE request_id=? AND state='accepted'",
                (binding.request_id,),
            )
            kb._append_event(
                conn,
                task_id,
                "resume_request_accepted",
                {
                    "request_id": binding.request_id,
                    "action": SUPPORTED_ACTION,
                    "retry_after_spawn_failure": True,
                },
            )
        else:
            conn.execute(
                "UPDATE kanban_resume_requests SET state='rejected', "
                "result_code='dispatch_spawn_failure', "
                "detail='spawn failure was not safely retryable', finished_at=? "
                "WHERE request_id=? AND state='accepted'",
                (int(time.time()), binding.request_id),
            )


def policies_from_config(raw: object) -> list[ResumePolicy]:
    if not isinstance(raw, list):
        return []
    out: list[ResumePolicy] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item_map = cast(dict[str, Any], item)
        try:
            out.append(
                ResumePolicy(
                    board=str(item_map["board"]),
                    task_id=str(item_map["task_id"]),
                    action=str(item_map["action"]),
                    workspace_path=str(item_map["workspace_path"]),
                    branch=str(item_map["branch"]),
                    sha=str(item_map["sha"]),
                    candidate_fingerprint=str(item_map["candidate_fingerprint"]),
                    block_reason_sha256=str(item_map["block_reason_sha256"]),
                    workspace_kind=str(item_map.get("workspace_kind", "dir")),
                    bind_legacy_metadata=item_map.get("bind_legacy_metadata", False)
                    is True,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out
