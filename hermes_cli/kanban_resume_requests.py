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
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from hermes_cli import kanban_db as kb

PRODUCER_KIND = "host-no-agent"
SUPPORTED_ACTION = "resume_iteration_budget"
SUPPORTED_BLOCK_KIND = "needs_input"


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


@dataclass(frozen=True)
class ResumeRequest:
    request_id: str
    state: str
    result_code: Optional[str] = None
    detail: Optional[str] = None


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


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, timeout=30
    )


def candidate_fingerprint(workspace: Path, expected_sha: str) -> str:
    """Hash HEAD, branch, tracked diff, and bounded untracked bytes deterministically."""
    repo = workspace.resolve()
    head = _git(repo, "rev-parse", "HEAD").strip().decode("ascii")
    if head != expected_sha:
        raise ValueError("workspace HEAD does not match expected SHA")
    branch = _git(repo, "symbolic-ref", "--short", "HEAD").strip()
    diff = _git(repo, "diff", "--binary", "--full-index", expected_sha, "--")
    untracked = [
        p
        for p in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if p
    ]
    digest = hashlib.sha256()
    for label, payload in (
        (b"head", head.encode()),
        (b"branch", branch),
        (b"diff", diff),
    ):
        digest.update(len(label).to_bytes(4, "big") + label)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    total = 0
    for raw in sorted(untracked):
        rel = raw.decode("utf-8", "surrogateescape")
        path = repo / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate contains unsupported untracked object")
        data = path.read_bytes()
        total += len(data)
        if total > 32 * 1024 * 1024:
            raise ValueError("untracked candidate exceeds 32 MiB fingerprint budget")
        digest.update(len(raw).to_bytes(4, "big") + raw)
        digest.update(len(data).to_bytes(8, "big") + data)
    return "sha256:" + digest.hexdigest()


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


def consume_resume_requests(
    conn: sqlite3.Connection,
    *,
    board: str,
    gateway_profile: str,
    policies: Iterable[ResumePolicy],
    lease_seconds: int = 60,
) -> list[ResumeRequest]:
    """Lease, validate and atomically accept/reject pending requests.

    The accepted transition and terminal request state share one IMMEDIATE
    transaction. A process death before commit rolls both back; after commit a
    restart sees a ready task and terminal request, so normal dispatch resumes it once.
    """
    if (
        _is_delegated_child()
        or gateway_profile != "default"
        or active_profile_name() != "default"
    ):
        raise PermissionError(
            "only the mutation-authorized default gateway may consume resume requests"
        )
    now = int(time.time())
    owner = f"default:{os.getpid()}"
    results: list[ResumeRequest] = []
    policy_list = tuple(policies)
    with kb.write_txn(conn):
        rows = conn.execute(
            "SELECT * FROM kanban_resume_requests WHERE state='pending' "
            "OR (state='leased' AND lease_expires < ?) ORDER BY created_at, request_id",
            (now,),
        ).fetchall()
        for initial in rows:
            cur = conn.execute(
                "UPDATE kanban_resume_requests SET state='leased', lease_owner=?, "
                "lease_expires=?, fence=fence+1 WHERE request_id=? AND "
                "(state='pending' OR (state='leased' AND lease_expires < ?))",
                (owner, now + max(1, int(lease_seconds)), initial["request_id"], now),
            )
            if cur.rowcount != 1:
                continue
            row = conn.execute(
                "SELECT * FROM kanban_resume_requests WHERE request_id=?",
                (initial["request_id"],),
            ).fetchone()
            policy = _matching_policy(row, policy_list)
            task = conn.execute(
                "SELECT * FROM tasks WHERE id=?", (row["task_id"],)
            ).fetchone()
            if policy is None:
                results.append(
                    _reject(
                        conn,
                        row,
                        "policy_mismatch",
                        "request is not in the fixed gateway allowlist",
                        now,
                    )
                )
                continue
            if task is None:
                results.append(
                    _reject(conn, row, "missing_task", "task no longer exists", now)
                )
                continue
            run = conn.execute(
                "SELECT summary FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (row["task_id"],),
            ).fetchone()
            reason = str(run["summary"] or "") if run is not None else ""
            actual_reason_hash = (
                "sha256:" + hashlib.sha256(reason.encode("utf-8")).hexdigest()
            )
            checks = (
                (
                    row["board_slug"] == board,
                    "board_mismatch",
                    "request belongs to a different board",
                ),
                (
                    row["action"] == SUPPORTED_ACTION,
                    "unsupported_action",
                    "action is not recoverable",
                ),
                (
                    row["expected_block_kind"] == SUPPORTED_BLOCK_KIND
                    and task["block_kind"] == SUPPORTED_BLOCK_KIND,
                    "unsupported_block_kind",
                    "block kind requires human handling",
                ),
                (
                    actual_reason_hash == row["expected_block_reason_sha256"],
                    "stale_block_reason",
                    "block reason changed",
                ),
                (
                    task["status"] == row["expected_status"] == "blocked",
                    "stale_status",
                    "task status changed",
                ),
                (
                    str(Path(task["workspace_path"] or "").resolve())
                    == row["expected_workspace_path"],
                    "stale_path",
                    "workspace path changed",
                ),
                (
                    task["branch_name"] == row["expected_branch"],
                    "stale_branch",
                    "task branch changed",
                ),
                (
                    task["expected_workspace_sha"] == row["expected_sha"],
                    "stale_sha",
                    "expected SHA changed",
                ),
                (
                    task["claim_lock"] is None
                    and task["worker_pid"] is None
                    and task["current_run_id"] is None,
                    "active_worker",
                    "task has a live claim or run",
                ),
            )
            failed = next(
                ((code, detail) for ok, code, detail in checks if not ok), None
            )
            version = conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM task_events WHERE task_id=?",
                (row["task_id"],),
            ).fetchone()[0]
            if failed is None and int(version) != int(row["expected_state_version"]):
                failed = ("stale_version", "task event version changed")
            if failed is None:
                try:
                    branch = (
                        _git(
                            Path(row["expected_workspace_path"]),
                            "symbolic-ref",
                            "--short",
                            "HEAD",
                        )
                        .strip()
                        .decode()
                    )
                    actual_fp = candidate_fingerprint(
                        Path(row["expected_workspace_path"]), row["expected_sha"]
                    )
                except Exception as exc:
                    failed = ("candidate_unreadable", str(exc))
                else:
                    if branch != row["expected_branch"]:
                        failed = ("stale_branch", "workspace branch changed")
                    elif actual_fp != row["expected_candidate_fingerprint"]:
                        failed = ("stale_fingerprint", "candidate bytes changed")
            if failed is not None:
                results.append(_reject(conn, row, failed[0], failed[1], now))
                continue
            updated = conn.execute(
                "UPDATE tasks SET status='ready' WHERE id=? AND status='blocked' "
                "AND block_kind=? AND claim_lock IS NULL AND worker_pid IS NULL AND current_run_id IS NULL",
                (row["task_id"], SUPPORTED_BLOCK_KIND),
            )
            if updated.rowcount != 1:
                results.append(
                    _reject(conn, row, "cas_failed", "task changed during consume", now)
                )
                continue
            kb._append_event(
                conn,
                row["task_id"],
                "resume_request_accepted",
                {"request_id": row["request_id"], "action": row["action"]},
            )
            conn.execute(
                "UPDATE kanban_resume_requests SET state='accepted', result_code='accepted', "
                "detail='validated; task made ready for default dispatcher', finished_at=?, "
                "lease_owner=NULL, lease_expires=NULL WHERE request_id=? AND state='leased' AND lease_owner=?",
                (now, row["request_id"], owner),
            )
            results.append(
                ResumeRequest(row["request_id"], "accepted", "accepted", "validated")
            )
    return results


def revalidate_accepted_request_before_dispatch(
    conn: sqlite3.Connection, task_id: str
) -> bool:
    """Recheck an accepted candidate immediately before the ordinary claim.

    Returns ``True`` for tasks unrelated to resume requests. On a stale accepted
    request, atomically re-blocks the still-unclaimed task and records a failed
    terminal outcome so no worker observes unreviewed bytes.
    """
    row = conn.execute(
        "SELECT * FROM kanban_resume_requests WHERE task_id=? AND state='accepted' "
        "ORDER BY finished_at DESC, request_id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return True
    failure: Optional[tuple[str, str]] = None
    task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task is None or task["status"] != "ready" or task["claim_lock"] is not None:
        return False
    try:
        branch = (
            _git(
                Path(row["expected_workspace_path"]), "symbolic-ref", "--short", "HEAD"
            )
            .strip()
            .decode()
        )
        actual = candidate_fingerprint(
            Path(row["expected_workspace_path"]), row["expected_sha"]
        )
    except Exception as exc:
        failure = ("dispatch_candidate_unreadable", str(exc))
    else:
        if branch != row["expected_branch"]:
            failure = (
                "dispatch_stale_branch",
                "workspace branch changed before dispatch",
            )
        elif actual != row["expected_candidate_fingerprint"]:
            failure = (
                "dispatch_stale_fingerprint",
                "candidate bytes changed before dispatch",
            )
    if failure is None:
        return True
    now = int(time.time())
    with kb.write_txn(conn):
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
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out
