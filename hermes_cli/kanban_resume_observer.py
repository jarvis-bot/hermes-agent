"""Deterministic no-agent observer for tightly allow-listed resume requests."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_resume_requests as rr


def transition_output(
    state_file: Path, task_state: dict, request_result: Optional[dict]
) -> Optional[dict]:
    """Return one transition event, or None when observable state is identical."""
    current = {
        "task_id": task_state.get("task_id"),
        "status": task_state.get("status"),
        "block_kind": task_state.get("block_kind"),
        "block_reason_sha256": task_state.get("block_reason_sha256"),
        "state_version": task_state.get("state_version"),
        "request": request_result,
    }
    identity = json.dumps(current, sort_keys=True, separators=(",", ":"))
    previous = {}
    try:
        previous = json.loads(state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    if previous.get("identity") == identity:
        return None
    state_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=state_file.name + ".", dir=state_file.parent)
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"identity": identity}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, state_file)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    if request_result:
        state = request_result.get("state")
        if state == "accepted" and task_state.get("status") == "running":
            return {"event": "continuation_claimed", "ok": True, **request_result}
        if state == "accepted":
            return {"event": "request_accepted", "ok": True, **request_result}
        if state == "rejected":
            return {"event": "request_rejected", "ok": False, **request_result}
        return {"event": "request_pending", "ok": False, **request_result}
    return {"event": "new_blocker", "ok": False, **task_state}


def _load_manifest(path: Path) -> dict:
    info = path.stat()
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError("observer manifest must not be group/world writable")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "board",
        "db_path",
        "task_id",
        "workspace_path",
        "branch",
        "sha",
        "candidate_fingerprint",
        "state_file",
    }
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(
            "observer manifest must contain exactly the fixed allowlist fields"
        )
    return data


def run(manifest_path: Path) -> Optional[dict]:
    manifest = _load_manifest(manifest_path.resolve())
    board = str(manifest["board"])
    if kb._normalize_board_slug(board) != board:
        raise ValueError("manifest board is not canonical")
    db_path = Path(str(manifest["db_path"])).resolve()
    workspace = Path(str(manifest["workspace_path"])).resolve()
    task = rr.inspect_task_read_only(db_path, str(manifest["task_id"]))
    result = rr.latest_resume_request_read_only(db_path, str(manifest["task_id"]))
    if task["status"] == "blocked" and task["block_kind"] == rr.SUPPORTED_BLOCK_KIND:
        spec = rr.ResumeRequestSpec(
            board=board,
            task_id=str(manifest["task_id"]),
            action=rr.SUPPORTED_ACTION,
            expected_status="blocked",
            expected_state_version=int(task["state_version"]),
            expected_workspace_path=str(workspace),
            expected_branch=str(manifest["branch"]),
            expected_sha=str(manifest["sha"]),
            expected_candidate_fingerprint=str(manifest["candidate_fingerprint"]),
            expected_block_kind=rr.SUPPORTED_BLOCK_KIND,
            expected_block_reason_sha256=str(task["block_reason_sha256"]),
        )
        os.environ[rr.TRUSTED_PRODUCER_ENV] = rr.TRUSTED_PRODUCER
        # Writable open occurs only after the read-only eligibility check. The
        # immutable manifest, not task text, supplies every action/path value.
        with kb.connect_closing(db_path) as conn:
            appended = rr.append_resume_request(
                conn, spec, producer=rr.TRUSTED_PRODUCER
            )
        result = {
            "request_id": appended.request_id,
            "state": appended.state,
            "result_code": appended.result_code,
        }
    return transition_output(Path(str(manifest["state_file"])).resolve(), task, result)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic Kanban resume observer")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = run(args.manifest)
    except Exception as exc:
        print(
            json.dumps(
                {"event": "observer_failure", "ok": False, "error": str(exc)[:500]},
                sort_keys=True,
            )
        )
        return 1
    if output is not None:
        print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
