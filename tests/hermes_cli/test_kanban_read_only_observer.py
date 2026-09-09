from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_resume_observer as observer
from hermes_cli import kanban_resume_requests as rr


def _directory_snapshot(path: Path) -> dict:
    snapshot = {}
    for item in sorted(path.iterdir()):
        info = item.stat(follow_symlinks=False)
        snapshot[item.name] = (
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_size,
            info.st_mtime_ns,
            hashlib.sha256(item.read_bytes()).hexdigest()
            if item.is_file() and not item.name.endswith("-shm")
            else None,
        )
    return snapshot


def test_read_only_list_performs_no_database_writes(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="parent", assignee="default", initial_status="running"
        )
        child = kb.create_task(
            conn,
            title="child",
            assignee="default",
            parents=(parent,),
            initial_status="running",
        )
        kb.complete_task(conn, parent)
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
    db = kb.kanban_db_path()
    writer = sqlite3.connect(db)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("UPDATE tasks SET title='fresh from wal' WHERE id=?", (child,))
    writer.commit()
    assert (db.parent / (db.name + "-wal")).exists()
    assert (db.parent / (db.name + "-shm")).exists()
    before = _directory_snapshot(db.parent)

    parser = __import__("argparse").ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    cli.build_parser(subs)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    args = parser.parse_args(["kanban", "list", "--read-only", "--json"])
    assert cli.kanban_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    observed = next(item for item in payload if item["id"] == child)
    assert observed["status"] == "todo"
    assert observed["title"] == "fresh from wal"
    assert _directory_snapshot(db.parent) == before

    mutating_args = parser.parse_args(["kanban", "list", "--json"])
    assert cli.kanban_command(mutating_args) == 1
    assert "cannot mutate" in capsys.readouterr().err
    writer.close()


def test_read_only_connection_refuses_sql_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with kb.connect() as conn:
        kb.create_task(conn, title="x", initial_status="running")
    with kb.connect_read_only() as conn:
        try:
            conn.execute("UPDATE tasks SET title='bad'")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower() or "read-only" in str(exc).lower()
        else:
            raise AssertionError("read-only connection accepted a write")


def test_full_observer_reads_live_wal_without_board_directory_metadata_writes(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "candidate.txt").write_text("retained\n", encoding="utf-8")
    subprocess.run(["git", "add", "candidate.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repo, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="candidate",
            assignee="reviewer",
            workspace_kind="dir",
            workspace_path=str(repo),
            expected_workspace_sha=sha,
            initial_status="running",
        )
        conn.execute("UPDATE tasks SET branch_name='main' WHERE id=?", (task_id,))
        kb.block_task(
            conn, task_id, reason="iteration budget exhausted", kind="needs_input"
        )
    db = kb.kanban_db_path()
    writer = sqlite3.connect(db)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        "UPDATE tasks SET title='fresh observer state' WHERE id=?", (task_id,)
    )
    writer.commit()
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o730)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "board": "default",
            "db_path": str(db),
            "task_id": task_id,
            "workspace_path": str(repo),
            "branch": "main",
            "sha": sha,
            "candidate_fingerprint": rr.candidate_fingerprint(repo, sha),
            "state_file": str(tmp_path / "state.json"),
            "outbox_dir": str(outbox),
            "policy_index": 0,
            "producer_uid": os.getuid(),
        }),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    expected_version = rr.inspect_task_read_only(db, task_id)["state_version"]
    before = _directory_snapshot(db.parent)
    result = observer.run(manifest)
    assert result["event"] == "request_pending"
    assert observer.run(manifest) is None
    assert json.loads((outbox / "resume-0.json").read_text(encoding="utf-8")) == {
        "policy_index": 0,
        "state_version": expected_version,
    }
    assert _directory_snapshot(db.parent) == before
    writer.close()


def test_observer_is_transition_based_and_failure_is_alert(tmp_path):
    state_file = tmp_path / "state.json"
    blocked = {
        "task_id": "t_fixed",
        "status": "blocked",
        "block_kind": "needs_input",
        "state_version": 7,
    }
    first = observer.transition_output(state_file, blocked, request_result=None)
    second = observer.transition_output(state_file, blocked, request_result=None)
    rejected = observer.transition_output(
        state_file,
        blocked,
        request_result={
            "request_id": "r_x",
            "state": "rejected",
            "result_code": "stale_sha",
        },
    )
    same_rejected = observer.transition_output(
        state_file,
        blocked,
        request_result={
            "request_id": "r_x",
            "state": "rejected",
            "result_code": "stale_sha",
        },
    )
    assert first["event"] == "new_blocker"
    assert second is None
    assert rejected["event"] == "request_rejected"
    assert rejected["ok"] is False
    assert same_rejected is None
