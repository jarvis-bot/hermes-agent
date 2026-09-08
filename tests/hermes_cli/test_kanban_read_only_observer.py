from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_resume_observer as observer


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
    before = db.read_bytes()
    before_sidecars = {p.name: p.read_bytes() for p in db.parent.glob(db.name + "-*")}

    parser = __import__("argparse").ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    cli.build_parser(subs)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    args = parser.parse_args(["kanban", "list", "--read-only", "--json"])
    assert cli.kanban_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert next(item for item in payload if item["id"] == child)["status"] == "todo"
    assert db.read_bytes() == before
    assert {
        p.name: p.read_bytes() for p in db.parent.glob(db.name + "-*")
    } == before_sidecars

    mutating_args = parser.parse_args(["kanban", "list", "--json"])
    assert cli.kanban_command(mutating_args) == 1
    assert "cannot mutate" in capsys.readouterr().err


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
