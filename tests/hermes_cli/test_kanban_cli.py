"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------



def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_capabilities_json_reports_atomic_idempotency_schema_evidence(kanban_home):
    payload = json.loads(kc.run_slash("capabilities --json"))

    assert payload == {
        "schema": "hermes-kanban-capabilities/v6",
        "atomic_idempotency": {
            "supported": True,
            "diagnostic": "permanent tenant-scoped uniqueness and identity immutability verified",
            "scope": "tenant",
            "default_tenant": "null",
            "indexes": [
                {
                    "name": "idx_tasks_idempotency",
                    "unique": True,
                    "partial": True,
                    "columns": ["tenant", "idempotency_key"],
                    "predicate": "tenant IS NOT NULL AND idempotency_key IS NOT NULL",
                },
                {
                    "name": "idx_tasks_idempotency_default",
                    "unique": True,
                    "partial": True,
                    "columns": ["idempotency_key"],
                    "predicate": "tenant IS NULL AND idempotency_key IS NOT NULL",
                },
            ],
            "duplicate_keys": 0,
            "null_keys": "distinct",
            "archived_keys": "permanent",
            "deletion_protection": {
                "name": "trg_tasks_protect_idempotency_owner_delete",
                "timing": "before",
                "event": "delete",
                "table": "tasks",
                "predicate": "OLD.idempotency_key IS NOT NULL",
                "action": "abort",
                "message": "permanent idempotency owner cannot be deleted",
                "verified": True,
            },
            "identity_update_protection": {
                "name": "trg_tasks_protect_idempotency_owner_update",
                "timing": "before",
                "event": "update",
                "table": "tasks",
                "predicate": (
                    "((OLD.idempotency_key IS NOT NULL AND "
                    "(NEW.idempotency_key IS NOT OLD.idempotency_key OR "
                    "NEW.tenant IS NOT OLD.tenant)) OR "
                    "(NEW.idempotency_key IS NOT NULL AND EXISTS "
                    "(SELECT 1 FROM tasks WHERE tasks.id IS NOT NEW.id AND "
                    "tasks.tenant IS NEW.tenant AND "
                    "tasks.idempotency_key IS NEW.idempotency_key)))"
                ),
                "action": "abort",
                "message": "keyed task has immutable idempotency identity",
                "verified": True,
            },
            "identity_reuse_protection": {
                "name": "trg_tasks_protect_idempotency_owner_replacement",
                "timing": "before",
                "event": "insert",
                "table": "tasks",
                "predicate": (
                    "(EXISTS (SELECT 1 FROM tasks WHERE tasks.id IS NEW.id AND "
                    "tasks.idempotency_key IS NOT NULL) OR "
                    "(NEW.idempotency_key IS NOT NULL AND EXISTS "
                    "(SELECT 1 FROM tasks WHERE tasks.tenant IS NEW.tenant AND "
                    "tasks.idempotency_key IS NEW.idempotency_key)))"
                ),
                "action": "abort",
                "message": "idempotency identity already has a permanent owner",
                "verified": True,
            },
            "attestation": "single-read-transaction",
        },
    }


def test_capabilities_probe_does_not_create_missing_storage(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    payload = json.loads(kc.run_slash("capabilities --json"))

    assert payload["schema"] == "hermes-kanban-capabilities/v6"
    assert payload["atomic_idempotency"]["supported"] is False
    assert "storage unavailable" in payload["atomic_idempotency"]["diagnostic"]
    assert not kb.kanban_db_path().exists()


def test_capabilities_probe_does_not_migrate_existing_legacy_storage(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tasks (id TEXT, idempotency_key TEXT, status TEXT)")
    conn.execute("CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)")
    conn.commit()
    conn.close()
    before = db_path.read_bytes()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    payload = json.loads(kc.run_slash("capabilities --json"))

    assert payload["atomic_idempotency"]["supported"] is False
    assert db_path.read_bytes() == before
    assert not Path(f"{db_path}-wal").exists()
    assert not Path(f"{db_path}-shm").exists()


def test_archive_rm_help_warns_keyed_tombstones_cannot_be_removed(capsys):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["kanban", "archive", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "without idempotency keys" in help_text
    assert "cannot be removed" in help_text


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kb.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kb.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


