from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_resume_requests as rr


def _make_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "candidate.txt").write_text("retained\n", encoding="utf-8")
    subprocess.run(["git", "add", "candidate.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=path, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _blocked_task(tmp_path: Path, monkeypatch, *, kind: str = "needs_input"):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    sha = _make_repo(repo)
    conn = kb.connect()
    task_id = kb.create_task(
        conn,
        title="League Table",
        assignee="reviewer",
        workspace_kind="dir",
        workspace_path=str(repo),
        expected_workspace_sha=sha,
        initial_status="running",
    )
    conn.execute("UPDATE tasks SET branch_name='main' WHERE id=?", (task_id,))
    kb.block_task(conn, task_id, reason="iteration budget exhausted", kind=kind)
    snapshot = rr.inspect_task_read_only(kb.kanban_db_path(), task_id)
    return conn, task_id, repo, sha, snapshot


def _request(snapshot, repo: Path, sha: str, task_id: str):
    return rr.ResumeRequestSpec(
        board="default",
        task_id=task_id,
        action="resume_iteration_budget",
        expected_status="blocked",
        expected_state_version=snapshot["state_version"],
        expected_workspace_path=str(repo.resolve()),
        expected_branch="main",
        expected_sha=sha,
        expected_candidate_fingerprint=rr.candidate_fingerprint(repo, sha),
        expected_block_kind="needs_input",
        expected_block_reason_sha256=snapshot["block_reason_sha256"],
    )


def _policy(spec):
    return rr.ResumePolicy(
        board="default",
        task_id=spec.task_id,
        action=spec.action,
        workspace_path=spec.expected_workspace_path,
        branch=spec.expected_branch,
        sha=spec.expected_sha,
        candidate_fingerprint=spec.expected_candidate_fingerprint,
        block_reason_sha256=spec.expected_block_reason_sha256,
    )


def _append_worker(db_path: str, spec, barrier, queue):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    os.environ[rr.TRUSTED_PRODUCER_ENV] = rr.TRUSTED_PRODUCER
    conn = kb.connect(Path(db_path))
    barrier.wait()
    try:
        queue.put(
            rr.append_resume_request(
                conn, spec, producer=rr.TRUSTED_PRODUCER
            ).request_id
        )
    finally:
        conn.close()


def _consume_then_exit(db_path: str, policy, committed, proceed):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    conn = kb.connect(Path(db_path))
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[policy]
    )
    committed.set()
    proceed.wait()
    os._exit(17)


def _consume_exit_before_commit(db_path: str, policy, reached, proceed):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    conn = kb.connect(Path(db_path))

    def stop_during_validation(*args):
        reached.set()
        proceed.wait()
        os._exit(23)

    rr.candidate_fingerprint = stop_during_validation
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[policy]
    )


def test_append_is_trusted_but_direct_child_mutation_remains_denied(
    tmp_path, monkeypatch
):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    with pytest.raises(PermissionError, match="cannot mutate"):
        kb.unblock_task(conn, task_id)
    with pytest.raises(PermissionError, match="trusted host producer"):
        rr.append_resume_request(conn, spec, producer="host-no-agent")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    request = rr.append_resume_request(conn, spec, producer="host-no-agent")
    assert request.state == "pending"
    conn.close()


def test_duplicate_requests_create_one_continuation_run(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    first = rr.append_resume_request(conn, spec, producer="host-no-agent")
    second = rr.append_resume_request(conn, spec, producer="host-no-agent")
    assert first.request_id == second.request_id

    accepted = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert [(r.request_id, r.state) for r in accepted] == [
        (first.request_id, "accepted")
    ]
    first_claim = kb.claim_task(conn, task_id)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    second_claim = kb.claim_task(conn, task_id)
    assert first_claim is not None
    assert second_claim is None
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        == 2
    )
    conn.close()


def test_multiprocess_duplicate_append_has_one_durable_identity(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    conn.close()
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(3)
    queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_append_worker,
            args=(str(kb.kanban_db_path()), spec, barrier, queue),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0
    ids = [queue.get(timeout=2), queue.get(timeout=2)]
    assert ids[0] == ids[1] == rr.request_identity(spec)
    with kb.connect() as check:
        assert (
            check.execute("SELECT COUNT(*) FROM kanban_resume_requests").fetchone()[0]
            == 1
        )


def test_crash_replay_before_and_after_atomic_consume(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    request = rr.append_resume_request(conn, spec, producer="host-no-agent")
    # Crash before consume: durable pending row remains replayable.
    conn.close()
    conn = kb.connect()
    out = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert out[0].state == "accepted"
    conn.close()
    # Crash after commit: restart sees accepted request and ready task, never re-applies.
    conn = kb.connect()
    assert (
        rr.consume_resume_requests(
            conn, board="default", gateway_profile="default", policies=[_policy(spec)]
        )
        == []
    )
    assert kb.get_task(conn, task_id).status == "ready"
    row = conn.execute(
        "SELECT state, fence FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(row) == ("accepted", 1)
    conn.close()


def test_gateway_process_exit_after_consume_commit_is_restart_safe(
    tmp_path, monkeypatch
):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    rr.append_resume_request(conn, spec, producer="host-no-agent")
    conn.close()
    ctx = multiprocessing.get_context("fork")
    committed = ctx.Event()
    proceed = ctx.Event()
    worker = ctx.Process(
        target=_consume_then_exit,
        args=(str(kb.kanban_db_path()), _policy(spec), committed, proceed),
    )
    worker.start()
    assert committed.wait(20)
    proceed.set()
    worker.join(20)
    assert worker.exitcode == 17
    with kb.connect() as restarted:
        assert (
            rr.consume_resume_requests(
                restarted,
                board="default",
                gateway_profile="default",
                policies=[_policy(spec)],
            )
            == []
        )
        assert kb.get_task(restarted, task_id).status == "ready"


def test_candidate_is_revalidated_immediately_before_dispatch(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    request = rr.append_resume_request(conn, spec, producer="host-no-agent")
    assert (
        rr.consume_resume_requests(
            conn, board="default", gateway_profile="default", policies=[_policy(spec)]
        )[0].state
        == "accepted"
    )
    (repo / "candidate.txt").write_text("changed after acceptance\n", encoding="utf-8")
    assert rr.revalidate_accepted_request_before_dispatch(conn, task_id) is False
    assert kb.get_task(conn, task_id).status == "blocked"
    row = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert row["state"] == "rejected"
    assert row["result_code"] == "dispatch_stale_fingerprint"
    conn.close()


def test_gateway_process_exit_before_consume_commit_rolls_back_for_replay(
    tmp_path, monkeypatch
):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    request = rr.append_resume_request(conn, spec, producer="host-no-agent")
    conn.close()
    ctx = multiprocessing.get_context("fork")
    reached = ctx.Event()
    proceed = ctx.Event()
    worker = ctx.Process(
        target=_consume_exit_before_commit,
        args=(str(kb.kanban_db_path()), _policy(spec), reached, proceed),
    )
    worker.start()
    assert reached.wait(20)
    proceed.set()
    worker.join(20)
    assert worker.exitcode == 23
    with kb.connect() as restarted:
        row = restarted.execute(
            "SELECT state, fence FROM kanban_resume_requests WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
        assert tuple(row) == ("pending", 0)
        assert kb.get_task(restarted, task_id).status == "blocked"
        assert (
            rr.consume_resume_requests(
                restarted,
                board="default",
                gateway_profile="default",
                policies=[_policy(spec)],
            )[0].state
            == "accepted"
        )


@pytest.mark.parametrize(
    "field",
    ["status", "version", "block_reason", "path", "branch", "sha", "fingerprint"],
)
def test_stale_expectations_reject_without_unblocking(tmp_path, monkeypatch, field):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    rr.append_resume_request(conn, spec, producer="host-no-agent")
    if field == "status":
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (task_id,))
    elif field == "version":
        kb.add_comment(conn, task_id, "operator", "state changed")
    elif field == "block_reason":
        conn.execute(
            "UPDATE task_runs SET summary='credential needed' WHERE task_id=?",
            (task_id,),
        )
    elif field == "path":
        conn.execute(
            "UPDATE tasks SET workspace_path=? WHERE id=?",
            (str(tmp_path / "other"), task_id),
        )
    elif field == "branch":
        conn.execute("UPDATE tasks SET branch_name='other' WHERE id=?", (task_id,))
    elif field == "sha":
        conn.execute(
            "UPDATE tasks SET expected_workspace_sha=? WHERE id=?", ("0" * 40, task_id)
        )
    else:
        (repo / "candidate.txt").write_text("changed\n", encoding="utf-8")
    result = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert result[0].state == "rejected"
    assert kb.get_task(conn, task_id).status != "ready"
    conn.close()


def test_active_worker_and_unsupported_block_are_untouched(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    rr.append_resume_request(conn, spec, producer="host-no-agent")
    conn.execute(
        "UPDATE tasks SET worker_pid=999, claim_lock='live' WHERE id=?", (task_id,)
    )
    result = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert result[0].state == "rejected"
    assert kb.get_task(conn, task_id).status == "blocked"
    conn.close()

    conn, task_id, repo, sha, snap = _blocked_task(
        tmp_path / "two", monkeypatch, kind="capability"
    )
    spec = _request(snap, repo, sha, task_id)
    spec = rr.ResumeRequestSpec(**{
        **spec.__dict__,
        "expected_block_kind": "capability",
    })
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    rr.append_resume_request(conn, spec, producer="host-no-agent")
    result = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert result[0].state == "rejected"
    assert kb.get_task(conn, task_id).status == "blocked"
    conn.close()


def test_only_actual_default_gateway_consumes(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    rr.append_resume_request(conn, spec, producer="host-no-agent")
    monkeypatch.setattr(rr, "active_profile_name", lambda: "reviewer")
    with pytest.raises(PermissionError, match="default gateway"):
        rr.consume_resume_requests(
            conn, board="default", gateway_profile="reviewer", policies=[_policy(spec)]
        )
    with pytest.raises(PermissionError, match="default gateway"):
        rr.consume_resume_requests(
            conn, board="default", gateway_profile="default", policies=[_policy(spec)]
        )
    monkeypatch.setattr(rr, "active_profile_name", lambda: "default")
    assert (
        rr.consume_resume_requests(
            conn, board="default", gateway_profile="default", policies=[_policy(spec)]
        )[0].state
        == "accepted"
    )
    conn.close()


def test_request_failure_is_recorded_not_success(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    monkeypatch.setenv(rr.TRUSTED_PRODUCER_ENV, "host-no-agent")
    request = rr.append_resume_request(conn, spec, producer="host-no-agent")
    bad_policy = rr.ResumePolicy(**{**_policy(spec).__dict__, "sha": "f" * 40})
    result = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[bad_policy]
    )
    assert result[0].state == "rejected"
    row = conn.execute(
        "SELECT state, result_code, finished_at FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert row["state"] == "rejected"
    assert row["result_code"] != "ok"
    assert row["finished_at"] is not None
    conn.close()
