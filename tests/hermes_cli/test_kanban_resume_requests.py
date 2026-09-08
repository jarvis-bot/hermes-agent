from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_resume_requests as rr
from hermes_cli.kanban_workspace_preflight import WorkspaceCapability


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
        branch_name="main",
        expected_workspace_sha=sha,
        initial_status="running",
    )
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


def _policy(spec, *, bind_legacy_metadata: bool = False):
    return rr.ResumePolicy(
        board="default",
        task_id=spec.task_id,
        action=spec.action,
        workspace_path=spec.expected_workspace_path,
        branch=spec.expected_branch,
        sha=spec.expected_sha,
        candidate_fingerprint=spec.expected_candidate_fingerprint,
        block_reason_sha256=spec.expected_block_reason_sha256,
        bind_legacy_metadata=bind_legacy_metadata,
    )


def test_outbox_rejects_same_uid_and_forged_environment(tmp_path, monkeypatch):
    """Append authority is a distinct OS identity, never an env/caller flag."""
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o730)
    request_file = outbox / "request.json"
    request_file.write_text(
        json.dumps({"policy_index": 0, "state_version": snap["state_version"]}) + "\n",
        encoding="utf-8",
    )
    request_file.chmod(0o600)
    monkeypatch.setenv("HERMES_KANBAN_RESUME_PRODUCER", "host-no-agent")

    with pytest.raises(PermissionError, match="distinct producer UID"):
        rr.ingest_resume_outbox(
            conn,
            board="default",
            outbox_dir=outbox,
            producer_uid=os.getuid(),
            policies=[_policy(spec)],
        )
    assert (
        conn.execute("SELECT COUNT(*) FROM kanban_resume_requests").fetchone()[0] == 0
    )


def test_outbox_accepts_only_owned_fixed_policy_request(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o730)
    request_file = outbox / "request.json"
    request_file.write_text(
        json.dumps({"policy_index": 0, "state_version": snap["state_version"]}) + "\n",
        encoding="utf-8",
    )
    request_file.chmod(0o600)
    gateway_uid = os.getuid() + 1
    monkeypatch.setattr(rr.os, "getuid", lambda: gateway_uid)

    policy = _policy(spec)
    producer_uid = request_file.stat().st_uid
    requests = rr.ingest_resume_outbox(
        conn,
        board="default",
        outbox_dir=outbox,
        producer_uid=producer_uid,
        policies=[policy],
    )
    assert len(requests) == 1
    assert requests[0].state == "pending"
    assert request_file.exists()  # durable retry witness until terminal state
    conn.close()  # gateway dies after durable append, before consume
    conn = kb.connect()  # restarted gateway replays the same outbox witness
    replayed = rr.ingest_resume_outbox(
        conn,
        board="default",
        outbox_dir=outbox,
        producer_uid=producer_uid,
        policies=[policy],
    )
    assert len(replayed) == 1 and replayed[0].request_id == requests[0].request_id
    consumed = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[policy]
    )
    assert consumed[0].state == "accepted"
    assert (
        rr.ingest_resume_outbox(
            conn,
            board="default",
            outbox_dir=outbox,
            producer_uid=producer_uid,
            policies=[policy],
        )[0].state
        == "accepted"
    )
    assert not request_file.exists()
    conn.close()


def test_outbox_rejects_symlink_endpoint_and_negative_policy_index(
    tmp_path, monkeypatch
):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    policy = _policy(_request(snap, repo, sha, task_id))
    real = tmp_path / "real"
    real.mkdir(mode=0o730)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    request_file = real / "request.json"
    request_file.write_text(
        json.dumps({"policy_index": -1, "state_version": snap["state_version"]}) + "\n",
        encoding="utf-8",
    )
    request_file.chmod(0o600)
    gateway_uid = os.getuid() + 1
    monkeypatch.setattr(rr.os, "getuid", lambda: gateway_uid)
    with pytest.raises(PermissionError, match="symlink"):
        rr.ingest_resume_outbox(
            conn,
            board="default",
            outbox_dir=linked,
            producer_uid=request_file.stat().st_uid,
            policies=[policy],
        )
    assert (
        rr.ingest_resume_outbox(
            conn,
            board="default",
            outbox_dir=real,
            producer_uid=request_file.stat().st_uid,
            policies=[policy],
        )
        == []
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM kanban_resume_requests").fetchone()[0] == 0
    )


def _consume_exit_before_commit(db_path: str, policy, reached, proceed):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    conn = kb.connect(Path(db_path))

    def stop_during_validation(*args, **kwargs):
        reached.set()
        proceed.wait()
        os._exit(23)

    rr.candidate_fingerprint = stop_during_validation
    rr.consume_resume_requests(
        conn,
        board="default",
        gateway_profile="default",
        policies=[policy],
        lease_seconds=1,
    )


def _consumer_dispatcher(db_path: str, policy, barrier, launch_log: str):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    conn = kb.connect(Path(db_path))

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino
        )

    def launch(task, workspace, **kwargs):
        fd = os.open(launch_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(
                fd, f"{task.id}:{task.current_run_id}:{task.claim_lock}\n".encode()
            )
        finally:
            os.close(fd)
        return os.getpid()

    barrier.wait()
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[policy]
    )
    kb.dispatch_once(conn, spawn_fn=launch, workspace_capability_fn=capable)
    conn.close()


def test_candidate_is_revalidated_immediately_before_dispatch(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    request = rr._append_verified_request(conn, spec)
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
    request = rr._append_verified_request(conn, spec)
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
        assert tuple(row) == ("leased", 1)
        assert kb.get_task(restarted, task_id).status == "blocked"
        time.sleep(2)
        assert (
            rr.consume_resume_requests(
                restarted,
                board="default",
                gateway_profile="default",
                policies=[_policy(spec)],
            )[0].state
            == "accepted"
        )
        assert restarted.execute(
            "SELECT fence FROM kanban_resume_requests WHERE request_id=?",
            (request.request_id,),
        ).fetchone()[0] == 2


def test_two_consumer_dispatchers_emit_one_real_launch_intent(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    conn.close()
    launch_log = tmp_path / "launch.log"
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(3)
    workers = [
        ctx.Process(
            target=_consumer_dispatcher,
            args=(str(kb.kanban_db_path()), _policy(spec), barrier, str(launch_log)),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    intents = launch_log.read_text(encoding="utf-8").splitlines()
    assert len(intents) == 1
    with kb.connect() as check:
        task = kb.get_task(check, task_id)
        assert task.status == "running"
        assert task.current_run_id is not None
        assert task.claim_lock
        assert (
            check.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND status='running'",
                (task_id,),
            ).fetchone()[0]
            == 1
        )


def test_claim_before_launch_failure_is_fenced_then_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino
        )

    first = kb.dispatch_once(
        conn,
        spawn_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pre-exec")
        ),
        workspace_capability_fn=capable,
        failure_limit=2,
    )
    assert first.spawned == []
    assert kb.get_task(conn, task_id).status == "ready"
    ended = conn.execute(
        "SELECT status, ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert ended["status"] == "spawn_failed"
    assert ended["ended_at"] is not None

    launched = []
    second = kb.dispatch_once(
        conn,
        spawn_fn=lambda task, _path, **_kwargs: (
            launched.append(task.current_run_id) or 999999
        ),
        workspace_capability_fn=capable,
        failure_limit=2,
    )
    assert len(second.spawned) == 1
    assert len(launched) == 1
    assert kb.get_task(conn, task_id).status == "running"
    conn.close()


@pytest.mark.parametrize(
    "field",
    ["status", "version", "block_reason", "path", "branch", "sha", "fingerprint"],
)
def test_stale_expectations_reject_without_unblocking(tmp_path, monkeypatch, field):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
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
    rr._append_verified_request(conn, spec)
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
    rr._append_verified_request(conn, spec)
    result = rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    assert result[0].state == "rejected"
    assert kb.get_task(conn, task_id).status == "blocked"
    conn.close()


def test_only_actual_default_gateway_consumes(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
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
    request = rr._append_verified_request(conn, spec)
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


def test_candidate_fingerprint_never_executes_candidate_git_configuration(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    sha = _make_repo(repo)
    marker = tmp_path / "textconv-executed"
    helper = tmp_path / "textconv"
    helper.write_text(f"#!/bin/sh\ntouch {marker}\ncat \"$1\"\n", encoding="utf-8")
    helper.chmod(0o755)
    subprocess.run(
        ["git", "config", "diff.evil.textconv", str(helper)], cwd=repo, check=True
    )
    (repo / ".gitattributes").write_text("candidate.txt diff=evil\n", encoding="utf-8")
    (repo / "candidate.txt").write_text("dirty\n", encoding="utf-8")

    first = rr.candidate_fingerprint(repo, sha)
    second = rr.candidate_fingerprint(repo, sha)

    assert first == second
    assert not marker.exists()


def test_candidate_fingerprint_rejects_sparse_file_before_reading(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha = _make_repo(repo)
    huge = repo / "huge.bin"
    with huge.open("wb") as handle:
        handle.truncate(rr.MAX_CANDIDATE_FILE_BYTES + 1)

    with pytest.raises(ValueError, match="file byte limit"):
        rr.candidate_fingerprint(repo, sha)


def test_candidate_fingerprint_hashes_symlink_text_without_following_it(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret-a", encoding="utf-8")
    (root / "link").symlink_to(outside)
    first = rr.candidate_fingerprint(root, "a" * 40)
    outside.write_text("secret-b", encoding="utf-8")
    assert rr.candidate_fingerprint(root, "a" * 40) == first
    (root / "link").unlink()
    (root / "link").symlink_to(tmp_path / "different")
    assert rr.candidate_fingerprint(root, "a" * 40) != first


def test_candidate_fingerprint_authenticates_hardlink_content(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("shared", encoding="utf-8")
    os.link(outside, root / "hardlink")
    first = rr.candidate_fingerprint(root, "a" * 40)
    outside.write_text("changed", encoding="utf-8")
    assert rr.candidate_fingerprint(root, "a" * 40) != first


def test_candidate_fingerprint_rejects_mutation_during_stream(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha = _make_repo(repo)
    candidate = repo / "large.bin"
    candidate.write_bytes(b"x" * (2 * 1024 * 1024))
    original_read = rr.os.read
    mutated = False

    def racing_read(fd, size):
        nonlocal mutated
        data = original_read(fd, min(size, 1024))
        if data and not mutated:
            mutated = True
            candidate.write_bytes(b"y" * candidate.stat().st_size)
        return data

    monkeypatch.setattr(rr.os, "read", racing_read)
    with pytest.raises(ValueError, match="changed during authentication"):
        rr.candidate_fingerprint(repo, sha)


def test_candidate_fingerprint_honors_end_to_end_deadline(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha = _make_repo(repo)
    with pytest.raises(ValueError, match="deadline"):
        rr.candidate_fingerprint(repo, sha, deadline=time.monotonic() - 1)


def test_slow_validation_does_not_block_unrelated_writer(tmp_path, monkeypatch):
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    reached = threading.Event()
    proceed = threading.Event()
    original = rr.candidate_fingerprint

    def paused(*args, **kwargs):
        reached.set()
        assert proceed.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(rr, "candidate_fingerprint", paused)

    def consume():
        with kb.connect(kb.kanban_db_path()) as other:
            rr.consume_resume_requests(
                other,
                board="default",
                gateway_profile="default",
                policies=[_policy(spec)],
            )

    worker = threading.Thread(target=consume)
    worker.start()
    assert reached.wait(10)
    started = time.monotonic()
    kb.add_comment(conn, task_id, "operator", "writer remained available")
    assert time.monotonic() - started < 1
    proceed.set()
    worker.join(10)
    assert not worker.is_alive()


def test_legacy_dir_metadata_binding_uses_trusted_policy_and_public_api(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "league-table"
    sha = _make_repo(repo)
    conn = kb.connect()
    task_id = kb.create_task(
        conn,
        title="League Table legacy task",
        assignee="reviewer",
        workspace_kind="dir",
        workspace_path=str(repo),
        initial_status="running",
    )
    kb.block_task(
        conn, task_id, reason="iteration budget exhausted", kind="needs_input"
    )
    snap = rr.inspect_task_read_only(kb.kanban_db_path(), task_id)
    assert snap["branch"] is None and snap["expected_sha"] is None
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)

    result = rr.consume_resume_requests(
        conn,
        board="default",
        gateway_profile="default",
        policies=[_policy(spec, bind_legacy_metadata=True)],
    )

    assert result[0].state == "accepted"
    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.workspace_kind == "dir"
    assert task.branch_name == "main"
    assert task.expected_workspace_sha == sha


def test_accepted_request_cannot_redirect_dispatch_from_workspace_a_to_b(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo_a, sha_a, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo_a, sha_a, task_id)
    request = rr._append_verified_request(conn, spec)
    assert rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )[0].state == "accepted"
    repo_b = tmp_path / "repo-b"
    sha_b = _make_repo(repo_b)
    conn.execute(
        "UPDATE tasks SET workspace_path=?, branch_name='main', expected_workspace_sha=? "
        "WHERE id=?",
        (str(repo_b), sha_b, task_id),
    )
    launched = []

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), device=info.st_dev, inode=info.st_ino
        )

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *_args, **_kwargs: launched.append(True) or 123,
        workspace_capability_fn=capable,
    )

    assert result.spawned == []
    assert launched == []
    outcome = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(outcome) == ("rejected", "dispatch_stale_path")


def test_candidate_mutation_after_claim_is_reblocked_before_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    request = rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )

    real_claim = kb.claim_task

    def claim_then_mutate(*args, **kwargs):
        claimed = real_claim(*args, **kwargs)
        if claimed is not None:
            (repo / "candidate.txt").write_text("changed after claim\n", encoding="utf-8")
        return claimed

    monkeypatch.setattr(kb, "claim_task", claim_then_mutate)
    launches = []

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino
        )

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: launches.append((args, kwargs)) or 12345,
        workspace_capability_fn=capable,
    )

    assert result.spawned == []
    assert launches == []
    assert kb.get_task(conn, task_id).status == "blocked"
    outcome = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(outcome) == ("rejected", "dispatch_stale_fingerprint")


def test_consumer_leases_only_requested_bounded_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    conn = kb.connect()
    policies = []
    for index in range(3):
        repo = tmp_path / f"repo-{index}"
        sha = _make_repo(repo)
        task_id = kb.create_task(
            conn,
            title=f"bounded {index}",
            assignee="reviewer",
            workspace_kind="dir",
            workspace_path=str(repo),
            branch_name="main",
            expected_workspace_sha=sha,
            initial_status="running",
        )
        kb.block_task(conn, task_id, reason="iteration budget exhausted", kind="needs_input")
        spec = _request(
            rr.inspect_task_read_only(kb.kanban_db_path(), task_id), repo, sha, task_id
        )
        rr._append_verified_request(conn, spec)
        policies.append(_policy(spec))

    results = rr.consume_resume_requests(
        conn,
        board="default",
        gateway_profile="default",
        policies=policies,
        batch_size=2,
    )

    assert len(results) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_resume_requests WHERE state='pending'"
    ).fetchone()[0] == 1


def test_dir_branch_requires_an_authenticated_sha(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    conn = kb.connect()
    with pytest.raises(ValueError, match="requires expected_workspace_sha"):
        kb.create_task(
            conn,
            title="unsafe dir metadata",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
            branch_name="main",
        )
