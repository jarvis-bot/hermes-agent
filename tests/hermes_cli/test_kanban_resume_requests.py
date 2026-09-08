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
from hermes_cli import kanban_worker_launcher as worker_launcher
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


def _dispatch_crash_after_handshake(
    db_path: str, expected_fingerprint: str, worker_pid_file: str
):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    import hermes_cli.profiles as profiles

    profiles.profile_exists = lambda _profile: True
    conn = kb.connect(Path(db_path))

    def capable(profile, candidate):
        candidate = Path(candidate)
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=expected_fingerprint.removeprefix("sha256:"),
        )

    rr._current_workspace_capability = capable

    def spawn_then_crash(_task, _workspace, *, launch_intent):
        release_file = worker_pid_file + ".release"
        worker = subprocess.Popen(
            ["/bin/sh", "-c", 'while [ ! -f "$1" ]; do sleep 0.1; done', "sh", release_file],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not rr.handshake_resume_launch_intent(
            Path(db_path),
            intent_id=launch_intent.intent_id,
            generation=launch_intent.generation,
            task_id=launch_intent.task_id,
            run_id=launch_intent.run_id,
            claim_lock=launch_intent.claim_lock,
            worker_pid=worker.pid,
        ):
            worker.terminate()
            os._exit(24)
        Path(worker_pid_file).write_text(str(worker.pid), encoding="ascii")
        os._exit(23)

    kb.dispatch_once(conn, spawn_fn=spawn_then_crash, workspace_capability_fn=capable)


def _dispatch_crash_after_intent(db_path: str, expected_fingerprint: str):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    import hermes_cli.profiles as profiles

    profiles.profile_exists = lambda _profile: True
    conn = kb.connect(Path(db_path))

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=expected_fingerprint.removeprefix("sha256:"),
        )

    def crash(*_args, **_kwargs):
        os._exit(23)

    kb.dispatch_once(conn, spawn_fn=crash, workspace_capability_fn=capable)


def _consumer_dispatcher(
    db_path: str, policy, start, launch_log: str, entered=None, release=None
):
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    import hermes_cli.profiles as profiles

    profiles.profile_exists = lambda _profile: True
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
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(20)
        return os.getpid()

    assert start.wait(20)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[policy]
    )
    kb.dispatch_once(
        conn,
        board="default",
        spawn_fn=launch,
        workspace_capability_fn=capable,
    )
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
    first_start = ctx.Event()
    second_start = ctx.Event()
    entered = ctx.Event()
    release = ctx.Event()
    first = ctx.Process(
        target=_consumer_dispatcher,
        args=(str(kb.kanban_db_path()), _policy(spec), first_start, str(launch_log), entered, release),
    )
    second = ctx.Process(
        target=_consumer_dispatcher,
        args=(str(kb.kanban_db_path()), _policy(spec), second_start, str(launch_log)),
    )
    first.start()
    first_start.set()
    assert entered.wait(20)  # first owns the durable intent and is paused outside SQLite
    second.start()
    second_start.set()
    second.join(20)
    assert second.exitcode == 0
    release.set()
    first.join(20)
    assert first.exitcode == 0
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
        assert check.execute(
            "SELECT COUNT(*) FROM kanban_resume_launch_intents WHERE task_id=?",
            (task_id,),
        ).fetchone()[0] == 1


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

    launch_attempts = 0

    def fail_before_exec(*_args, **_kwargs):
        nonlocal launch_attempts
        launch_attempts += 1
        raise TypeError("pre-exec")

    first = kb.dispatch_once(
        conn,
        spawn_fn=fail_before_exec,
        workspace_capability_fn=capable,
        failure_limit=2,
    )
    assert first.spawned == []
    assert launch_attempts == 1
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


def test_spawn_callback_runs_without_sqlite_writer_lock(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    other_id = kb.create_task(conn, title="unrelated comment", initial_status="blocked")
    heartbeat_id = kb.create_task(
        conn, title="unrelated heartbeat", assignee="reviewer", initial_status="blocked"
    )
    conn.execute(
        "UPDATE tasks SET status='ready', block_kind=NULL WHERE id=?", (heartbeat_id,)
    )
    heartbeat_task = kb.claim_task(conn, heartbeat_id)
    assert heartbeat_task is not None
    claim_id = kb.create_task(
        conn, title="unrelated claim", assignee="reviewer", initial_status="blocked"
    )
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    entered = threading.Event()
    release = threading.Event()

    def paused_spawn(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return 4242

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    def run_dispatch():
        with kb.connect() as dispatch_conn:
            kb.dispatch_once(
                dispatch_conn,
                spawn_fn=paused_spawn,
                workspace_capability_fn=capable,
            )

    dispatch = threading.Thread(target=run_dispatch)
    dispatch.start()
    assert entered.wait(5)
    callback_entered_at = time.monotonic()
    started = time.monotonic()
    with kb.connect() as writer:
        kb.add_comment(writer, other_id, "test", "must not wait for spawn")
        assert kb.heartbeat_claim(
            writer, heartbeat_id, claimer=heartbeat_task.claim_lock
        )
        writer.execute(
            "UPDATE tasks SET status='ready', block_kind=NULL WHERE id=?", (claim_id,)
        )
        assert kb.claim_task(writer, claim_id) is not None
    elapsed = time.monotonic() - started
    # Keep the callback paused for a deterministic two-second window while
    # proving all three unrelated writer paths completed promptly.
    time.sleep(max(0.0, 2.0 - (time.monotonic() - callback_entered_at)))
    release.set()
    dispatch.join(10)
    assert not dispatch.is_alive()
    assert elapsed < 0.5


def test_spawn_callback_can_reenter_database_write(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    other_id = kb.create_task(conn, title="unrelated", initial_status="blocked")
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    def reentrant_spawn(*_args, **_kwargs):
        kb.add_comment(conn, other_id, "spawn", "reentrant write")
        return 4242

    result = kb.dispatch_once(
        conn, spawn_fn=reentrant_spawn, workspace_capability_fn=capable
    )
    assert [item[0] for item in result.spawned] == [task_id]
    assert conn.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id=?", (other_id,)
    ).fetchone()[0] == 1


def test_provenance_change_after_intent_commit_rejects_worker_handshake(
    tmp_path, monkeypatch
):
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
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    monkeypatch.setattr(
        rr,
        "_current_workspace_capability",
        lambda profile, workspace: capable(profile, Path(workspace)),
    )
    task_work = []

    def launch(_task, _workspace, *, launch_intent):
        with kb.connect() as attacker:
            attacker.execute(
                "UPDATE tasks SET expected_workspace_sha=? WHERE id=?",
                ("b" * 40, task_id),
            )
        if rr.handshake_resume_launch_intent(
            kb.kanban_db_path(),
            intent_id=launch_intent.intent_id,
            generation=launch_intent.generation,
            task_id=launch_intent.task_id,
            run_id=launch_intent.run_id,
            claim_lock=launch_intent.claim_lock,
            worker_pid=4242,
        ):
            task_work.append("ran")
        return 4242

    result = kb.dispatch_once(conn, spawn_fn=launch, workspace_capability_fn=capable)
    assert result.spawned == []
    assert task_work == []
    intent = conn.execute(
        "SELECT state FROM kanban_resume_launch_intents WHERE task_id=?", (task_id,)
    ).fetchone()
    assert intent["state"] == "rejected"


def test_handshaken_worker_survives_coordinator_receipt_gap_without_duplicate(
    tmp_path, monkeypatch
):
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
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    monkeypatch.setattr(
        rr,
        "_current_workspace_capability",
        lambda profile, workspace: capable(profile, Path(workspace)),
    )
    launches = []

    def launch(_task, _workspace, *, launch_intent):
        launches.append(launch_intent.intent_id)
        assert rr.handshake_resume_launch_intent(
            kb.kanban_db_path(),
            intent_id=launch_intent.intent_id,
            generation=launch_intent.generation,
            task_id=launch_intent.task_id,
            run_id=launch_intent.run_id,
            claim_lock=launch_intent.claim_lock,
            worker_pid=os.getpid(),
        )
        return os.getpid()

    first = kb.dispatch_once(conn, spawn_fn=launch, workspace_capability_fn=capable)
    assert [item[0] for item in first.spawned] == [task_id]
    assert len(launches) == 1
    state = conn.execute(
        "SELECT state, worker_pid FROM kanban_resume_launch_intents WHERE task_id=?",
        (task_id,),
    ).fetchone()
    assert tuple(state) == ("handshaken", os.getpid())

    # Model a Popen receipt persisted just before coordinator death while the
    # child is live but has not yet completed its handshake. Reconciliation
    # must renew this exact owner/claim instead of letting generic lease reaping
    # create a duplicate generation.
    conn.execute(
        "UPDATE kanban_resume_launch_intents SET state='spawned', expires_at=0 "
        "WHERE task_id=?", (task_id,),
    )
    conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task_id,))
    conn.execute("UPDATE task_runs SET claim_expires=0 WHERE task_id=?", (task_id,))
    assert rr.reconcile_resume_launch_intents(conn) == 0
    assert len(launches) == 1
    renewed = conn.execute(
        "SELECT i.state, i.expires_at, t.status, t.claim_expires "
        "FROM kanban_resume_launch_intents i JOIN tasks t ON t.id=i.task_id "
        "WHERE i.task_id=?", (task_id,),
    ).fetchone()
    assert renewed["state"] == "spawned"
    assert renewed["expires_at"] > int(time.time())
    assert renewed["status"] == "running"
    assert renewed["claim_expires"] == renewed["expires_at"]
    identity = conn.execute(
        "SELECT intent_id, generation, run_id, claim_lock FROM "
        "kanban_resume_launch_intents WHERE task_id=?", (task_id,),
    ).fetchone()
    assert rr.handshake_resume_launch_intent(
        kb.kanban_db_path(),
        intent_id=identity["intent_id"],
        generation=identity["generation"],
        task_id=task_id,
        run_id=identity["run_id"],
        claim_lock=identity["claim_lock"],
        worker_pid=os.getpid(),
    )


def test_real_launcher_handshakes_after_parent_records_spawned(tmp_path, monkeypatch):
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
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    monkeypatch.setattr(
        rr,
        "_current_workspace_capability",
        lambda profile, workspace: capable(profile, Path(workspace)),
    )
    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *_args, **_kwargs: os.getpid(),
        workspace_capability_fn=capable,
    )
    assert [item[0] for item in result.spawned] == [task_id]
    identity = conn.execute(
        "SELECT intent_id, generation, run_id, claim_lock, state FROM "
        "kanban_resume_launch_intents WHERE task_id=?", (task_id,),
    ).fetchone()
    assert identity["state"] == "spawned"
    assert conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()[0] == "spawned"

    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_KANBAN_LAUNCH_INTENT_ID", identity["intent_id"])
    monkeypatch.setenv("HERMES_KANBAN_LAUNCH_GENERATION", str(identity["generation"]))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(identity["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", identity["claim_lock"])

    class ExecReached(Exception):
        pass

    def exec_after_handshake(*_args):
        raise ExecReached

    monkeypatch.setattr(worker_launcher.os, "execvpe", exec_after_handshake)
    with pytest.raises(ExecReached):
        worker_launcher.main(["--", "hermes", "chat", "-q", "work"])
    persisted = conn.execute(
        "SELECT state, worker_pid FROM kanban_resume_launch_intents WHERE intent_id=?",
        (identity["intent_id"],),
    ).fetchone()
    assert tuple(persisted) == ("handshaken", os.getpid())


def test_stale_spawned_generation_cannot_handshake_after_replay(tmp_path, monkeypatch):
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
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    monkeypatch.setattr(
        rr,
        "_current_workspace_capability",
        lambda profile, workspace: capable(profile, Path(workspace)),
    )
    intents = []

    def launch(_task, _workspace, *, launch_intent):
        intents.append(launch_intent)
        return 999999

    first = kb.dispatch_once(conn, spawn_fn=launch, workspace_capability_fn=capable)
    assert len(first.spawned) == 1
    stale = intents[0]
    conn.execute(
        "UPDATE kanban_resume_launch_intents SET expires_at=0 WHERE intent_id=?",
        (stale.intent_id,),
    )
    conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task_id,))
    conn.execute("UPDATE task_runs SET claim_expires=0 WHERE task_id=?", (task_id,))
    second = kb.dispatch_once(conn, spawn_fn=launch, workspace_capability_fn=capable)
    assert len(second.spawned) == 1
    assert [intent.generation for intent in intents] == [1, 2]

    # Even if stale state is corrupted back to a nominally pending receipt,
    # its old claim/run provenance cannot authorize task code.
    conn.execute(
        "UPDATE kanban_resume_launch_intents SET state='spawned', expires_at=? "
        "WHERE intent_id=?", (int(time.time()) + 60, stale.intent_id),
    )
    assert not rr.handshake_resume_launch_intent(
        kb.kanban_db_path(),
        intent_id=stale.intent_id,
        generation=stale.generation,
        task_id=stale.task_id,
        run_id=stale.run_id,
        claim_lock=stale.claim_lock,
        worker_pid=999999,
    )
    assert conn.execute(
        "SELECT state FROM kanban_resume_launch_intents WHERE intent_id=?",
        (stale.intent_id,),
    ).fetchone()[0] == "rejected"


def test_popen_success_then_coordinator_crash_does_not_duplicate_active_worker(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    pid_file = tmp_path / "worker.pid"
    ctx = multiprocessing.get_context("fork")
    coordinator = ctx.Process(
        target=_dispatch_crash_after_handshake,
        args=(
            str(kb.kanban_db_path()),
            spec.expected_candidate_fingerprint,
            str(pid_file),
        ),
    )
    coordinator.start()
    coordinator.join(30)
    assert coordinator.exitcode == 23
    worker_pid = int(pid_file.read_text(encoding="ascii"))
    try:
        assert kb._pid_alive(worker_pid)
        launches = []
        duplicate = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: launches.append(True) or 9898,
            workspace_capability_fn=lambda profile, candidate: WorkspaceCapability(
                True,
                profile,
                str(candidate),
                "",
                device=candidate.stat().st_dev,
                inode=candidate.stat().st_ino,
                content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
            ),
        )
        assert duplicate.spawned == []
        assert launches == []
        intent = conn.execute(
            "SELECT state, worker_pid FROM kanban_resume_launch_intents WHERE task_id=?",
            (task_id,),
        ).fetchone()
        assert tuple(intent) == ("handshaken", worker_pid)
    finally:
        Path(str(pid_file) + ".release").touch()
        deadline = time.monotonic() + 5
        while kb._pid_alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)


def test_expired_orphan_intent_replays_with_new_generation(tmp_path, monkeypatch):
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
            True, profile, str(candidate), "", device=info.st_dev, inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    ctx = multiprocessing.get_context("fork")
    crashed = ctx.Process(
        target=_dispatch_crash_after_intent,
        args=(str(kb.kanban_db_path()), spec.expected_candidate_fingerprint),
    )
    crashed.start()
    crashed.join(30)
    assert crashed.exitcode == 23
    first = conn.execute(
        "SELECT intent_id, generation FROM kanban_resume_launch_intents WHERE task_id=?",
        (task_id,),
    ).fetchone()
    assert first is not None
    # A real abrupt death cannot run the exception reconciler. Restore the
    # persisted pre-callback state and expire both intent and claim deterministically.
    conn.execute(
        "UPDATE kanban_resume_launch_intents SET state='prepared', expires_at=0, finished_at=NULL "
        "WHERE intent_id=?", (first["intent_id"],),
    )
    conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task_id,))
    conn.execute("UPDATE task_runs SET claim_expires=0 WHERE task_id=?", (task_id,))
    launches = []
    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *_args, **_kwargs: launches.append(True) or 4343,
        workspace_capability_fn=capable,
    )
    assert [item[0] for item in result.spawned] == [task_id], (
        result,
        dict(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()),
        dict(conn.execute("SELECT * FROM kanban_resume_requests").fetchone()),
        [dict(row) for row in conn.execute("SELECT * FROM kanban_resume_launch_intents")],
    )
    assert launches == [True]
    generations = [row[0] for row in conn.execute(
        "SELECT generation FROM kanban_resume_launch_intents WHERE task_id=? ORDER BY generation",
        (task_id,),
    )]
    assert generations == [1, 2]


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


@pytest.mark.parametrize(
    ("drift", "expected_code"),
    [
        ("path", "dispatch_stale_path"),
        ("kind", "dispatch_stale_workspace_kind"),
        ("branch", "dispatch_stale_branch"),
        ("sha", "dispatch_stale_sha"),
        ("version", "dispatch_stale_version"),
        ("accepted_event", "dispatch_stale_version"),
        ("request_state", "dispatch_stale_request"),
        ("request_lease", "dispatch_stale_request"),
        ("policy", "dispatch_stale_request"),
    ],
)
def test_task_provenance_mutation_after_real_claim_never_launches(
    tmp_path, monkeypatch, drift, expected_code
):
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
        if claimed is None:
            return None
        if drift == "path":
            conn.execute(
                "UPDATE tasks SET workspace_path=? WHERE id=?",
                (str(tmp_path / "other"), task_id),
            )
        elif drift == "kind":
            conn.execute("UPDATE tasks SET workspace_kind='scratch' WHERE id=?", (task_id,))
        elif drift == "branch":
            conn.execute("UPDATE tasks SET branch_name='other' WHERE id=?", (task_id,))
        elif drift == "sha":
            conn.execute(
                "UPDATE tasks SET expected_workspace_sha=? WHERE id=?",
                ("b" * 40, task_id),
            )
        elif drift == "version":
            kb._append_event(conn, task_id, "commented", {"race": True})
        elif drift == "accepted_event":
            conn.execute(
                "UPDATE task_events SET payload=? WHERE id=(SELECT MAX(id) FROM task_events "
                "WHERE task_id=? AND kind='resume_request_accepted')",
                ('{"request_id":"tampered"}', task_id),
            )
        elif drift == "request_state":
            conn.execute(
                "UPDATE kanban_resume_requests SET result_code='superseded' "
                "WHERE request_id=?",
                (request.request_id,),
            )
        elif drift == "request_lease":
            conn.execute(
                "UPDATE kanban_resume_requests SET lease_owner='attacker', "
                "lease_expires=? WHERE request_id=?",
                (int(time.time()) + 60, request.request_id),
            )
        elif drift == "policy":
            conn.execute(
                "UPDATE kanban_resume_requests SET expected_block_reason_sha256=? "
                "WHERE request_id=?",
                ("b" * 64, request.request_id),
            )
        return claimed

    monkeypatch.setattr(kb, "claim_task", claim_then_mutate)
    launches = []

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True,
            profile,
            str(candidate),
            "",
            device=info.st_dev,
            inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: launches.append((args, kwargs)) or 12345,
        workspace_capability_fn=capable,
    )

    assert result.spawned == []
    assert launches == []
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.claim_lock is None
    assert task.current_run_id is None
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND status='running'",
        (task_id,),
    ).fetchone()[0] == 0
    outcome = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(outcome) == ("rejected", expected_code)


def test_task_provenance_mutation_at_final_spawn_fence_never_launches(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    request = rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    real_finalize = rr.finalize_resume_dispatch_spawned

    def mutate_then_finalize(connection, binding, *args, **kwargs):
        connection.execute(
            "UPDATE tasks SET expected_workspace_sha=? WHERE id=?",
            ("b" * 40, task_id),
        )
        return real_finalize(connection, binding, *args, **kwargs)

    monkeypatch.setattr(rr, "finalize_resume_dispatch_spawned", mutate_then_finalize)
    launches = []

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True,
            profile,
            str(candidate),
            "",
            device=info.st_dev,
            inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: launches.append((args, kwargs)) or 12345,
        workspace_capability_fn=capable,
    )

    assert result.spawned == []
    assert launches == []
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.claim_lock is None
    assert task.current_run_id is None
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND status='running'",
        (task_id,),
    ).fetchone()[0] == 0
    outcome = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(outcome) == ("rejected", "dispatch_stale_sha")


@pytest.mark.parametrize("winner_drift", ["claim", "run"])
def test_post_claim_race_loser_cleanup_preserves_unrelated_winner(
    tmp_path, monkeypatch, winner_drift
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    conn, task_id, repo, sha, snap = _blocked_task(tmp_path, monkeypatch)
    spec = _request(snap, repo, sha, task_id)
    request = rr._append_verified_request(conn, spec)
    rr.consume_resume_requests(
        conn, board="default", gateway_profile="default", policies=[_policy(spec)]
    )
    real_claim = kb.claim_task
    winner_lock = "unrelated-winner"
    winner_run_id = None
    loser_run_id = None

    def claim_then_install_winner(*args, **kwargs):
        nonlocal winner_run_id, loser_run_id
        claimed = real_claim(*args, **kwargs)
        assert claimed is not None
        loser_run_id = claimed.current_run_id
        if winner_drift == "claim":
            winner_run_id = loser_run_id
            conn.execute(
                "UPDATE tasks SET claim_lock=? WHERE id=?",
                (winner_lock, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET claim_lock=? WHERE id=?",
                (winner_lock, winner_run_id),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
                "claim_expires, started_at) VALUES (?, 'reviewer', 'running', ?, ?, ?)",
                (task_id, winner_lock, int(time.time()) + 60, int(time.time())),
            )
            winner_run_id = int(cursor.lastrowid)
            conn.execute(
                "UPDATE tasks SET claim_lock=?, current_run_id=? WHERE id=?",
                (winner_lock, winner_run_id, task_id),
            )
        return claimed

    monkeypatch.setattr(kb, "claim_task", claim_then_install_winner)
    launches = []

    def capable(profile, candidate):
        info = candidate.stat()
        return WorkspaceCapability(
            True,
            profile,
            str(candidate),
            device=info.st_dev,
            inode=info.st_ino,
            content_sha256=spec.expected_candidate_fingerprint.removeprefix("sha256:"),
        )

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda *args, **kwargs: launches.append((args, kwargs)) or 12345,
        workspace_capability_fn=capable,
    )

    assert result.spawned == []
    assert launches == []
    task = kb.get_task(conn, task_id)
    assert task.status == "running"
    assert task.claim_lock == winner_lock
    assert task.current_run_id == winner_run_id
    winner = conn.execute(
        "SELECT status, ended_at, claim_lock FROM task_runs WHERE id=?",
        (winner_run_id,),
    ).fetchone()
    assert tuple(winner) == ("running", None, winner_lock)
    if loser_run_id != winner_run_id:
        loser = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=?", (loser_run_id,)
        ).fetchone()
        assert loser["status"] == "workspace_changed"
        assert loser["ended_at"] is not None
    outcome = conn.execute(
        "SELECT state, result_code FROM kanban_resume_requests WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    assert tuple(outcome) == ("rejected", "dispatch_stale_claim")


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
