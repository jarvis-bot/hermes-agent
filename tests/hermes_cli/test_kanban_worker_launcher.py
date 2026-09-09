from __future__ import annotations

import os

from hermes_cli import kanban_worker_launcher as launcher


def _env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/board.db")
    monkeypatch.setenv("HERMES_KANBAN_LAUNCH_INTENT_ID", "rli_test")
    monkeypatch.setenv("HERMES_KANBAN_LAUNCH_GENERATION", "3")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "claim")


def test_stale_handshake_exits_without_executing_task(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.kanban_resume_requests.handshake_resume_launch_intent",
        lambda *_args, **_kwargs: False,
    )
    executed = []
    monkeypatch.setattr(os, "execvpe", lambda *args: executed.append(args))

    assert launcher.main(["--", "hermes", "chat", "-q", "work"]) == 75
    assert executed == []


def test_valid_handshake_executes_exact_worker_argv(monkeypatch):
    _env(monkeypatch)
    observed = {}

    def handshake(_path, **identity):
        observed.update(identity)
        return True

    monkeypatch.setattr(
        "hermes_cli.kanban_resume_requests.handshake_resume_launch_intent", handshake
    )

    class ExecReached(Exception):
        pass

    def execvpe(executable, argv, env):
        observed["exec"] = (
            executable,
            argv,
            env["HERMES_KANBAN_LAUNCH_INTENT_ID"],
            env["PYTHONSAFEPATH"],
            env["PYTHONPATH"],
        )
        raise ExecReached

    monkeypatch.setattr(os, "execvpe", execvpe)
    try:
        launcher.main(["--", "hermes", "chat", "-q", "work"])
    except ExecReached:
        pass
    else:  # pragma: no cover
        raise AssertionError("launcher did not exec the worker")

    assert observed["intent_id"] == "rli_test"
    assert observed["generation"] == 3
    assert observed["task_id"] == "t_test"
    assert observed["run_id"] == 7
    assert observed["claim_lock"] == "claim"
    assert observed["exec"] == (
        "hermes",
        ["hermes", "chat", "-q", "work"],
        "rli_test",
        "1",
        launcher._TRUSTED_ROOT,
    )
