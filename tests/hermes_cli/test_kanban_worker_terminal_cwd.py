"""Tests: kanban worker spawn pins TERMINAL_CWD to the task workspace.

Regression coverage for #34619 and #41312 (same root cause): ``_default_spawn``
launched the worker subprocess with ``cwd=workspace`` and set
``HERMES_KANBAN_WORKSPACE``, but did NOT set ``TERMINAL_CWD``. Because
``TERMINAL_CWD`` takes precedence over the process cwd in both
``tools/file_tools.py::_resolve_base_dir`` (relative ``write_file`` paths) and
``agent_init``'s context-file loader (``AGENTS.md`` discovery), workers inherited
the dispatching gateway's cwd — relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` was loaded (#34619).
Pinning ``TERMINAL_CWD`` to the workspace fixes both.
"""

from __future__ import annotations

import subprocess


def _make_task(
    kb,
    *,
    assignee: str = "w",
    expected_workspace_sha: str | None = None,
    requires_reviewer_isolation: bool = False,
):
    return kb.Task(
        id="t_cwd",
        title="cwd pin",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        expected_workspace_sha=expected_workspace_sha,
        requires_reviewer_isolation=requires_reviewer_isolation,
        current_run_id=1,
    )


def _capture_spawn_env(
    kb,
    monkeypatch,
    workspace: str,
    *,
    expected_workspace_sha: str | None = None,
    assignee: str = "w",
    requires_reviewer_isolation: bool = False,
) -> dict:
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(
        _make_task(
            kb,
            assignee=assignee,
            expected_workspace_sha=expected_workspace_sha,
            requires_reviewer_isolation=requires_reviewer_isolation,
        ),
        workspace,
    )
    return captured


def test_terminal_cwd_pinned_to_workspace(monkeypatch, tmp_path):
    """A real, absolute workspace dir is pinned as TERMINAL_CWD."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_expected_workspace_sha_is_pinned_in_worker_environment(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    assigned_sha = "a" * 40
    captured = _capture_spawn_env(
        kb,
        monkeypatch,
        str(workspace),
        expected_workspace_sha=assigned_sha,
    )

    assert captured["env"]["HERMES_KANBAN_EXPECTED_WORKSPACE_SHA"] == assigned_sha


def test_unpinned_worker_clears_inherited_expected_workspace_sha(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_EXPECTED_WORKSPACE_SHA", "b" * 40)

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert "HERMES_KANBAN_EXPECTED_WORKSPACE_SHA" not in captured["env"]


def test_read_only_reviewer_worker_forces_safe_restricted_tool_surface(
    monkeypatch, tmp_path
):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "security-reviewer"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "toolsets:\n"
        "  - hermes-cli\n"
        "terminal:\n"
        "  backend: docker\n"
        "  docker_mount_cwd_to_workspace: true\n"
        "  docker_cwd_mount_mode: ro\n"
        "  docker_network: false\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - hermes-cli\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured = _capture_spawn_env(
        kb,
        monkeypatch,
        str(workspace),
        assignee="security-reviewer",
    )

    assert captured["env"]["HERMES_SAFE_MODE"] == "1"
    assert "--accept-hooks" not in captured["cmd"]
    assert "--ignore-rules" in captured["cmd"]
    toolsets_index = captured["cmd"].index("--toolsets")
    assert captured["cmd"][toolsets_index + 1] == "terminal,kanban"


def test_reviewer_isolation_survives_fallback_assignee_without_sha(
    monkeypatch, tmp_path
):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "default"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "toolsets:\n  - hermes-cli\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - hermes-cli\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured = _capture_spawn_env(
        kb,
        monkeypatch,
        str(workspace),
        assignee="default",
        requires_reviewer_isolation=True,
    )

    assert captured["env"]["HERMES_SAFE_MODE"] == "1"
    assert "--accept-hooks" not in captured["cmd"]
    assert "--ignore-rules" in captured["cmd"]
    toolsets_index = captured["cmd"].index("--toolsets")
    assert captured["cmd"][toolsets_index + 1] == "terminal,kanban"


