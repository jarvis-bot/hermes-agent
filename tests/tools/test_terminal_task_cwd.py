"""Regression tests for task/session cwd propagation in terminal_tool."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool


def _minimal_terminal_config(cwd="/default"):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 60,
        "lifetime_seconds": 3600,
    }


def test_foreground_command_uses_registered_task_cwd_for_existing_environment(monkeypatch):
    """ACP can update task cwd after the local env exists; foreground must honor it."""
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/acp", "bounded_capture": True})]


def test_explicit_workdir_still_wins_over_registered_task_cwd(monkeypatch):
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            workdir="/explicit/workdir",
        )
    )

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir", "bounded_capture": True}]


def test_exact_sha_reviewer_does_not_remount_recorded_container_cwd(monkeypatch, tmp_path):
    calls = []
    ticket = tmp_path / "ticket"
    ticket.mkdir()

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    config = {
        "env_type": "docker",
        "cwd": "/workspace",
        "host_cwd": str(ticket),
        "docker_image": "python:3.11-slim",
        "docker_mount_cwd_to_workspace": True,
        "timeout": 60,
        "lifetime_seconds": 3600,
    }
    monkeypatch.setenv("HERMES_KANBAN_EXPECTED_WORKSPACE_SHA", "a" * 40)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {"default": "/tmp/review"})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", workdir="/tmp"))

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/tmp", "bounded_capture": True}]


def test_background_command_prefers_recorded_session_cwd_over_init_time_cwd(monkeypatch):
    """Background process launches must also use the recorded session cwd."""

    class FakeEnv:
        env = {}
        cwd = "/workspace/live"

    class FakeRegistry:
        def __init__(self):
            self.calls = []
            self.pending_watchers = []

        def spawn_local(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id="proc_test", pid=1234)

    import tools.process_registry as process_registry_mod

    registry = FakeRegistry()
    task_id = "session-live-cwd-bg"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/init"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/workspace/init"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", registry)
    terminal_tool.record_session_cwd(task_id, "/workspace/live")

    result = json.loads(
        terminal_tool.terminal_tool(
            command="sleep 1",
            task_id=task_id,
            background=True,
        )
    )

    assert result["exit_code"] == 0
    # session_key falls back to the raw task_id when no gateway contextvar is set
    # (it doesn't propagate to tool-worker threads), so process.kill / stop can
    # still find and terminate this background process.
    assert registry.calls == [{
        "command": "sleep 1",
        "cwd": "/workspace/live",
        "task_id": task_id,
        "session_key": task_id,
        "env_vars": {},
        "use_pty": False,
    }]


def test_safe_getcwd_falls_back_to_home_when_no_terminal_cwd(monkeypatch):
    def _boom():
        raise FileNotFoundError()

    monkeypatch.setattr(terminal_tool.os, "getcwd", _boom)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool.os.path, "expanduser", lambda p: "/home/me")
    assert terminal_tool._safe_getcwd() == "/home/me"


def test_resolved_ticket_cwd_becomes_dynamic_docker_mount_source(tmp_path):
    ticket_cwd = tmp_path / "ticket-149"
    ticket_cwd.mkdir()
    config = {
        "cwd": "/workspace",
        "host_cwd": "/opt/data",
        "docker_mount_cwd_to_workspace": True,
    }

    cwd, host_cwd = terminal_tool.resolve_container_cwd_mount(
        "docker", str(ticket_cwd), config
    )

    assert cwd == "/workspace"
    assert host_cwd == str(ticket_cwd)


def test_new_docker_environment_mounts_registered_ticket_cwd(tmp_path, monkeypatch):
    ticket_cwd = tmp_path / "ticket-149"
    ticket_cwd.mkdir()
    task_id = "ticket-149-review"
    created = {}

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            return {"output": "ok", "returncode": 0}

    config = {
        "env_type": "docker",
        "cwd": "/workspace",
        "host_cwd": str(tmp_path),
        "docker_image": "python:3.11-slim",
        "docker_mount_cwd_to_workspace": True,
        "timeout": 60,
        "lifetime_seconds": 3600,
    }
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(
        terminal_tool, "_task_env_overrides", {task_id: {"cwd": str(ticket_cwd)}}
    )
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    def fake_create_environment(**kwargs):
        created.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert created["cwd"] == "/workspace"
    assert created["host_cwd"] == str(ticket_cwd)


def test_container_cwd_resolution_preserves_non_mount_behavior():
    config = {
        "cwd": "/root",
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": False,
    }

    cwd, host_cwd = terminal_tool.resolve_container_cwd_mount(
        "docker", "/home/user/ticket", config
    )

    assert cwd == "/root"
    assert host_cwd is None
