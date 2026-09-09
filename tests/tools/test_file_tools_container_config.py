"""Tests for docker container_config key propagation in file_tools."""

from unittest.mock import patch, MagicMock
import tools.file_tools as file_tools


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 180,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20480,
        "container_persistent": False,
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_complete_docker_policy_passed_to_file_environment(self):
        """File-first creation must preserve policy and reviewer rejections."""
        config = _make_env_config(
            docker_env={"UNSAFE": "value"},
            docker_extra_args=["--privileged"],
            docker_persist_across_processes=False,
            docker_tmp_storage="disk",
        )
        cc = self._run(config, "file-policy")["container_config"]
        assert cc["docker_env"] == {"UNSAFE": "value"}
        assert cc["docker_extra_args"] == ["--privileged"]
        assert cc["docker_persist_across_processes"] is False
        assert cc["docker_tmp_storage"] == "disk"


    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"

    def test_docker_file_environment_mounts_raw_task_workspace(self, tmp_path):
        ticket_cwd = tmp_path / "ticket-149"
        ticket_cwd.mkdir()
        task_id = "ticket-149-review"

        captured = self._run(
            _make_env_config(host_cwd=str(tmp_path)),
            task_id,
            task_env_overrides={task_id: {"cwd": str(ticket_cwd)}},
        )

        assert captured["cwd"] == "/workspace"
        assert captured["host_cwd"] == str(ticket_cwd)

    def test_exact_sha_file_first_recreation_uses_authenticated_host_source(
        self, monkeypatch, tmp_path
    ):
        workspace = tmp_path / "ticket-review"
        workspace.mkdir()
        monkeypatch.setenv("HERMES_KANBAN_EXPECTED_WORKSPACE_SHA", "a" * 40)

        with patch("tools.terminal_tool.get_session_cwd", return_value="/tmp/review"):
            captured = self._run(
                _make_env_config(cwd="/workspace", host_cwd=str(workspace)),
                "review-file-first",
            )

        assert captured["cwd"] == "/workspace"
        assert captured["host_cwd"] == str(workspace)
