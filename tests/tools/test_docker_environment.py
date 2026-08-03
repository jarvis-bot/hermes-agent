import logging
from io import BytesIO, StringIO
import os
import subprocess
import tarfile

import pytest

from tools.environments import docker as docker_env


def test_reviewer_rejects_scratch_mount_resolving_into_workspace(monkeypatch):
    """An image symlink must not turn an allowlisted mount into a workspace write."""
    env = object.__new__(docker_env.DockerEnvironment)
    env._docker_exe = "/usr/bin/docker"
    mounts = '[{"Destination":"/root","RW":true}]\n'
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, mounts, ""),
    )
    monkeypatch.setattr(
        env,
        "_container_resolved_path",
        lambda _container_id, path: "/workspace/root" if path == "/root" else path,
    )

    assert env._unexpected_reviewer_writable_mount("container") == "/root"


def test_reviewer_rejects_workspace_path_resolving_elsewhere(monkeypatch):
    env = object.__new__(docker_env.DockerEnvironment)
    env._reviewer_mode = True
    env._workspace_requires_ro = True
    env._tmp_storage = "tmpfs"
    monkeypatch.setattr(env, "_readonly_workspace_identity_violation", lambda: None)
    monkeypatch.setattr(env, "_unexpected_reviewer_writable_mount", lambda _id: None)
    monkeypatch.setattr(env, "_container_has_mount_at_or_below", lambda *a, **kw: False)
    monkeypatch.setattr(
        env,
        "_container_resolved_path",
        lambda _id, path: "/tmp/redirect" if path == "/workspace" else path,
    )

    violation = env._effective_policy_violation("container")
    assert violation is not None
    assert "read-only /workspace" in violation


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.

    Pre-seeds the cgroup-limit probe cache to ``True`` so the throwaway probe
    container (a ``docker run ... sleep 0``) does not run and pollute the
    captured call list — these tests inspect the real sandbox-start ``run``.
    Tests that exercise the probe itself live in test_docker_cgroup_limits.py.
    """
    docker_env._cgroup_limits_ok = True
    calls = []
    snapshot_digests = {}

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
            if cmd[1] == "exec" and len(cmd) >= 10 and cmd[-3] == "-c":
                run_cmd = next(
                    call[0]
                    for call in reversed(calls[:-1])
                    if isinstance(call[0], list) and call[0][1] == "run"
                )
                container_path = cmd[-1]
                workspace_spec = next(
                    run_cmd[index + 1]
                    for index, arg in enumerate(run_cmd[:-1])
                    if arg == "-v"
                    and run_cmd[index + 1].split(":", 2)[1] == container_path
                )
                source = workspace_spec.split(":", 1)[0]
                digest = snapshot_digests.get(source)
                if digest is None:
                    digest = docker_env._readonly_tree_digest(
                        docker_env.Path(source), include_root_mode=False
                    )
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{digest}\n", stderr=""
                )
            if cmd[1] == "image" and cmd[2] == "inspect":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
                protected = cmd[-1].rsplit(" ", 1)[-1]
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{protected}\n", stderr="")
            if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    def _materialize(
        _docker, _image, _source, expected, _expected_git_sha=None, *, disposable=False
    ):
        volume = f"hermes-ro-{str(expected['mounted_content_sha256'])[:24]}"
        snapshot_digests[volume] = str(expected["mounted_content_sha256"])
        return volume

    monkeypatch.setattr(docker_env, "_materialize_readonly_workspace", _materialize)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        forward_env=kwargs.get("forward_env"),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        cwd_mount_mode=kwargs.get("cwd_mount_mode", "rw"),
        cwd_path_mappings=kwargs.get("cwd_path_mappings"),
        cwd_allowed_roots=kwargs.get("cwd_allowed_roots"),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
        extra_args=kwargs.get("extra_args", []),
        tmp_storage=kwargs.get("tmp_storage", "tmpfs"),
        expected_git_sha=kwargs.get("expected_git_sha"),
        persist_across_processes=kwargs.get("persist_across_processes", True),
    )


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker cannot be found, raise a clear error before container setup."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called when docker is missing"),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker executable not found in PATH or known install locations" in str(excinfo.value)
    assert any(
        "no docker executable was found in PATH or known install locations"
        in record.getMessage()
        for record in caplog.records
    )


def test_auto_mount_host_cwd_adds_volume(monkeypatch, tmp_path):
    """Opt-in docker cwd mounting should bind the host cwd to /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Find the docker run call and check its args
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str


def test_disk_tmp_storage_uses_container_writable_layer(monkeypatch):
    """Disk mode must omit only the /tmp tmpfs while retaining other hardening."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(tmp_storage="disk", persist_across_processes=False)

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    tmpfs_mounts = [
        run_args[index + 1]
        for index, arg in enumerate(run_args[:-1])
        if arg == "--tmpfs"
    ]
    assert not any(mount.startswith("/tmp:") for mount in tmpfs_mounts)
    assert any(mount.startswith("/var/tmp:") for mount in tmpfs_mounts)
    assert "no-new-privileges" in run_args


def test_disk_tmp_storage_rejects_image_declared_tmp_volume(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="disk-container\n", stderr="")
        if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
            mounts = '[{"Type":"volume","Destination":"/tmp"}]\n'
            return subprocess.CompletedProcess(cmd, 0, stdout=mounts, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="container writable layer"):
        _make_dummy_env(tmp_storage="disk", persist_across_processes=False)

    assert ["/usr/bin/docker", "rm", "-f", "-v", "disk-container"] in calls


def test_disk_tmp_storage_rejects_ancestor_root_mount(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="disk-container\n", stderr="")
        if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="/tmp\n", stderr="")
        if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
            mounts = '[{"Type":"volume","Destination":"/","RW":true}]\n'
            return subprocess.CompletedProcess(cmd, 0, stdout=mounts, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="container writable layer"):
        _make_dummy_env(tmp_storage="disk", persist_across_processes=False)

    assert ["/usr/bin/docker", "rm", "-f", "-v", "disk-container"] in calls


def test_default_tmp_storage_preserves_hardened_tmpfs(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(persist_across_processes=False)

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    assert "/tmp:rw,nosuid,size=512m" in run_args


def test_disk_tmp_storage_rejects_image_volume_reached_through_tmp_symlink(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:symlink-image\n", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="disk-container\n", stderr="")
        if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="/scratch\n", stderr="")
        if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
            mounts = '[{"Type":"volume","Destination":"/scratch","RW":true}]\n'
            return subprocess.CompletedProcess(cmd, 0, stdout=mounts, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="container writable layer"):
        _make_dummy_env(tmp_storage="disk", persist_across_processes=False)

    assert ["/usr/bin/docker", "rm", "-f", "-v", "disk-container"] in calls


def test_disk_tmp_storage_rejects_tmp_symlink_below_ancestor_mount(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:symlink-image\n", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="disk-container\n", stderr="")
        if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="/workspace/tmp\n", stderr="")
        if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
            mounts = '[{"Type":"bind","Destination":"/workspace","RW":false}]\n'
            return subprocess.CompletedProcess(cmd, 0, stdout=mounts, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="container writable layer"):
        _make_dummy_env(tmp_storage="disk", persist_across_processes=False)

    assert ["/usr/bin/docker", "rm", "-f", "-v", "disk-container"] in calls


def test_tmp_storage_participates_in_container_reuse_fingerprint(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(tmp_storage="disk")

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    assert "hermes-tmp-storage=disk" in run_args
    reuse_probe = next(c[0] for c in calls if c[0][1:3] == ["ps", "-a"])
    assert "label=hermes-tmp-storage=disk" in reuse_probe


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--label", "hermes-tmp-storage=tmpfs"],
        ["--label=hermes-tmp-storage=tmpfs"],
        ["-l", "hermes-task-id=other"],
        ["-l=hermes-profile=other"],
        ["-dlhermes-tmp-storage=tmpfs"],
        ["--label-file", "/tmp/labels"],
        ["--label-file=/tmp/labels"],
    ],
)
def test_extra_args_cannot_override_reserved_reuse_labels(monkeypatch, extra_args):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="reserved Hermes labels"):
        _make_dummy_env(extra_args=extra_args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--tmpfs", "/tmp:size=1g"],
        ["--tmpfs=/tmp:size=1g"],
        ["--volume", "scratch:/tmp"],
        ["--volume=scratch:/tmp/cache"],
        ["-vscratch:/tmp"],
        ["-itvscratch:/tmp/cache"],
        ["-Pvscratch:/tmp/cache"],
        ["--mount", "type=tmpfs,target=/tmp"],
        ["--mount=type=volume,source=scratch,destination=/tmp/cache"],
        ['--mount=type=volume,source=scratch,"target=/tmp"'],
        ["--mount=type=volume,source=scratch,target=/x/../tmp/cache"],
        ["--mount=type=volume,source=scratch,TARGET=/tmp/cache"],
        ["--volume=scratch:/x/../tmp"],
        ["--tmpfs=//tmp/cache:size=1g"],
        ["--volumes-from", "tmp-donor"],
        ["--volumes-from=tmp-donor"],
    ],
)
def test_extra_args_cannot_override_tmp_storage_policy(monkeypatch, extra_args):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="use docker_tmp_storage"):
        _make_dummy_env(extra_args=extra_args)


@pytest.mark.parametrize("volume", ["scratch:/tmp", "/host/cache:/tmp/cache:ro"])
def test_docker_volumes_cannot_override_tmp_storage_policy(monkeypatch, volume):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="use docker_tmp_storage"):
        _make_dummy_env(volumes=[volume])


def test_attached_env_value_is_not_misparsed_as_volume(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(extra_args=["-eFOOv=/tmp/cache"], persist_across_processes=False)

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    assert "-eFOOv=/tmp/cache" in run_args


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-v", "/host/review:/workspace:ro"],
        ["-Pvscratch:/workspace/cache:ro"],
        ["--mount=type=bind,source=/host/review,target=/workspace"],
        ["--mount=type=bind,source=/host/review,target=/x/../workspace"],
        ["--mount=type=bind,source=/host/review,Destination=/workspace"],
        ["--volume=/host/review://workspace:ro"],
        ["--volume", "/workspace/cache"],
        ["-dv/workspace/cache"],
        ["--tmpfs=/x/../workspace/cache:size=1g"],
        ["--volumes-from", "workspace-donor"],
    ],
)
def test_extra_args_cannot_mount_workspace_without_reuse_fingerprint(
    monkeypatch, extra_args
):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="container reuse isolation"):
        _make_dummy_env(extra_args=extra_args)


@pytest.mark.parametrize("network_arg", [["--network=host"], ["--net", "bridge"]])
def test_network_disabled_rejects_extra_arg_override(monkeypatch, network_arg):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="docker_network=false"):
        _make_dummy_env(network=False, extra_args=network_arg)


def test_network_enabled_preserves_explicit_network_mode(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(network=True, extra_args=["--network=host"])

    run_cmd = next(cmd for cmd, _ in calls if isinstance(cmd, list) and cmd[1] == "run")
    assert "--network=host" in run_cmd


def test_explicit_workspace_mount_participates_in_reuse_fingerprint(
    monkeypatch, tmp_path
):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(volumes=[f"{review_dir}:/workspace:ro"])

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    workspace_label = next(
        arg.split("=", 1)[1]
        for arg in run_args
        if arg.startswith("hermes-workspace=")
    )
    assert workspace_label != "off"
    reuse_probe = next(c[0] for c in calls if c[0][1:3] == ["ps", "-a"])
    assert f"label=hermes-workspace={workspace_label}" in reuse_probe


def test_managed_workspace_policy_participates_in_reuse_fingerprint(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env()

    reuse_probe = next(c[0] for c in calls if c[0][1:3] == ["ps", "-a"])
    assert "label=hermes-workspace=managed-ephemeral" in reuse_probe


@pytest.mark.parametrize(
    "volume",
    ["review-source:/workspace:ro", r"C:\review:/workspace:ro"],
)
def test_read_only_workspace_requires_authenticatable_bind_source(monkeypatch, volume):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="authenticated host bind"):
        _make_dummy_env(volumes=[volume])


def test_env_file_contents_participate_in_reuse_fingerprint(monkeypatch, tmp_path):
    env_file = tmp_path / "container.env"
    env_file.write_text("VALUE=first\n", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(extra_args=["--env-file", str(env_file)])
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    env_file.write_text("VALUE=second\n", encoding="utf-8")
    _make_dummy_env(extra_args=["--env-file", str(env_file)])
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_mutable_image_identity_participates_in_reuse_fingerprint(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    identities = iter(["sha256:first", "sha256:second"])
    monkeypatch.setattr(docker_env, "_resolve_image_identity", lambda *_: next(identities))

    _make_dummy_env(image="reviewer:latest")
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))
    _make_dummy_env(image="reviewer:latest")
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_container_starts_from_authenticated_immutable_image(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env,
        "_resolve_image_identity",
        lambda *_: "sha256:authenticated-image",
    )
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(image="reviewer:latest", persist_across_processes=False)

    run_args = next(call[0] for call in calls if call[0][1] == "run")
    assert "sha256:authenticated-image" in run_args
    assert "reviewer:latest" not in run_args


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"image": "python:3.11"}, {"image": "python:3.12"}),
        ({"extra_args": ["--shm-size=64m"]}, {"extra_args": ["--shm-size=1g"]}),
        ({"persistent_filesystem": False}, {"persistent_filesystem": True}),
        ({"env": {"TOKEN": "before"}}, {"env": {"TOKEN": "after"}}),
    ],
)
def test_complete_container_policy_participates_in_reuse_fingerprint(
    monkeypatch, first, second
):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = _mock_subprocess_run(monkeypatch)
    _make_dummy_env(**first)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    _make_dummy_env(**second)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_read_only_workspace_content_participates_in_policy(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    candidate = project_dir / "candidate.txt"
    candidate.write_text("first candidate", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    options = {
        "cwd": "/workspace",
        "host_cwd": str(project_dir),
        "auto_mount_cwd": True,
        "cwd_mount_mode": "ro",
    }
    _make_dummy_env(**options)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    candidate.write_text("second candidate", encoding="utf-8")
    _make_dummy_env(**options)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_read_only_workspace_executable_mode_participates_in_policy(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    candidate = project_dir / "candidate.sh"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    options = {
        "cwd": "/workspace",
        "host_cwd": str(project_dir),
        "auto_mount_cwd": True,
        "cwd_mount_mode": "ro",
    }

    _make_dummy_env(**options)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    candidate.chmod(0o755)
    _make_dummy_env(**options)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_read_only_workspace_digest_failure_is_fail_closed(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env,
        "_readonly_tree_digest",
        lambda _root: (_ for _ in ()).throw(OSError("unreadable")),
    )
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="cannot authenticate read-only workspace"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


@pytest.mark.parametrize("node_kind", ["fifo", "socket"])
def test_read_only_workspace_rejects_unsupported_nodes(tmp_path, node_kind):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    node = project_dir / node_kind
    if node_kind == "fifo":
        os.mkfifo(node)
    else:
        import socket

        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(node))
        sock.close()

    with pytest.raises(ValueError, match="unsupported filesystem node"):
        docker_env._readonly_tree_digest(project_dir)


def test_container_workspace_digest_rejects_unsupported_nodes(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert "unsupported filesystem node" in cmd[-2]
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="unsupported filesystem node: /workspace/fifo"
        )

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="unsupported filesystem node"):
        docker_env._container_tree_digest("docker", "container", "/workspace")


def test_git_workspace_provenance_rejects_fabricated_head(tmp_path):
    project_dir = tmp_path / "review-target"
    git_dir = project_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("1" * 40 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="commit object is missing or invalid"):
        docker_env._verify_git_workspace_provenance(project_dir)


def test_git_workspace_provenance_requires_exact_clean_commit_tree(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    tracked = project_dir / "tracked.txt"
    tracked.write_text("assigned bytes\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)

    docker_env._verify_git_workspace_provenance(project_dir)

    tracked.write_text("candidate replacement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked files do not match HEAD"):
        docker_env._verify_git_workspace_provenance(project_dir)

    subprocess.run(["git", "checkout", "--", "tracked.txt"], cwd=project_dir, check=True)
    (project_dir / ".gitignore").write_text(".hidden\n", encoding="utf-8")
    (project_dir / ".hidden").write_text("candidate bytes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked files"):
        docker_env._verify_git_workspace_provenance(project_dir)


def test_git_workspace_provenance_rejects_forged_loose_object(tmp_path):
    import zlib

    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    payload = project_dir / "payload.txt"
    payload.write_bytes(b"assigned bytes\n")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    blob_id = subprocess.run(
        ["git", "rev-parse", "HEAD:payload.txt"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    forged = b"candidate-controlled bytes\n"
    loose_object = project_dir / ".git" / "objects" / blob_id[:2] / blob_id[2:]
    loose_object.chmod(0o644)
    loose_object.write_bytes(zlib.compress(b"blob " + str(len(forged)).encode() + b"\0" + forged))
    payload.write_bytes(forged)

    with pytest.raises(ValueError, match="loose Git object failed independent validation"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


def test_git_workspace_provenance_rejects_candidate_replacement_refs(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    payload = project_dir / "payload.txt"
    payload.write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    payload.write_text("candidate replacement\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "replacement"], cwd=project_dir, check=True)
    replacement = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", assigned], cwd=project_dir, check=True)
    subprocess.run(["git", "replace", assigned, replacement], cwd=project_dir, check=True)
    subprocess.run(["git", "pack-refs", "--all", "--prune"], cwd=project_dir, check=True)
    payload.write_text("assigned\n", encoding="utf-8")

    with pytest.raises(ValueError, match="replacement refs"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


def test_git_workspace_provenance_does_not_trust_candidate_index_stat_cache(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    payload = project_dir / "payload.txt"
    payload.write_text("SAFE-CONTENT\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    original_stat = payload.stat()
    subprocess.run(
        ["git", "config", "core.trustctime", "false"], cwd=project_dir, check=True
    )
    subprocess.run(
        ["git", "config", "core.checkStat", "minimal"], cwd=project_dir, check=True
    )
    payload.write_text("EVIL-CONTENT\n", encoding="utf-8")
    os.utime(payload, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(ValueError, match="tracked files do not match HEAD"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


@pytest.mark.parametrize("metadata", ["info/grafts", "shallow"])
def test_git_workspace_provenance_rejects_candidate_history_metadata(
    tmp_path, metadata
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    (project_dir / "payload.txt").write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata_path = project_dir / ".git" / metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(f"{assigned}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate-selected Git history metadata"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


def test_git_workspace_provenance_rejects_incomplete_ancestry(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    payload = project_dir / "payload.txt"
    payload.write_text("parent\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "parent"], cwd=project_dir, check=True)
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload.write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    (project_dir / ".git" / "objects" / parent[:2] / parent[2:]).unlink()

    with pytest.raises(ValueError, match="history is incomplete or invalid"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


def test_git_workspace_provenance_requires_out_of_band_assigned_sha(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    (project_dir / "payload.txt").write_text("tree\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "other"], cwd=project_dir, check=True)

    with pytest.raises(ValueError, match="assigned SHA"):
        docker_env._verify_git_workspace_provenance(project_dir, "1" * 40)


def test_pinned_workspace_archive_rebuilds_trusted_git_metadata(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "review-branch"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"], cwd=project_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    payload = project_dir / "payload.txt"
    payload.write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "config", "diff.hostile.external", "/bin/true"],
        cwd=project_dir,
        check=True,
    )
    (project_dir / ".git" / "hooks" / "status").write_text(
        "#!/bin/sh\nexit 99\n", encoding="utf-8"
    )

    archive, _digest = docker_env._readonly_workspace_archive(
        project_dir,
        docker_env._readonly_tree_metadata_digest(project_dir),
        assigned,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as handle:
        handle.extractall(extracted, filter="data")

    config = (extracted / ".git" / "config").read_text(encoding="utf-8")
    assert "hostile" not in config
    assert not (extracted / ".git" / "hooks").exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=extracted, check=True,
        capture_output=True, text=True,
    ).stdout == ""
    assert (extracted / ".git" / "HEAD").read_text(encoding="ascii").strip() == assigned
    assert subprocess.run(
        ["git", "symbolic-ref", "--quiet", "HEAD"], cwd=extracted,
        capture_output=True, text=True,
    ).returncode == 1


def test_trusted_git_objects_exclude_candidate_semantic_caches(tmp_path):
    repository = tmp_path / "repository"
    destination = tmp_path / "destination"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "review@test.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Review Test"], cwd=repository, check=True)
    (repository / "payload.txt").write_text("packed object\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "packed"], cwd=repository, check=True)
    subprocess.run(["git", "gc", "--prune=now"], cwd=repository, check=True)

    source = repository / ".git" / "objects"
    packed = source / "pack"
    info = source / "info"
    pack_path = next(packed.glob("pack-*.pack"))
    pack_id = pack_path.stem.removeprefix("pack-")
    candidate_index = packed / f"pack-{pack_id}.idx"
    candidate_index.chmod(0o644)
    candidate_index.write_bytes(b"candidate-selected-index")
    (packed / f"pack-{pack_id}.bitmap").write_bytes(b"candidate-bitmap")
    (packed / "multi-pack-index").write_bytes(b"candidate-midx")

    docker_env._copy_trusted_git_objects(source, destination)

    copied_pack = destination / "pack" / pack_path.name
    copied_index = copied_pack.with_suffix(".idx")
    assert copied_pack.read_bytes() == pack_path.read_bytes()
    assert copied_index.read_bytes() != b"candidate-selected-index"
    subprocess.run(
        ["git", "verify-pack", str(copied_index)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    assert not (destination / "pack" / f"pack-{pack_id}.bitmap").exists()
    assert not (destination / "pack" / "multi-pack-index").exists()
    assert not (destination / "info").exists()


def test_git_workspace_provenance_ignores_candidate_pack_index(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "review@test.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Review Test"], cwd=repository, check=True)
    (repository / "payload.txt").write_text("assigned packed tree\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=repository, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "gc", "--prune=now"], cwd=repository, check=True)
    candidate_index = next((repository / ".git" / "objects" / "pack").glob("pack-*.idx"))
    candidate_index.chmod(0o644)
    candidate_index.write_bytes(b"candidate-selected-index")

    docker_env._verify_git_workspace_provenance(repository, assigned)


def test_git_workspace_provenance_disables_lazy_fetch(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"], cwd=project_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    (project_dir / "payload.txt").write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    original_run = subprocess.run
    environments = []

    def recording_run(*args, **kwargs):
        environments.append(kwargs.get("env"))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", recording_run)

    docker_env._verify_git_workspace_provenance(project_dir, assigned)

    assert environments
    assert all(env is not None and env.get("GIT_NO_LAZY_FETCH") == "1" for env in environments)


def test_git_workspace_provenance_rejects_untracked_empty_directory(tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@test.invalid"], cwd=project_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Review Test"], cwd=project_dir, check=True
    )
    (project_dir / "payload.txt").write_text("assigned\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "assigned"], cwd=project_dir, check=True)
    assigned = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    (project_dir / "untracked-empty-dir").mkdir()

    with pytest.raises(ValueError, match="untracked directories"):
        docker_env._verify_git_workspace_provenance(project_dir, assigned)


def test_materialize_reviewer_updates_expected_mounted_digest(monkeypatch, tmp_path):
    calls = []
    content = "f" * 64
    expected: dict[str, object] = {
        "tree_metadata_sha256": "m",
        "mounted_content_sha256": "original",
    }
    monkeypatch.setattr(
        docker_env,
        "_readonly_workspace_archive",
        lambda *_args, **_kwargs: (b"archive", content),
    )
    monkeypatch.setattr(docker_env, "_container_tree_digest", lambda *_args: content)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["volume", "inspect"]:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")
        if command[1:3] == ["run", "--rm"]:
            return subprocess.CompletedProcess(command, 0, stdout=(content + "\n").encode(), stderr=b"")
        if command[1:3] == ["run", "-d"]:
            return subprocess.CompletedProcess(command, 0, stdout="verifier\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)

    docker_env._materialize_readonly_workspace(
        "docker", "image", str(tmp_path), expected, "1" * 40, disposable=True
    )

    assert expected["mounted_content_sha256"] == content


def test_materialize_reviewer_overrides_image_entrypoint(monkeypatch, tmp_path):
    """Helper containers must not execute an image-selected entrypoint."""
    calls = []
    content = "f" * 64
    expected: dict[str, object] = {
        "tree_metadata_sha256": "m",
        "mounted_content_sha256": "original",
    }
    monkeypatch.setattr(
        docker_env,
        "_readonly_workspace_archive",
        lambda *_args, **_kwargs: (b"archive", content),
    )
    monkeypatch.setattr(docker_env, "_container_tree_digest", lambda *_args: content)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["volume", "inspect"]:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")
        if command[1:3] == ["run", "--rm"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=(content + "\n").encode(), stderr=b""
            )
        if command[1:3] == ["run", "-d"]:
            return subprocess.CompletedProcess(command, 0, stdout="verifier\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)

    docker_env._materialize_readonly_workspace(
        "docker", "entrypoint-image", str(tmp_path), expected, "1" * 40, disposable=True
    )

    populate = next(command for command in calls if command[1:3] == ["run", "--rm"])
    verifier = next(command for command in calls if command[1:3] == ["run", "-d"])
    assert populate[populate.index("--entrypoint") + 1] == "python3"
    assert populate[populate.index("entrypoint-image") + 1 :][:2] == ["-I", "-c"]
    assert verifier[verifier.index("--entrypoint") + 1] == "sleep"
    assert verifier[verifier.index("entrypoint-image") + 1 :] == ["120"]


def test_materialize_reviewer_volume_is_removed_when_verifier_start_fails(
    monkeypatch, tmp_path
):
    calls = []

    monkeypatch.setattr(
        docker_env,
        "_readonly_workspace_archive",
        lambda *_args, **_kwargs: (b"archive", "f" * 64),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["volume", "inspect"]:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")
        if command[1:3] == ["volume", "create"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1:3] == ["run", "--rm"]:
            return subprocess.CompletedProcess(command, 0, stdout=("f" * 64 + "\n").encode(), stderr=b"")
        if command[1:3] == ["run", "-d"]:
            raise subprocess.TimeoutExpired(command, 120)
        if command[1:4] == ["volume", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)

    expected: dict[str, object] = {
        "tree_metadata_sha256": "m",
        "mounted_content_sha256": "original",
    }
    with pytest.raises(subprocess.TimeoutExpired):
        docker_env._materialize_readonly_workspace(
            "docker", "image", str(tmp_path), expected,
            "1" * 40, disposable=True,
        )

    assert any(command[1:4] == ["volume", "rm", "-f"] for command in calls)
    assert expected["mounted_content_sha256"] == "original"


def test_assigned_sha_requires_git_metadata(tmp_path):
    project_dir = tmp_path / "not-a-repository"
    project_dir.mkdir()

    with pytest.raises(ValueError, match="missing Git metadata"):
        docker_env._verify_git_workspace_provenance(project_dir, "1" * 40)


def test_loose_git_object_rejects_oversized_expansion(monkeypatch, tmp_path):
    import hashlib
    import zlib

    canonical = b"blob 32\0" + b"x" * 32
    object_id = hashlib.sha1(canonical).hexdigest()
    loose = tmp_path / object_id
    loose.write_bytes(zlib.compress(canonical))
    monkeypatch.setattr(docker_env, "_MAX_REVIEW_GIT_OBJECT_BYTES", 16)

    with pytest.raises(ValueError, match="exceeds reviewer size limit"):
        docker_env._validated_loose_git_object_bytes(loose, object_id)


def test_reviewer_workspace_rejects_too_many_nodes(monkeypatch, tmp_path):
    (tmp_path / "one").write_text("1", encoding="utf-8")
    (tmp_path / "two").write_text("2", encoding="utf-8")
    monkeypatch.setattr(docker_env, "_MAX_REVIEW_WORKSPACE_NODES", 1)

    with pytest.raises(ValueError, match="node limit"):
        docker_env._enforce_reviewer_workspace_bounds(tmp_path)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "docker_network=false"),
        ({"network": False, "forward_env": ["TOKEN"]}, "environment variables"),
        ({"network": False, "env": {"TOKEN": "secret"}}, "environment variables"),
        ({"network": False, "volumes": ["data:/data"]}, "configured Docker volumes"),
        ({"network": False, "extra_args": ["--read-only"]}, "raw Docker arguments"),
    ],
)
def test_assigned_reviewer_workspace_rejects_network_and_host_inputs(options, message):
    with pytest.raises(ValueError, match=message):
        _make_dummy_env(expected_git_sha="1" * 40, **options)


def test_assigned_reviewer_workspace_omits_automatic_host_data(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: pytest.fail("reviewer must not load credential mounts"),
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: pytest.fail("reviewer must not load skill mounts"),
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: pytest.fail("reviewer must not load cache mounts"),
    )
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: pytest.fail("reviewer must not load egress credentials"),
    )
    monkeypatch.setattr("tools.env_passthrough.get_all_passthrough", lambda: {"REVIEW_SECRET"})
    monkeypatch.setenv("REVIEW_SECRET", "credential-from-host")

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_mount_mode="ro",
        network=False,
        persistent_filesystem=True,
        expected_git_sha="1" * 40,
        persist_across_processes=True,
    )

    run_args = [call[0] for call in calls if call[0][1] == "run"][-1]
    assert "--network=none" in run_args
    assert run_args[run_args.index("-w") + 1] == "/tmp"
    assert any(arg.endswith(":/workspace:ro") for arg in run_args)
    assert not any(arg.startswith("/tmp:") for arg in run_args)
    assert "/root:rw,exec,size=1g" in run_args
    assert not any(
        run_args[index] == "-v" and run_args[index + 1].endswith(":/root")
        for index in range(len(run_args) - 1)
    )
    assert not any(call[0][1:3] == ["ps", "-a"] for call in calls)
    assert "REVIEW_SECRET" not in repr(calls)
    assert "credential-from-host" not in repr(calls)
    copy_call = next(
        call[0]
        for call in calls
        if call[0][1:3] == ["exec", "fake-container-id"]
        and "/tmp/review" in " ".join(call[0])
    )
    assert "shutil.rmtree(target)" in copy_call[-2]
    assert "source.iterdir()" in copy_call[-2]
    assert copy_call[-1] == "1" * 40


def test_assigned_reviewer_rejects_unexpected_image_writable_volume(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    original_run = docker_env.subprocess.run

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout='[{"Destination":"/scratch","RW":true,"Type":"volume"}]\n',
                stderr="",
            )
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="unexpected writable mount.*scratch"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
            network=False,
            expected_git_sha="1" * 40,
        )

    rejected_rm = [
        call[0] for call in calls
        if isinstance(call[0], list) and call[0][1:3] == ["rm", "-f"]
    ]
    assert rejected_rm and all("-v" in cmd for cmd in rejected_rm)


def test_reviewer_cleanup_removes_snapshot_volume(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = "review-container"
    env._persist_across_processes = False
    env._persistent = False
    env._reviewer_mode = True
    env._docker_exe = "/usr/bin/docker"
    env._workspace_dir = None
    env._home_dir = None
    env._snapshot_volumes = ["hermes-ro-review-snapshot"]

    env.cleanup()

    commands = [call[0] for call in calls if isinstance(call[0], list)]
    assert ["/usr/bin/docker", "rm", "-f", "-v", "review-container"] in commands
    assert [
        "/usr/bin/docker", "volume", "rm", "-f", "hermes-ro-review-snapshot"
    ] in commands


def test_reviewer_init_session_failure_removes_container_and_snapshot(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "init_session",
        lambda _self: (_ for _ in ()).throw(RuntimeError("injected init failure")),
    )

    with pytest.raises(RuntimeError, match="injected init failure"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
            network=False,
            expected_git_sha="1" * 40,
        )

    commands = [call[0] for call in calls if isinstance(call[0], list)]
    assert ["/usr/bin/docker", "rm", "-f", "-v", "fake-container-id"] in commands
    assert any(command[1:4] == ["volume", "rm", "-f"] for command in commands)


def test_cleanup_without_container_still_removes_snapshot_volume(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = None
    env._persistent = False
    env._docker_exe = "/usr/bin/docker"
    env._workspace_dir = None
    env._home_dir = None
    env._snapshot_volumes = ["hermes-ro-unattached-snapshot"]

    env.cleanup()

    commands = [call[0] for call in calls if isinstance(call[0], list)]
    assert [
        "/usr/bin/docker", "volume", "rm", "-f", "hermes-ro-unattached-snapshot"
    ] in commands


def test_general_read_only_workspace_retains_live_host_bind(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("authenticated", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_mount_mode="ro",
    )

    run_args = [call[0] for call in calls if call[0][1] == "run"][-1]
    workspace_spec = next(
        run_args[index + 1]
        for index, arg in enumerate(run_args[:-1])
        if arg == "-v" and run_args[index + 1].endswith(":/workspace:ro")
    )
    assert workspace_spec == f"{project_dir}:/workspace:ro"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-v", "/host/source:/data:ro"],
        ["--volume=/host/source:/data"],
        ["--mount", "type=bind,source=/host/source,target=/data"],
        ["--mount=type=bind,src=/host/source,dst=/data"],
        ["--mount=source=/host/source,target=/data,type=bind"],
    ],
)
def test_extra_args_reject_untracked_host_bind_sources(monkeypatch, extra_args):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="host bind sources"):
        _make_dummy_env(extra_args=extra_args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--security-opt", "seccomp=/host/profile.json"],
        ["--security-opt=seccomp=relative-profile.json"],
    ],
)
def test_extra_args_reject_host_backed_security_opt_files(monkeypatch, extra_args):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="host-backed security-opt"):
        _make_dummy_env(extra_args=extra_args)


def test_read_only_workspace_mutation_during_authentication_is_fail_closed(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    candidate = project_dir / "candidate.txt"
    candidate.write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    original_digest = docker_env._readonly_tree_digest

    def _mutating_digest(root, **kwargs):
        result = original_digest(root, **kwargs)
        candidate.write_text("changed during authentication", encoding="utf-8")
        return result

    monkeypatch.setattr(docker_env, "_readonly_tree_digest", _mutating_digest)
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="changed during authentication"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


def test_mounted_workspace_content_must_match_authenticated_source(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env,
        "_container_tree_digest",
        lambda *_, **__: "digest-from-swapped-mounted-tree",
        raising=False,
    )
    calls = _mock_subprocess_run(monkeypatch)

    with pytest.raises(RuntimeError, match="mounted read-only workspace differs"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
            network=False,
            tmp_storage="disk",
            expected_git_sha="1" * 40,
            persist_across_processes=False,
        )

    assert ["/usr/bin/docker", "rm", "-f", "-v", "fake-container-id"] in [
        call[0] for call in calls
    ]


def test_read_only_workspace_snapshot_ignores_later_host_change(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    candidate = project_dir / "candidate.txt"
    candidate.write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_mount_mode="ro",
    )

    candidate.write_text("changed after mount", encoding="utf-8")
    monkeypatch.setattr(
        docker_env.BaseEnvironment,
        "execute",
        lambda *args, **kwargs: {"output": "", "returncode": 0},
    )
    result = env.execute("true")

    assert result["returncode"] == 0


def test_host_mutation_during_command_cannot_change_snapshot(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    candidate = project_dir / "candidate.txt"
    candidate.write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_mount_mode="ro",
        persist_across_processes=False,
    )

    def _execute(*args, **kwargs):
        candidate.write_text("changed during command", encoding="utf-8")
        return {"output": "apparently successful", "returncode": 0}

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _execute)
    result = env.execute("inspect candidate")

    assert result == {"output": "apparently successful", "returncode": 0}


def test_read_only_workspace_resolves_packed_git_head(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    git_dir = project_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/review\n", encoding="utf-8")
    first_sha = "1" * 40
    second_sha = "2" * 40
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{first_sha} refs/heads/review\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    options = {
        "cwd": "/workspace",
        "host_cwd": str(project_dir),
        "auto_mount_cwd": True,
        "cwd_mount_mode": "ro",
    }

    _make_dummy_env(**options)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{second_sha} refs/heads/review\n",
        encoding="utf-8",
    )
    _make_dummy_env(**options)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_read_only_workspace_git_metadata_participates_in_policy(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    git_dir = project_dir / ".git"
    replace_dir = git_dir / "refs" / "replace"
    replace_dir.mkdir(parents=True)
    assigned_sha = "1" * 40
    (git_dir / "HEAD").write_text(assigned_sha + "\n", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    options = {
        "cwd": "/workspace",
        "host_cwd": str(project_dir),
        "auto_mount_cwd": True,
        "cwd_mount_mode": "ro",
    }

    _make_dummy_env(**options)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    (replace_dir / assigned_sha).write_text("2" * 40 + "\n", encoding="utf-8")
    _make_dummy_env(**options)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_linked_workspace_external_git_metadata_is_rejected(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    common_git_dir = tmp_path / "repository.git"
    linked_git_dir = common_git_dir / "worktrees" / "review-target"
    linked_git_dir.mkdir(parents=True)
    assigned_sha = "1" * 40
    (project_dir / ".git").write_text(
        f"gitdir: {linked_git_dir}\n", encoding="utf-8"
    )
    (linked_git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (linked_git_dir / "HEAD").write_text(assigned_sha + "\n", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="external Git metadata"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


def test_read_only_workspace_git_symlink_is_rejected(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    external_git_dir = tmp_path / "candidate-selected.git"
    external_git_dir.mkdir()
    (external_git_dir / "HEAD").write_text("1" * 40 + "\n", encoding="utf-8")
    (project_dir / ".git").symlink_to(external_git_dir, target_is_directory=True)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="external Git metadata"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


@pytest.mark.parametrize("metadata_path", ["HEAD", "packed-refs", "refs/heads/review"])
def test_read_only_workspace_nested_git_symlink_is_rejected(
    monkeypatch, tmp_path, metadata_path
):
    project_dir = tmp_path / "review-target"
    git_dir = project_dir / ".git"
    target = tmp_path / "candidate-selected-metadata"
    target.write_text("1" * 40 + "\n", encoding="utf-8")
    link = git_dir / metadata_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    if metadata_path != "HEAD":
        (git_dir / "HEAD").write_text("ref: refs/heads/review\n", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="external Git metadata"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


def test_read_only_workspace_git_commondir_is_rejected(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    git_dir = project_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("1" * 40 + "\n", encoding="utf-8")
    (git_dir / "commondir").write_text(str(tmp_path) + "\n", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="external Git metadata"):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
        )


def test_container_workspace_digest_uses_isolated_python(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="a" * 64 + "\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)

    assert docker_env._container_tree_digest("docker", "container", "/workspace") == (
        "a" * 64
    )
    assert len(calls) == 1
    assert calls[0][:9] == [
        "docker", "exec", "-w", "/", "container", "python3", "-I", "-c", calls[0][8]
    ]
    assert calls[0][-1] == "/workspace"


def test_read_only_workspace_submount_is_snapshotted(monkeypatch, tmp_path):
    component_dir = tmp_path / "component"
    component_dir.mkdir()
    candidate = component_dir / "candidate.txt"
    candidate.write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(volumes=[f"{component_dir}:/workspace/component:ro"])

    candidate.write_text("changed after mount", encoding="utf-8")
    monkeypatch.setattr(
        docker_env.BaseEnvironment,
        "execute",
        lambda *args, **kwargs: {"output": "", "returncode": 0},
    )
    result = env.execute("true")

    assert result["returncode"] == 0


def test_mapped_read_only_workspace_uses_canonical_source_identity(monkeypatch, tmp_path):
    container_root = tmp_path / "container-data"
    project_dir = container_root / "review-target"
    project_dir.mkdir(parents=True)
    candidate = project_dir / "candidate.txt"
    candidate.write_text("first candidate", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(
        docker_env,
        "_container_tree_digest",
        lambda *_, **__: docker_env._readonly_tree_digest(
            project_dir, include_root_mode=False
        ),
    )

    options = {
        "cwd": "/workspace",
        "host_cwd": str(project_dir),
        "auto_mount_cwd": True,
        "cwd_mount_mode": "ro",
        "cwd_path_mappings": {str(container_root): "/docker-host/data"},
    }
    _make_dummy_env(**options)
    first_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    first_label = next(arg for arg in first_run if arg.startswith("hermes-policy="))

    candidate.write_text("second candidate", encoding="utf-8")
    _make_dummy_env(**options)
    second_run = [call[0] for call in calls if call[0][1] == "run"][-1]
    second_label = next(arg for arg in second_run if arg.startswith("hermes-policy="))

    assert first_label != second_label


def test_auto_mount_host_cwd_read_only_adds_ro_volume(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_mount_mode="ro",
    )

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    assert f"{project_dir}:/workspace:ro" in run_args
    reuse_probe = next(c[0] for c in calls if c[0][1:3] == ["ps", "-a"])
    assert any(arg.startswith("label=hermes-workspace=") for arg in reuse_probe)


def test_auto_mount_translates_container_path_to_docker_host_path(monkeypatch, tmp_path):
    container_root = tmp_path / "container-data"
    project_dir = container_root / "tasks" / "review-target"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        cwd_path_mappings={str(container_root): "/home/ubuntu/.hermes"},
    )

    run_args = next(c[0] for c in calls if c[0][1] == "run")
    assert "/home/ubuntu/.hermes/tasks/review-target:/workspace" in run_args
    assert f"{project_dir}:/workspace" not in run_args


def test_auto_mount_rejects_cwd_outside_allowed_roots(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    rejected = tmp_path / "rejected"
    allowed.mkdir()
    rejected.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="allowed workspace root"):
        _make_dummy_env(
            host_cwd=str(rejected),
            auto_mount_cwd=True,
            cwd_allowed_roots=[str(allowed)],
        )


def test_auto_mount_rejects_symlink_escape_from_allowed_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    escaped = allowed / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="allowed workspace root"):
        _make_dummy_env(
            host_cwd=str(escaped),
            auto_mount_cwd=True,
            cwd_allowed_roots=[str(allowed)],
        )


def test_auto_mount_rejects_cwd_without_required_path_mapping(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    unrelated = tmp_path / "unrelated"
    project_dir.mkdir()
    unrelated.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="path mapping"):
        _make_dummy_env(
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_path_mappings={str(unrelated): "/docker-host/unrelated"},
        )


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-v", "/tmp/override:/workspace"],
        ["--volume=/tmp/override:/workspace/subdir:ro"],
        ["--mount", "type=bind,src=/tmp/override,dst=/workspace"],
        ["--mount=type=bind,src=/tmp/override,target=/workspace/subdir"],
    ],
)
def test_auto_mount_rejects_extra_args_workspace_mounts(monkeypatch, tmp_path, extra_args):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    with pytest.raises(ValueError, match="docker_extra_args mounts /workspace"):
        _make_dummy_env(
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
            extra_args=extra_args,
        )


@pytest.mark.parametrize(
    "volume",
    [
        "/somewhere/else:/workspace:ro",
        "/somewhere/else:/workspace/generated",
        r"C:\review-target:/workspace:ro",
    ],
)
def test_auto_mount_rejects_explicit_workspace_volume(monkeypatch, tmp_path, volume):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(ValueError, match="docker_volumes.*workspace"):
        _make_dummy_env(
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            volumes=[volume],
        )


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When persist_across_processes=false, cleanup() must docker stop AND
    docker rm so containers don't leak across hermes processes.

    Updated for issue #20561: the previous implementation used fire-and-forget
    ``subprocess.Popen("... &", shell=True)`` which raced with parent exit;
    the new implementation uses ``subprocess.run`` on a daemon thread with
    bounded timeouts. See test_cleanup_with_persist_disabled_stops_and_rms
    for the full behavior contract.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    # Run the worker thread synchronously so assertions can observe its work.
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)

    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="ephemeral-task", persistent_filesystem=False,
        persist_across_processes=False,
    )
    container_id = env._container_id
    assert container_id

    # Capture cleanup-time docker calls (everything before this was init).
    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capture(cmd, **kw):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kw))
        return real_run(cmd, **kw)

    monkeypatch.setattr(docker_env.subprocess, "run", _capture)
    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and c[0][1:2] == ["stop"]]
    assert stops, f"cleanup() should docker stop {container_id}; got {cleanup_calls}"


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode


def _make_execute_only_env(forward_env=None):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.cwd = "/root"
    env.timeout = 60
    env._forward_env = forward_env or []
    env._env = {}
    env._prepare_command = lambda command: (command, None)
    env._timeout_result = lambda timeout: {"output": f"timed out after {timeout}", "returncode": 124}
    env._container_id = "test-container"
    env._docker_exe = "/usr/bin/docker"
    # Base class attributes needed by unified execute()
    env._session_id = "test123"
    env._snapshot_path = "/tmp/hermes-snap-test123.sh"
    env._cwd_file = "/tmp/hermes-cwd-test123.txt"
    env._cwd_marker = "__HERMES_CWD_test123__"
    env._snapshot_ready = True
    env._last_sync_time = None
    env._init_env_args = []
    return env


def test_init_env_args_uses_hermes_dotenv_for_allowlisted_env(monkeypatch):
    """_build_init_env_args picks up forwarded env vars from .env file at init time."""
    # Use a var that is NOT in _HERMES_PROVIDER_ENV_BLOCKLIST (GITHUB_TOKEN
    # is in the copilot provider's api_key_env_vars and gets stripped).
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_dotenv" in args_str


def test_init_env_args_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    """Shell env vars take priority over .env file values in init env args."""
    env = _make_execute_only_env(["DATABASE_URL"])

    monkeypatch.setenv("DATABASE_URL", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "value_from_dotenv"})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "DATABASE_URL=value_from_shell" in args_str
    assert "value_from_dotenv" not in args_str


def test_init_env_args_uses_hermes_dotenv_for_empty_shell_env(monkeypatch):
    """A transient empty-string in the live env must fall back to .env, not win.

    Regression: the disk fallback used to fire only on `value is None`, so a
    present-but-empty `MY_SECRET=""` skipped it and was forwarded as `-e
    MY_SECRET=`, clobbering the correct value sitting in ~/.hermes/.env.
    """
    env = _make_execute_only_env(["MY_SECRET"])

    monkeypatch.setenv("MY_SECRET", "")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"MY_SECRET": "value_from_dotenv"})

    args = env._build_init_env_args()

    # Assert on the resolved value, not the printed -e flag: the disk value
    # must win and a blank "MY_SECRET=" flag must never be emitted.
    assert "MY_SECRET=value_from_dotenv" in args
    assert "MY_SECRET=" not in args


# ── docker_env tests ──────────────────────────────────────────────


def test_docker_env_appears_in_run_command(monkeypatch):
    """Explicit docker_env values should be passed via -e at docker run time."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock", "GNUPGHOME": "/root/.gnupg"})

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]
    run_args_str = " ".join(run_args)
    assert "SSH_AUTH_SOCK=/run/user/1000/ssh-agent.sock" in run_args_str
    assert "GNUPGHOME=/root/.gnupg" in run_args_str


def _node_options_from_run(calls):
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    args = run_calls[0][0]
    for i, a in enumerate(args):
        if a == "-e" and i + 1 < len(args) and args[i + 1].startswith("NODE_OPTIONS="):
            return args[i + 1].split("=", 1)[1]
    return None


def test_egress_node_options_overrides_conflicting_ca_flag(monkeypatch):
    """maxpetrusenko P1: a conflicting docker_env NODE_OPTIONS CA-mode flag
    (--use-bundled-ca) must be replaced by the egress-required --use-openssl-ca,
    not left to survive alongside it (final Node trust would depend on order)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env, "_egress_proxy_args_for_docker",
        lambda: ([], {"_HERMES_EGRESS_NODE_OPTIONS_APPEND": "--use-openssl-ca"}, []),
    )
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(env={"NODE_OPTIONS": "--max-old-space-size=8192 --use-bundled-ca"})

    node_opts = (_node_options_from_run(calls) or "").split()
    assert "--use-openssl-ca" in node_opts, "egress CA flag must be present"
    assert "--use-bundled-ca" not in node_opts, "conflicting CA flag must be stripped"
    # Operator's unrelated tuning must be preserved.
    assert "--max-old-space-size=8192" in node_opts


def test_forward_env_overrides_docker_env_in_init_args(monkeypatch):
    """docker_forward_env should override docker_env for the same key."""
    env = _make_execute_only_env(forward_env=["MY_KEY"])
    env._env = {"MY_KEY": "static_value"}

    monkeypatch.setenv("MY_KEY", "dynamic_value")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})

    args = env._build_init_env_args()
    args_str = " ".join(args)

    assert "MY_KEY=dynamic_value" in args_str
    assert "MY_KEY=static_value" not in args_str


def test_normalize_env_dict_filters_invalid_keys():
    """_normalize_env_dict should reject invalid variable names."""
    result = docker_env._normalize_env_dict({
        "VALID_KEY": "ok",
        "123bad": "rejected",
        "": "rejected",
        "also valid": "rejected",  # spaces invalid
        "GOOD": "ok",
    })
    assert result == {"VALID_KEY": "ok", "GOOD": "ok"}


def test_security_args_include_setuid_setgid_for_privdrop(monkeypatch):
    """The default (run_as_host_user=False) invocation must include SETUID and
    SETGID caps so the image's init can drop from root to a non-root user
    (e.g. via ``s6-setuidgid`` in the bundled Hermes image, or ``gosu``/``su``
    in user-provided images).

    Without these caps the privilege-drop helper fails with
    ``operation not permitted`` and the container exits immediately (exit 1)
    before running any work.

    ``no-new-privileges`` is kept, so the dropped process still cannot
    escalate back to root after the drop — the drop is a one-way transition
    performed before the ``no_new_privs`` bit is enforced on the exec boundary.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env()

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    added = {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--cap-add"
    }
    assert "SETUID" in added, "SETUID cap missing — image privilege-drop will fail"
    assert "SETGID" in added, "SETGID cap missing — image privilege-drop will fail"


# ── run_as_host_user tests ────────────────────────────────────────


def test_run_as_host_user_passes_uid_gid(monkeypatch):
    """With run_as_host_user=True, --user <uid>:<gid> is added to docker run."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(docker_env.os, "getgid", lambda: 5678, raising=False)
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(run_as_host_user=True)

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    # --user must be present and must be paired with "1234:5678"
    assert "--user" in run_args, f"--user flag missing from docker run args: {run_args}"
    idx = run_args.index("--user")
    assert run_args[idx + 1] == "1234:5678", (
        f"expected --user 1234:5678, got --user {run_args[idx + 1]}"
    )


def test_run_as_host_user_drops_setuid_setgid_caps(monkeypatch):
    """When --user is passed, the container already starts unprivileged and
    never needs a privilege drop, so SETUID/SETGID caps are omitted for a
    tighter security posture."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(docker_env.os, "getgid", lambda: 1000, raising=False)
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(run_as_host_user=True)

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    run_args = run_calls[0][0]

    added = {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--cap-add"
    }
    assert "SETUID" not in added, (
        "SETUID cap should be dropped when running as host user — no privilege drop is needed"
    )
    assert "SETGID" not in added, (
        "SETGID cap should be dropped when running as host user — no privilege drop is needed"
    )
    # Core non-privilege-drop caps must still be there (pip/npm/apt need them).
    assert "DAC_OVERRIDE" in added
    assert "CHOWN" in added
    assert "FOWNER" in added


# ── Docker labels (issue #20561) ──────────────────────────────────


def _run_args_from_calls(calls):
    """Pull the argv list passed to the first ``docker run`` invocation."""
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    return run_calls[0][0]


def _labels_in_run_args(run_args):
    """Return the set of ``key=value`` strings passed via ``--label``."""
    return {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--label"
    }


def test_run_command_tags_hermes_agent_label(monkeypatch):
    """Every container hermes-agent starts must carry the hermes-agent=1 label
    so the orphan reaper (and external operators) can identify them with a
    single ``docker ps --filter label=hermes-agent=1`` call. Regression test
    for issue #20561 — without the label there is no global sweep target."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(task_id="my-task")

    labels = _labels_in_run_args(_run_args_from_calls(calls))
    assert "hermes-agent=1" in labels, (
        f"hermes-agent=1 label missing; got labels: {sorted(labels)}"
    )


def test_label_sanitizer_rejects_invalid_characters_without_aliasing():
    """Docker label values must be safe, bounded, and collision-resistant."""
    assert docker_env._sanitize_label_value("plain-name_1.0") == "plain-name_1.0"

    slash = docker_env._sanitize_label_value("with/slash")
    colon = docker_env._sanitize_label_value("with:slash")
    assert slash.startswith("with_slash-")
    assert colon.startswith("with_slash-")
    assert slash != colon

    unicode_value = docker_env._sanitize_label_value("emoji-😀-here")
    assert unicode_value.startswith("emoji-_-here-")
    assert all(character.isascii() and (character.isalnum() or character in "_.-") for character in unicode_value)

    # Empty / non-string inputs must collapse to a queryable token, not "".
    assert docker_env._sanitize_label_value("") == "unknown"
    assert docker_env._sanitize_label_value(None) == "unknown"  # type: ignore[arg-type]
    # >63 chars must remain bounded while differing inputs remain distinguishable.
    long_value = "x" * 100
    sanitized_long = docker_env._sanitize_label_value(long_value)
    assert len(sanitized_long) == 63
    assert sanitized_long != docker_env._sanitize_label_value("x" * 99 + "y")


def test_run_command_sanitizes_unsafe_task_id(monkeypatch):
    """A task_id containing characters Docker rejects in label values must be
    sanitized before reaching ``docker run --label``; otherwise the daemon
    refuses the run with an inscrutable error and the agent's first command
    blows up."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(task_id="task/with:weird*chars")

    labels = _labels_in_run_args(_run_args_from_calls(calls))
    task_labels = [label for label in labels if label.startswith("hermes-task-id=")]
    assert len(task_labels) == 1
    assert task_labels[0].startswith("hermes-task-id=task_with_weird_chars-")


def test_labels_attribute_populated_after_init(monkeypatch):
    """``self._labels`` must be set to the same key/value pairs that went onto
    docker run, so subsequent reuse / reaper paths can match without re-running
    the sanitizer or re-importing the profile module."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)

    env = _make_dummy_env(task_id="abc")

    assert env._labels == {
        "hermes-agent": "1",
        "hermes-task-id": "abc",
        "hermes-profile": "default",
        "hermes-egress": "off",
        "hermes-workspace": "managed-ephemeral",
        "hermes-tmp-storage": "tmpfs",
        "hermes-policy": env._labels["hermes-policy"],
    }
    assert len(env._labels["hermes-policy"]) == 24


# ── Cross-process container reuse (issue #20561) ──────────────────


def _mock_subprocess_run_with_reuse(monkeypatch, ps_state: str | None,
                                     start_succeeds: bool = True):
    """Reuse-aware subprocess.run mock.

    ``ps_state`` controls what ``docker ps -a --filter ...`` returns:
      * ``None`` → no match (empty stdout). Forces a fresh ``docker run``.
      * ``"running"`` / ``"exited"`` / ... → emit ``CID\\tSTATE`` so the reuse
        path picks it up. ``"running"`` skips ``docker start``; other states
        trigger ``docker start`` (which can be forced to fail via
        ``start_succeeds=False``).

    Returns the captured call list so the test can verify which docker
    commands actually ran.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                if ps_state is None:
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                # 3-field format: ID, State, EgressLabel.  When egress_label
                # is "off" the code parses all three fields; <no value> means
                # the container has no egress label, which is acceptable.
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout=f"reused-cid\t{ps_state}\t<no value>\n",
                    stderr="",
                )
            if sub == "start":
                if not start_succeeds:
                    # Real subprocess.run with check=True raises on non-zero exit;
                    # mirror that so the production code's except clause fires.
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such container")
                return subprocess.CompletedProcess(cmd, 0, stdout="reused-cid\n", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
            if sub == "image" and cmd[2] == "inspect":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if sub == "exec" and cmd[-3:-1] == ["-f", "--"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{cmd[-1]}\n", stderr="")
            if sub == "inspect" and "{{.HostConfig.NetworkMode}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="bridge\n", stderr="")
            if sub == "inspect" and "{{json .Mounts}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_reuse_attaches_to_running_container_without_docker_run(monkeypatch):
    """When a labeled container is already ``running``, the reuse probe
    must pick it up and skip ``docker run`` entirely. Regression for the
    issue #20561 root cause: every Hermes process spawning a new container
    despite docs claiming "ONE long-lived container shared across sessions"."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    calls = _mock_subprocess_run_with_reuse(monkeypatch, ps_state="running")

    env = _make_dummy_env(task_id="reuse-test")

    # The reuse path must populate _container_id from the ps probe output.
    assert env._container_id == "reused-cid", (
        f"expected reused container id, got {env._container_id!r}"
    )
    # And it must NOT have run `docker run`.
    run_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert not run_invocations, (
        f"docker run should be skipped on reuse, got: {run_invocations}"
    )
    # And it must have NOT issued a `docker start` for an already-running container.
    start_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "start"]
    assert not start_invocations, (
        f"docker start should be skipped when container already running, got: {start_invocations}"
    )


def test_egress_enabled_does_not_reuse_pre_egress_container(monkeypatch):
    """A container created before egress was enabled lacks the proxy env vars
    and CA mount.  Reusing it would silently bypass the credential firewall."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            ["-v", "/tmp/ca:/etc/ssl/certs/hermes-egress-ca.crt:ro"],
            {"HTTPS_PROXY": "http://host.docker.internal:9090"},
            ["--add-host", "host.docker.internal:host-gateway"],
        ),
    )
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "ps":
                # Simulate an old pre-egress container: without the egress label
                # filter it would match; with the filter Docker returns no match.
                assert any(str(part).startswith("label=hermes-egress=") for part in cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
            if sub == "image" and cmd[2] == "inspect":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if sub == "exec" and cmd[-3:-1] == ["-f", "--"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{cmd[-1]}\n", stderr="")
            if sub == "inspect" and "{{.HostConfig.NetworkMode}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="bridge\n", stderr="")
            if sub == "inspect" and "{{json .Mounts}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="reuse-egress")

    assert env._container_id == "fresh-cid"
    run_invocations = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_invocations, "egress-enabled containers require a fresh docker run"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-e", "HTTPS_PROXY="],
        ["-eHTTPS_PROXY="],
        ["-eOPENROUTER_API_KEY"],
        ["-deOPENROUTER_API_KEY"],
    ],
)
def test_extra_args_proxy_override_refuses_under_egress(monkeypatch, extra_args):
    """docker_extra_args are appended after Hermes args, so egress enforcement
    must reject critical overrides before Docker sees them."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            [],
            {
                "HTTPS_PROXY": "http://host.docker.internal:9090",
                "OPENROUTER_API_KEY": "proxy-token",
            },
            [],
        ),
    )
    _mock_subprocess_run(monkeypatch)

    with pytest.raises(RuntimeError, match="docker_extra_args"):
        _make_dummy_env(extra_args=extra_args)


def test_reuse_starts_stopped_container_before_attaching(monkeypatch):
    """A labeled container in ``exited`` state must be restarted via
    ``docker start`` before the new Hermes process uses it. Without this
    step, ``docker exec`` against a stopped container errors out and the
    first agent command fails opaquely."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    calls = _mock_subprocess_run_with_reuse(monkeypatch, ps_state="exited")

    env = _make_dummy_env(task_id="reuse-stopped")

    assert env._container_id == "reused-cid"
    start_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "start"]
    assert start_invocations, "expected docker start for exited container"
    run_invocations = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert not run_invocations, "should not docker run when reusing an exited container"


def test_failed_docker_run_cleans_up_orphaned_container(monkeypatch):
    """When ``docker run`` fails (e.g. exit 125), the partially-created
    container must be removed by name.

    Docker can create the container object before failing to start it,
    leaving a stale ``Created`` container. The exited-only orphan reaper
    (``reap_orphan_containers``, ``status=exited``) never catches a
    ``Created`` orphan, so without this cleanup it leaks permanently.
    Regression for #7439. Salvage of #7440 (@Tranquil-Flow).
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    cleanup_calls = []

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "image":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if sub == "ps":
                # No reusable container -> fall through to a fresh `docker run`.
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                raise subprocess.CalledProcessError(
                    125, cmd, output="", stderr="docker: Error response from daemon"
                )
            if sub == "rm":
                cleanup_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(subprocess.CalledProcessError):
        _make_dummy_env()

    assert len(cleanup_calls) == 1, "docker rm should be called once for the orphaned container"
    rm_cmd = cleanup_calls[0]
    assert rm_cmd[1] == "rm" and rm_cmd[2] == "-f"
    assert rm_cmd[3].startswith("hermes-"), "should remove the container by its generated name"


def test_failed_reviewer_docker_run_removes_snapshot_volume(monkeypatch, tmp_path):
    project_dir = tmp_path / "review-target"
    project_dir.mkdir()
    (project_dir / "candidate.txt").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "reviewer")
    monkeypatch.setattr(
        docker_env,
        "_materialize_readonly_workspace",
        lambda *args, **kwargs: "hermes-ro-failed-review",
    )
    removed_volumes = []

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "image":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if sub == "run":
                raise subprocess.CalledProcessError(125, cmd, stderr="start failed")
            if sub == "volume" and cmd[2:4] == ["rm", "-f"]:
                removed_volumes.append(cmd[4])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env._cgroup_limits_ok = True

    with pytest.raises(subprocess.CalledProcessError):
        _make_dummy_env(
            cwd="/workspace",
            host_cwd=str(project_dir),
            auto_mount_cwd=True,
            cwd_mount_mode="ro",
            network=False,
            expected_git_sha="1" * 40,
        )

    assert removed_volumes == ["hermes-ro-failed-review"]


def test_docker_run_timeout_cleans_up_orphaned_container(monkeypatch):
    """When ``docker run`` times out (e.g. slow image pull), the
    partially-created container must be removed. Salvage of #7440
    (@Tranquil-Flow); regression for #7439.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    cleanup_calls = []

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if sub == "image":
                return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
            if sub == "ps":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if sub == "run":
                raise subprocess.TimeoutExpired(cmd, 120)
            if sub == "rm":
                cleanup_calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    with pytest.raises(subprocess.TimeoutExpired):
        _make_dummy_env()

    assert len(cleanup_calls) == 1, "docker rm should be called once for the orphaned container"
    rm_cmd = cleanup_calls[0]
    assert rm_cmd[1] == "rm" and rm_cmd[2] == "-f"
    assert rm_cmd[3].startswith("hermes-"), "should remove the container by its generated name"


def test_find_reusable_handles_empty_label_string(monkeypatch):
    """Docker CLI v29.5.3 returns an empty string (NOT ``<no value>``)
    for absent labels.  The trailing tab produces ``cid\\trunning\\t\\n``;
    we must not strip the trailing tab or the three-field parser drops the
    container.  Regression test for the egilewski review on #48073."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            if cmd[1] == "ps":
                # Docker v29.5.3: absent label → empty string, trailing tab
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="safe-cid\trunning\t\n",
                    stderr="",
                )
            if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{cmd[-1]}\n", stderr="")
            if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_dummy_env(task_id="empty-label")
    assert env._container_id == "safe-cid", (
        f"container with empty-string label should be reused, got {env._container_id!r}"
    )


# ── Cleanup correctness (issue #20561) ────────────────────────────


class _FakeThread:
    """Stand-in for threading.Thread that captures target/args and calls
    target() synchronously when .start() runs, so cleanup behavior is
    observable without actually backgrounding subprocess calls."""

    def __init__(self, target=None, daemon=None, name=None):
        self._target = target
        self.daemon = daemon
        self.name = name
        self._done = False

    def start(self):
        if self._target is not None:
            self._target()
        self._done = True

    def is_alive(self):
        return not self._done

    def join(self, timeout=None):
        self._done = True


def _install_fake_thread(monkeypatch):
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)


def test_cleanup_with_persist_is_noop_for_container(monkeypatch):
    """``persist_across_processes=True`` (default) cleanup must NEITHER stop
    NOR remove the container — the docs promise "ONE long-lived container
    shared across sessions", and any docker stop would kill background
    processes inside the container (npm watchers, pytest watchers, etc.).

    Resource reclamation in this mode happens via the orphan reaper on next
    Hermes startup, not on graceful exit. Issue #20561 — the first iteration
    of this PR did docker stop here, which Ben caught as contradicting the
    "ONE long-lived container" semantics."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    env = _make_dummy_env(task_id="cleanup-persist", persistent_filesystem=False)
    # Default persist_across_processes=True.
    container_id = env._container_id
    assert container_id

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert not stops, (
        f"docker stop must NOT be called when persist_across_processes=True; "
        f"container has to stay running so background processes survive. "
        f"Got: {stops}"
    )
    assert not rms, (
        f"docker rm must NOT be called when persist_across_processes=True; "
        f"reuse would be impossible. Got: {rms}"
    )
    # The in-process handle must still be cleared so the next __init__
    # re-probes via labels (and reuses the still-running container).
    assert env._container_id is None, (
        "in-process container_id should be cleared even in no-op cleanup"
    )


def test_cleanup_vm_default_honors_persist_mode(monkeypatch):
    """``cleanup_vm(task_id)`` without ``force_remove=True`` must be a no-op
    for a persist-mode container.

    Regression for the bug Ben caught after commit 4: ``AIAgent.close()``
    (which is called from ``tui_gateway/server.py`` on session.close, from
    ``gateway/run.py`` on per-session teardown, and from per-turn cleanup)
    calls ``cleanup_vm(task_id)``. If that defaulted to ``force_remove=True``
    we'd tear down the container on every TUI session close, defeating the
    "ONE long-lived container shared across sessions" contract.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    from tools import terminal_tool

    env = _make_dummy_env(task_id="session-close-test")
    container_id = env._container_id
    terminal_tool._active_environments["session-close-test"] = env

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    try:
        terminal_tool.cleanup_vm("session-close-test")
    finally:
        terminal_tool._active_environments.pop("session-close-test", None)

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert not stops, (
        f"cleanup_vm() default must not docker stop a persist-mode container; "
        f"got: {stops}"
    )
    assert not rms, (
        f"cleanup_vm() default must not docker rm a persist-mode container; "
        f"got: {rms}"
    )


def test_cleanup_with_persist_disabled_stops_and_rms(monkeypatch):
    """``persist_across_processes=False`` cleanup must docker stop AND docker
    rm so containers don't leak. Crucially, this runs regardless of the
    ``persistent_filesystem`` setting — the original code only rm'd when
    ``not self._persistent``, which meant the default-on ``container_persistent:
    true`` users (the documented happy path) leaked Exited containers forever.
    Issue #20561 root-cause fix."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    # Note: persistent_filesystem=True (the prior-leak scenario) + the new
    # cross-process toggle OFF must still result in a clean rm.
    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="cleanup-no-persist", persistent_filesystem=True,
        persist_across_processes=False,
    )

    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capturing_run(cmd, **kwargs):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", _capturing_run)

    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "stop"]
    rms = [c for c in cleanup_calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "rm"]
    assert stops, "expected docker stop"
    assert rms, (
        "docker rm MUST run when persist_across_processes=False, even with "
        "persistent_filesystem=True — that gating was the leak source in #20561."
    )


def test_cleanup_uses_subprocess_run_not_detached_shell(monkeypatch):
    """The pre-fix code used ``subprocess.Popen("... &", shell=True)`` which
    raced with parent-process exit and silently dropped cleanup work. The
    new code must use ``subprocess.run`` with bounded ``timeout=`` so the
    work actually completes within the process lifetime.

    Asserts cleanup never reaches into shell-mode Popen. Uses
    ``force_remove=True`` so cleanup actually issues docker calls — the
    default persist-mode path is now a no-op (commit 4) and would trivially
    pass this assertion without exercising the docker code at all.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    _install_fake_thread(monkeypatch)

    def _forbidden_popen(*args, **kwargs):
        raise AssertionError(
            f"cleanup must not use subprocess.Popen anymore (issue #20561); "
            f"got args={args} kwargs={kwargs}"
        )

    monkeypatch.setattr(docker_env.subprocess, "Popen", _forbidden_popen)

    env = _make_dummy_env(task_id="no-popen-cleanup")
    env.cleanup(force_remove=True)  # must not raise


def test_cleanup_on_env_with_no_container_id_does_not_raise(monkeypatch):
    """A DockerEnvironment whose ``__init__`` failed before the container_id
    was set (image-pull error, docker daemon down) should still be safe to
    cleanup() — the post-creation failure path in callers always tries.
    Without this guard the daemon-down case used to NameError on the cleanup
    branch."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = None
    env._persistent = False
    env._workspace_dir = None
    env._home_dir = None
    # No exception expected.
    env.cleanup()


# ── Orphan reaper (issue #20561) ──────────────────────────────────


def _now_iso(offset_seconds: int = 0) -> str:
    """Return an RFC3339 timestamp ``offset_seconds`` in the past."""
    import datetime
    t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=offset_seconds)
    # Format like Docker emits — with nanoseconds-style trailing digits.
    return t.isoformat().replace("+00:00", ".123456789Z")


def _reaper_run_mock(monkeypatch, ps_ids: list[str], inspect_responses: dict[str, str],
                      rm_succeeds: bool = True):
    """Build a subprocess.run mock for reaper tests.

    * ``ps_ids`` — what ``docker ps -a --filter ... --format '{{.ID}}'`` returns
    * ``inspect_responses[cid]`` — what ``docker inspect ... FinishedAt`` returns
      for each cid; ``""`` means "field unset".
    * ``rm_succeeds`` — whether ``docker rm -f`` returns 0.

    Captures every call so tests can assert which containers were rm'd.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        sub = cmd[1]
        if sub == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="\n".join(ps_ids) + ("\n" if ps_ids else ""), stderr="",
            )
        if sub == "inspect":
            # cmd is [docker, inspect, --format, '{{.State.FinishedAt}}', cid]
            cid = cmd[-1]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=inspect_responses.get(cid, "") + "\n", stderr="",
            )
        if sub == "rm":
            return subprocess.CompletedProcess(
                cmd, 0 if rm_succeeds else 1,
                stdout="", stderr="" if rm_succeeds else "no such container",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_reap_orphan_returns_zero_when_no_matches(monkeypatch):
    """No labeled containers → no rm calls, returns 0. Establishes the
    happy-path baseline for the orphan reaper (issue #20561)."""
    calls = _reaper_run_mock(monkeypatch, ps_ids=[], inspect_responses={})

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    assert removed == 0
    rms = [c for c in calls if isinstance(c[0], list) and c[0][1:2] == ["rm"]]
    assert not rms, "no rm calls expected when ps returns empty"


def test_reap_orphan_continues_after_individual_rm_failure(monkeypatch):
    """If ``docker rm -f`` fails on one container (already removed by a
    concurrent process, container locked, etc.), the reaper must log and
    continue to the next candidate rather than aborting the whole sweep."""
    old = _now_iso(offset_seconds=900)
    rm_calls = []

    def _run(cmd, **kwargs):
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        sub = cmd[1]
        if sub == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="cid-a\ncid-b\ncid-c\n", stderr="",
            )
        if sub == "inspect":
            return subprocess.CompletedProcess(cmd, 0, stdout=old + "\n", stderr="")
        if sub == "rm":
            rm_calls.append(cmd[-1])
            # cid-b fails; cid-a and cid-c succeed.
            if cmd[-1] == "cid-b":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such container")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker",
    )

    # All three were attempted, two succeeded.
    assert removed == 2
    assert set(rm_calls) == {"cid-a", "cid-b", "cid-c"}, (
        f"reaper must attempt all candidates even when one fails; got: {rm_calls}"
    )


def test_container_finished_at_parses_nanosecond_timestamp(monkeypatch):
    """Docker emits FinishedAt with nanosecond precision (RFC3339 with up to
    9 fractional digits), but Python's fromisoformat caps at microseconds.
    The helper must trim the extra digits without raising — otherwise every
    candidate gets skipped and the reaper does nothing."""

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout="2026-05-28T13:45:00.123456789Z\n",
            stderr="",
        )

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    result = docker_env._container_finished_at("/usr/bin/docker", "test-cid")
    assert result is not None, "must parse RFC3339 with nanoseconds"
    import datetime
    assert result.tzinfo == datetime.timezone.utc
    assert result.year == 2026 and result.month == 5 and result.day == 28


def test_container_finished_at_returns_none_on_zero_value():
    """Docker's zero-value ``0001-01-01T00:00:00Z`` (never finished) must
    map to None so the reaper treats the container as unreapable."""
    # Direct test of the parsing helper — no subprocess needed since the
    # check happens after the inspect call returns.
    import subprocess as _subprocess

    class _MockRun:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    import unittest.mock
    with unittest.mock.patch.object(
        docker_env.subprocess, "run", return_value=_MockRun("0001-01-01T00:00:00Z\n"),
    ):
        result = docker_env._container_finished_at("/usr/bin/docker", "never-finished")
    assert result is None


def test_credential_mount_skipped_when_source_is_directory(monkeypatch, tmp_path, caplog):
    """Credential mount should be skipped when source path is a directory.

    In Docker-in-Docker scenarios, Docker may auto-create the source path as
    a directory when it doesn't exist on the host.  Mounting a directory over
    a file destination causes exit 125.
    """
    # Create a directory that looks like a corrupted credential file path
    corrupted_dir = tmp_path / "google_token.json"
    corrupted_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    # Mock get_credential_file_mounts to return the corrupted entry
    fake_mounts = [
        {"host_path": str(corrupted_dir), "container_path": "/root/.hermes/google_token.json"},
    ]
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: fake_mounts,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: [],
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: [],
    )

    with caplog.at_level(logging.WARNING):
        _make_dummy_env()

    # The corrupted mount should be skipped
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert "google_token.json" not in run_args_str

    # Should log a warning about the directory source
    assert any(
        "source is a directory" in rec.getMessage()
        for rec in caplog.records
    )


def test_credential_mount_skipped_when_source_missing(monkeypatch, tmp_path, caplog):
    """Credential mount should be skipped when source file no longer exists."""
    missing_path = tmp_path / "deleted_token.json"
    # Don't create the file — it's "missing"

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    fake_mounts = [
        {"host_path": str(missing_path), "container_path": "/root/.hermes/deleted_token.json"},
    ]
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        lambda: fake_mounts,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount",
        lambda: [],
    )
    monkeypatch.setattr(
        "tools.credential_files.get_cache_directory_mounts",
        lambda: [],
    )

    with caplog.at_level(logging.WARNING):
        _make_dummy_env()

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert "deleted_token.json" not in run_args_str

    assert any(
        "source not found" in rec.getMessage()
        for rec in caplog.records
    )


# ── s6-overlay /init image handling (issue #34628) ────────────────


def _mock_subprocess_run_with_entrypoint(monkeypatch, entrypoint_json):
    """Like _mock_subprocess_run, but `docker image inspect` returns the given
    entrypoint JSON so _image_uses_init_entrypoint can be exercised end-to-end.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "image" and len(cmd) >= 3 and cmd[2] == "inspect":
                if "{{.Id}}" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout=entrypoint_json + "\n", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
            if cmd[1] == "exec" and cmd[-3:-1] == ["-f", "--"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{cmd[-1]}\n", stderr="")
            if cmd[1] == "inspect" and "{{json .Mounts}}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_s6_image_skips_docker_init_and_mounts_run_exec(monkeypatch):
    """For an s6-overlay /init image, docker run must omit --init and mount
    /run with exec (issue #34628)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run_with_entrypoint(monkeypatch, '["/init"]')

    _make_dummy_env(image="hermes-agent:latest")

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args = run_calls[0][0]

    assert "--init" not in run_args, "s6 /init image must not get Docker --init"

    tmpfs_vals = [run_args[i + 1] for i, a in enumerate(run_args[:-1]) if a == "--tmpfs"]
    run_mounts = [v for v in tmpfs_vals if v.startswith("/run:")]
    assert run_mounts, f"no /run tmpfs mount found in {tmpfs_vals}"
    assert "exec" in run_mounts[0] and "noexec" not in run_mounts[0], (
        f"/run must be mounted exec for s6 images, got: {run_mounts[0]}"
    )


# ---------------------------------------------------------------------------
# Out-of-band container removal recovery (issue #36266, PR #36631)
# ---------------------------------------------------------------------------


def test_execute_does_not_recover_when_not_persistent(monkeypatch):
    """A non-persistent session must NOT trigger container recreation on a
    "No such container" error — recovery is only meaningful for the persistent,
    cross-process container that can be removed out-of-band.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        persistent_filesystem=True,
        persist_across_processes=False,
    )

    def _fake_super_execute(self, command, cwd="", **kwargs):
        return {"output": "No such container: x", "returncode": 1}

    def _fail_recreate(self):
        pytest.fail("recreation must not run when persist_across_processes is False")

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _fake_super_execute)
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_recreate_container", _fail_recreate
    )

    result = env.execute("echo hi")
    assert result.get("returncode") == 1, "the original error must pass through unchanged"


def test_execute_does_not_recover_on_ordinary_failure(monkeypatch):
    """A genuine non-zero exit that is NOT a container-gone error must pass
    through without triggering recovery (guards against over-eager recreation).
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    _mock_subprocess_run(monkeypatch)
    env = _make_dummy_env(
        persistent_filesystem=True,
        persist_across_processes=True,
    )

    def _fake_super_execute(self, command, cwd="", **kwargs):
        return {"output": "bash: badcmd: command not found", "returncode": 127}

    def _fail_recreate(self):
        pytest.fail("recreation must not run for an ordinary command failure")

    monkeypatch.setattr(docker_env.BaseEnvironment, "execute", _fake_super_execute)
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_recreate_container", _fail_recreate
    )

    result = env.execute("badcmd")
    assert result.get("returncode") == 127
    assert "command not found" in result.get("output", "")
