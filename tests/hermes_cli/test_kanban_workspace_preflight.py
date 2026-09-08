from __future__ import annotations

from pathlib import Path
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_workspace_preflight import (
    WorkspaceCapability,
    _profile_config,
    _runtime_mount_probe,
    preflight_workspace_for_profile,
    profile_uses_restricted_reviewer_runtime,
    route_children_to_capable_profiles,
)


def _docker_config(root: Path, host_root: Path) -> dict:
    return {
        "terminal": {
            "backend": "docker",
            "docker_image": "python:3.11-slim",
            "docker_mount_cwd_to_workspace": True,
            "docker_cwd_mount_mode": "ro",
            "docker_cwd_allowed_roots": [str(root)],
            "docker_cwd_path_mappings": {str(root): str(host_root)},
            "docker_network": False,
            "docker_forward_env": [],
            "docker_env": {},
            "docker_volumes": [],
            "docker_extra_args": [],
        },
        "toolsets": ["hermes-cli"],
    }


def test_effective_profile_config_honors_managed_overlay_and_env_expansion(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "plain-worker"
    managed = tmp_path / "managed"
    profile_home.mkdir(parents=True)
    managed.mkdir()
    (profile_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n  docker_mount_cwd_to_workspace: false\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        "terminal:\n"
        "  reviewer_mode: true\n"
        "  backend: docker\n"
        "  docker_mount_cwd_to_workspace: true\n"
        "  docker_cwd_mount_mode: ro\n"
        "  docker_cwd_allowed_roots: ['${REVIEW_ROOT}']\n"
        "  docker_network: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("REVIEW_ROOT", str(tmp_path))
    from hermes_cli import managed_scope
    managed_scope.invalidate_managed_cache()

    config = _profile_config("plain-worker")

    assert config["terminal"]["backend"] == "docker"
    assert config["terminal"]["docker_mount_cwd_to_workspace"] is True
    assert config["terminal"]["docker_cwd_mount_mode"] == "ro"
    assert config["terminal"]["docker_cwd_allowed_roots"] == [str(tmp_path)]
    assert profile_uses_restricted_reviewer_runtime(
        "plain-worker", profile_config=config
    ) is True


def test_dynamic_workspace_is_translated_and_probed_read_only(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    workspace = run_root / "ticket-123" / "repo"
    workspace.mkdir(parents=True)
    host_root = Path("/srv/hermes/runs")
    observed: dict[str, object] = {}

    def probe(**kwargs):
        observed.update(kwargs)

    result = preflight_workspace_for_profile(
        "reviewer",
        workspace,
        profile_config=_docker_config(run_root, host_root),
        runtime_probe=probe,
    )

    assert result.available is True
    assert result.read_only is True
    assert result.runtime_path == str(host_root / "ticket-123" / "repo")
    assert observed["docker_source"] == result.runtime_path
    assert observed["read_only"] is True
    assert observed["network_enabled"] is False


def test_mapping_miss_fails_closed_without_runtime_probe(tmp_path: Path) -> None:
    workspace = tmp_path / "other" / "repo"
    workspace.mkdir(parents=True)
    called = False

    def probe(**kwargs):
        nonlocal called
        called = True

    config = _docker_config(tmp_path / "runs", Path("/srv/hermes/runs"))
    config["terminal"]["docker_cwd_allowed_roots"] = [str(tmp_path)]
    result = preflight_workspace_for_profile(
        "reviewer", workspace, profile_config=config, runtime_probe=probe
    )

    assert result.available is False
    assert "no matching path mapping" in result.reason
    assert called is False


def test_missing_profile_fails_closed_instead_of_defaulting_to_local(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: False)

    result = preflight_workspace_for_profile("removed-reviewer", workspace)

    assert result.available is False
    assert "does not exist" in result.reason


def test_symlink_escape_fails_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)

    result = preflight_workspace_for_profile(
        "reviewer",
        allowed / "escape",
        profile_config=_docker_config(allowed, Path("/srv/allowed")),
        runtime_probe=lambda **_: None,
    )

    assert result.available is False
    assert "outside every allowed workspace root" in result.reason


def test_reviewer_writable_mount_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "runs" / "repo"
    workspace.mkdir(parents=True)
    config = _docker_config(tmp_path / "runs", Path("/srv/runs"))
    config["terminal"]["docker_cwd_mount_mode"] = "rw"

    result = preflight_workspace_for_profile(
        "security-reviewer",
        workspace,
        profile_config=config,
        runtime_probe=lambda **_: None,
    )

    assert result.available is False
    assert "read-only" in result.reason


def test_reviewer_requires_nonempty_canonical_workspace_allowlist(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = _docker_config(tmp_path, Path("/srv/runs"))
    config["terminal"]["docker_cwd_allowed_roots"] = []

    result = preflight_workspace_for_profile(
        "security-reviewer", workspace, profile_config=config,
        runtime_probe=lambda **_: None,
    )

    assert result.available is False
    assert "non-empty" in result.reason


def test_reviewer_mutating_tool_surface_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "runs" / "repo"
    workspace.mkdir(parents=True)
    config = _docker_config(tmp_path / "runs", Path("/srv/runs"))
    config["toolsets"] = ["terminal", "file", "kanban"]

    result = preflight_workspace_for_profile(
        "quality-reviewer",
        workspace,
        profile_config=config,
        runtime_probe=lambda **_: None,
    )

    assert result.available is False
    assert "mutating tool surface" in result.reason


def test_all_reviewers_unavailable_route_to_capable_fallback(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    children = [
        {"title": "functional", "assignee": "functional-reviewer", "parents": []},
        {"title": "security", "assignee": "security-reviewer", "parents": []},
    ]

    def capability(profile: str, _workspace: Path) -> WorkspaceCapability:
        return WorkspaceCapability(
            available=profile == "fallback-reviewer",
            profile=profile,
            workspace=str(workspace),
            reason="mount denied" if profile != "fallback-reviewer" else "",
            read_only=True,
            device=workspace.stat().st_dev,
            inode=workspace.stat().st_ino,
        )

    routed, failures = route_children_to_capable_profiles(
        children,
        workspace,
        fallback_profile="fallback-reviewer",
        capability_fn=capability,
    )

    assert [child["assignee"] for child in routed] == [
        "fallback-reviewer", "fallback-reviewer"
    ]
    assert {failure.profile for failure in failures} == {
        "functional-reviewer",
        "security-reviewer",
    }


def test_partial_reviewer_availability_preserves_capable_assignment(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    children = [
        {"title": "functional", "assignee": "functional-reviewer", "parents": []},
        {"title": "security", "assignee": "security-reviewer", "parents": [0]},
    ]

    def capability(profile: str, _workspace: Path) -> WorkspaceCapability:
        return WorkspaceCapability(
            available=profile in {"functional-reviewer", "fallback-reviewer"},
            profile=profile,
            workspace=str(workspace),
            reason="mount denied" if profile == "security-reviewer" else "",
            read_only="reviewer" in profile,
            device=workspace.stat().st_dev,
            inode=workspace.stat().st_ino,
        )

    routed, failures = route_children_to_capable_profiles(
        children,
        workspace,
        fallback_profile="fallback-reviewer",
        capability_fn=capability,
    )

    assert [child["assignee"] for child in routed] == [
        "functional-reviewer",
        "fallback-reviewer",
    ]
    assert len(failures) == 1


def test_reviewer_fallback_marks_child_for_durable_isolation(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    identity = workspace.stat()

    def capability(profile: str, _workspace: Path) -> WorkspaceCapability:
        return WorkspaceCapability(
            available=profile == "quality-reviewer",
            profile=profile,
            workspace=str(workspace),
            reason="unavailable" if profile != "quality-reviewer" else "",
            read_only=profile == "quality-reviewer",
            device=identity.st_dev,
            inode=identity.st_ino,
        )

    routed, _ = route_children_to_capable_profiles(
        [{"title": "review", "assignee": "implementer", "parents": []}],
        workspace,
        fallback_profile="quality-reviewer",
        capability_fn=capability,
    )

    assert routed[0]["assignee"] == "quality-reviewer"
    assert routed[0]["requires_reviewer_isolation"] is True


def test_no_capable_fallback_aborts_entire_route(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    children = [{"title": "review", "assignee": "reviewer", "parents": []}]

    def unavailable(profile: str, _workspace: Path) -> WorkspaceCapability:
        return WorkspaceCapability(
            available=False,
            profile=profile,
            workspace=str(workspace),
            reason="unavailable",
        )

    with pytest.raises(RuntimeError, match="fallback profile 'default'.*unavailable"):
        route_children_to_capable_profiles(
            children,
            workspace,
            fallback_profile="default",
            capability_fn=unavailable,
        )


def test_runtime_probe_is_networkless_read_only_and_credential_free(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    observed = {}

    monkeypatch.setattr("tools.environments.docker.find_docker", lambda: "/usr/bin/docker")

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    _runtime_mount_probe(
        docker_source=str(source),
        image="reviewer@sha256:deadbeef",
        expected_inode=source.stat().st_ino,
        read_only=True,
        network_enabled=False,
    )

    command = observed["command"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert f"{source}:/workspace:ro" in command
    assert observed["kwargs"]["stdin"] is subprocess.DEVNULL
    assert set(observed["kwargs"]["env"]) == {"PATH"}


def test_runtime_probe_supports_ordinary_writable_networked_docker_profile(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    observed = {}

    monkeypatch.setattr("tools.environments.docker.find_docker", lambda: "/usr/bin/docker")

    def run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    _runtime_mount_probe(
        docker_source=str(source),
        image="worker@sha256:deadbeef",
        expected_inode=source.stat().st_ino,
        read_only=False,
        network_enabled=True,
    )

    command = observed["command"]
    assert "--network=none" not in command
    assert f"{source}:/workspace:rw" in command
    assert "--read-only" not in command


def test_reviewer_preflight_enforces_workspace_file_bounds_before_probe(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "candidate.txt").write_text("candidate", encoding="utf-8")
    called = False

    def probe(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("tools.environments.docker._MAX_REVIEW_WORKSPACE_NODES", 0)
    result = preflight_workspace_for_profile(
        "quality-reviewer", workspace,
        profile_config=_docker_config(tmp_path, Path("/srv")),
        runtime_probe=probe,
    )

    assert result.available is False
    assert "node limit" in result.reason
    assert called is False


def test_cached_preflight_deduplicates_before_expensive_probe_and_publishes_fresh_ttl(
    tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_workspace_preflight as module

    workspace = tmp_path / "repo"
    workspace.mkdir()
    identity = workspace.stat()
    capability = WorkspaceCapability(
        True, "quality-reviewer", str(workspace.resolve()), read_only=True,
        device=identity.st_dev, inode=identity.st_ino,
    )
    module._probe_cache.clear()
    module._probe_inflight.clear()
    monkeypatch.setattr(module, "_profile_config", lambda _profile: {"terminal": {}})
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def expensive(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(2)
        return capability

    monkeypatch.setattr(module, "preflight_workspace_for_profile", expensive)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                module.cached_preflight_workspace_for_profile(
                    "quality-reviewer", workspace
                )
            )
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(2)
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(2)
    published_at = time.monotonic()

    assert calls == 1
    assert results == [capability, capability]
    assert module._probe_cache
    expiry = next(iter(module._probe_cache.values()))[0]
    assert expiry >= published_at + module._PROBE_CACHE_TTL_SECONDS - 0.5


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_DOCKER_WORKSPACE_CANARY") != "1",
    reason="opt-in real Docker runtime canary",
)
def test_real_docker_runtime_canary_rejects_workspace_write(tmp_path: Path) -> None:
    """Exercise the same real bind-mount canary used by the dispatcher."""
    source = tmp_path / "repo"
    source.mkdir()
    _runtime_mount_probe(
        docker_source=str(source),
        image=os.environ.get(
            "HERMES_DOCKER_CANARY_IMAGE",
            "nikolaik/python-nodejs:python3.11-nodejs20",
        ),
        expected_inode=source.stat().st_ino,
        read_only=True,
        network_enabled=False,
    )
    assert not list(source.glob(".hermes-workspace-preflight-*"))
