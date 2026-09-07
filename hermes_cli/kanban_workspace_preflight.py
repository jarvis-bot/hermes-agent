"""Per-profile workspace capability preflight for Kanban routing.

The decomposer and dispatcher use this module before assigning a task to a
profile.  In particular, Docker-backed reviewer profiles must prove that the
*exact* dynamically allocated workspace is visible as a read-only bind mount;
a profile directory or a successful historical canary is not sufficient.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class WorkspaceCapability:
    available: bool
    profile: str
    workspace: str
    reason: str = ""
    read_only: bool = False
    runtime_path: Optional[str] = None


_MUTATING_TOOLSETS = {
    "file",
    "files",
    "code",
    "code-execution",
    "computer",
    "computer-use",
    "browser",
    "cron",
    "delegation",
}


def _profile_config(profile: str) -> dict:
    import yaml

    from hermes_cli.profiles import resolve_profile_env

    home = Path(resolve_profile_env(profile))
    config_path = home / "config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"profile {profile!r} config is not an object")
    return loaded


def _is_reviewer_profile(profile: str, terminal: dict) -> bool:
    return "reviewer" in profile.lower() or terminal.get("docker_cwd_mount_mode") == "ro"


def profile_uses_restricted_reviewer_runtime(
    profile: str, *, profile_config: Optional[dict] = None
) -> bool:
    """Return whether worker startup must force the reviewer-safe surface.

    The name check is deliberate defense in depth: even a temporarily broken
    reviewer config must not make the next worker boot with hooks and the full
    interactive toolset before capability preflight gets another chance.
    """
    try:
        config = profile_config if profile_config is not None else _profile_config(profile)
        terminal = config.get("terminal") if isinstance(config, dict) else {}
        return _is_reviewer_profile(
            profile, terminal if isinstance(terminal, dict) else {}
        )
    except Exception:
        return "reviewer" in profile.lower()


def _validate_reviewer_policy(profile: str, config: dict, terminal: dict) -> Optional[str]:
    if terminal.get("backend", "local") != "docker":
        return "reviewer profile must use the Docker terminal backend"
    if terminal.get("docker_mount_cwd_to_workspace") is not True:
        return "reviewer profile must mount the assigned workspace"
    if terminal.get("docker_cwd_mount_mode") != "ro":
        return "reviewer workspace mount must be read-only"
    if terminal.get("docker_network", True) is not False:
        return "reviewer Docker network must be disabled"
    if terminal.get("docker_forward_env") or terminal.get("docker_env"):
        return "reviewer Docker runtime cannot receive environment credentials"
    if terminal.get("docker_volumes") or terminal.get("docker_extra_args"):
        return "reviewer Docker runtime cannot receive additional mounts or raw arguments"

    toolsets = config.get("toolsets") or []
    if not isinstance(toolsets, list):
        return "reviewer toolsets must be an explicit list"
    unsafe = sorted(
        str(tool).strip().lower()
        for tool in toolsets
        if str(tool).strip().lower() in _MUTATING_TOOLSETS
    )
    if unsafe:
        return f"reviewer mutating tool surface is enabled: {', '.join(unsafe)}"
    return None


def _runtime_mount_probe(
    *,
    docker_source: str,
    image: str,
    expected_inode: int,
    read_only: bool,
    network_enabled: bool,
    timeout: int = 30,
) -> None:
    """Run a bounded real Docker canary for one exact workspace mapping."""
    if not read_only:
        raise RuntimeError("workspace runtime probe requires a read-only mount")
    if network_enabled:
        raise RuntimeError("workspace runtime probe requires network-disabled reviewer policy")

    from tools.environments.docker import find_docker

    docker = find_docker()
    if not docker:
        raise RuntimeError("Docker executable is unavailable")
    sentinel = f".hermes-workspace-preflight-{uuid.uuid4().hex}"
    script = (
        'test -d /workspace && test -r /workspace && '
        'test "$(stat -c %i /workspace)" = "$1" && '
        'if touch "/workspace/$2" 2>/dev/null; then '
        'rm -f "/workspace/$2"; exit 44; else exit 0; fi'
    )
    command = [
        docker,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "-v",
        f"{docker_source}:/workspace:ro",
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        script,
        "hermes-preflight",
        str(expected_inode),
        sentinel,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker workspace canary failed to execute: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:400]
        if result.returncode == 44:
            detail = "workspace accepted a write despite read-only reviewer policy"
        raise RuntimeError(
            f"Docker workspace canary failed (exit {result.returncode})"
            + (f": {detail}" if detail else "")
        )
    if (Path(docker_source) / sentinel).exists():
        raise RuntimeError("Docker workspace canary left a source-side write")


def preflight_workspace_for_profile(
    profile: str,
    workspace: str | Path,
    *,
    profile_config: Optional[dict] = None,
    runtime_probe: Callable[..., None] = _runtime_mount_probe,
) -> WorkspaceCapability:
    """Prove that *profile* can safely access this exact workspace.

    Expected failures are returned as an unavailable capability so callers can
    route to a known-safe owner without creating or claiming a doomed task.
    """
    requested = Path(workspace).expanduser()
    try:
        canonical = requested.resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError("workspace is not a directory")
        config = profile_config if profile_config is not None else _profile_config(profile)
        if not isinstance(config, dict):
            raise ValueError("profile config is not an object")
        terminal = config.get("terminal") or {}
        if not isinstance(terminal, dict):
            raise ValueError("profile terminal config is not an object")
        backend = terminal.get("backend", "local")
        reviewer = _is_reviewer_profile(profile, terminal)
        if reviewer:
            policy_error = _validate_reviewer_policy(profile, config, terminal)
            if policy_error:
                raise ValueError(policy_error)
        if backend != "docker":
            if not os.access(canonical, os.R_OK | os.X_OK):
                raise ValueError("workspace is not readable by the local runtime")
            return WorkspaceCapability(
                True, profile, str(canonical), read_only=False, runtime_path=str(canonical)
            )
        if terminal.get("docker_mount_cwd_to_workspace") is not True:
            raise ValueError("Docker profile does not mount the assigned workspace")

        from tools.environments.docker import _resolve_cwd_mount_source

        canonical_source, docker_source = _resolve_cwd_mount_source(
            str(canonical),
            allowed_roots=terminal.get("docker_cwd_allowed_roots", []),
            path_mappings=terminal.get("docker_cwd_path_mappings", {}),
        )
        read_only = terminal.get("docker_cwd_mount_mode", "rw") == "ro"
        runtime_probe(
            docker_source=docker_source,
            image=terminal.get("docker_image")
            or "nikolaik/python-nodejs:python3.11-nodejs20",
            expected_inode=Path(canonical_source).stat().st_ino,
            read_only=read_only,
            network_enabled=bool(terminal.get("docker_network", True)),
        )
        return WorkspaceCapability(
            True,
            profile,
            canonical_source,
            read_only=read_only,
            runtime_path=docker_source,
        )
    except Exception as exc:
        return WorkspaceCapability(
            False,
            profile,
            str(requested),
            reason=str(exc) or type(exc).__name__,
        )


def route_children_to_capable_profiles(
    children: list[dict],
    workspace: str | Path,
    *,
    fallback_profile: str,
    capability_fn: Callable[[str, Path], WorkspaceCapability] = preflight_workspace_for_profile,
) -> tuple[list[dict], list[WorkspaceCapability]]:
    """Route children without ever retaining a known-incapable assignment."""
    workspace_path = Path(workspace)
    routed: list[dict] = []
    failures: list[WorkspaceCapability] = []
    cache: dict[str, WorkspaceCapability] = {}

    def capability(profile: str) -> WorkspaceCapability:
        if profile not in cache:
            cache[profile] = capability_fn(profile, workspace_path)
        return cache[profile]

    for child in children:
        selected = str(child.get("assignee") or fallback_profile)
        selected_capability = capability(selected)
        target = selected
        if not selected_capability.available:
            failures.append(selected_capability)
            fallback = capability(fallback_profile)
            if not fallback.available:
                raise RuntimeError(
                    f"selected profile {selected!r} cannot access workspace "
                    f"({selected_capability.reason}); fallback profile "
                    f"{fallback_profile!r} is unavailable ({fallback.reason})"
                )
            target = fallback_profile
        item = dict(child)
        item["assignee"] = target
        routed.append(item)
    return routed, failures
