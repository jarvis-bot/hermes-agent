"""Per-profile workspace capability preflight for Kanban routing.

The decomposer and dispatcher use this module before assigning a task to a
profile.  In particular, Docker-backed reviewer profiles must prove that the
*exact* dynamically allocated workspace is visible as a read-only bind mount;
a profile directory or a successful historical canary is not sufficient.
"""

from __future__ import annotations

import os
import hashlib
import json
import stat
import subprocess
import threading
import time
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
    device: Optional[int] = None
    inode: Optional[int] = None
    content_sha256: Optional[str] = None


_PROBE_CACHE_TTL_SECONDS = 30.0
_PROBE_CACHE_MAX_ENTRIES = 256
_probe_cache_lock = threading.Lock()
_probe_cache: dict[tuple[object, ...], tuple[float, WorkspaceCapability]] = {}
_probe_inflight: dict[tuple[object, ...], threading.Event] = {}


def workspace_capability_matches(
    capability: WorkspaceCapability, workspace: str | Path
) -> bool:
    """Return whether *workspace* is still the attested directory object.

    ``lstat`` semantics are intentional: replacing the probed directory with a
    symlink must not silently redirect a worker to a different object.
    """
    if (
        not capability.available
        or capability.device is None
        or capability.inode is None
    ):
        return False
    try:
        requested = Path(workspace).expanduser().resolve(strict=True)
        attested = Path(capability.workspace)
        current = os.stat(attested, follow_symlinks=False)
        return (
            requested == attested
            and stat.S_ISDIR(current.st_mode)
            and current.st_dev == capability.device
            and current.st_ino == capability.inode
        )
    except (OSError, RuntimeError):
        return False


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
    allowed_roots = terminal.get("docker_cwd_allowed_roots")
    if not isinstance(allowed_roots, list) or not allowed_roots:
        return "reviewer profiles require a non-empty docker_cwd_allowed_roots allowlist"
    for root in allowed_roots:
        if not isinstance(root, str) or not root.strip():
            return "reviewer workspace allowlist entries must be non-empty absolute paths"
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            return "reviewer workspace allowlist entries must be absolute paths"
        try:
            candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return "reviewer workspace allowlist entries must resolve to existing paths"
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
    from tools.environments.docker import find_docker

    docker = find_docker()
    if not docker:
        raise RuntimeError("Docker executable is unavailable")
    sentinel = f".hermes-workspace-preflight-{uuid.uuid4().hex}"
    access_probe = (
        'if touch "/workspace/$2" 2>/dev/null; then '
        'rm -f "/workspace/$2"; exit 44; else exit 0; fi'
        if read_only
        else 'touch "/workspace/$2" && rm -f "/workspace/$2"'
    )
    script = (
        'test -d /workspace && test -r /workspace && '
        'test "$(stat -c %i /workspace)" = "$1" && ' + access_probe
    )
    command = [docker, "run", "--rm"]
    if not network_enabled:
        command.append("--network=none")
    if read_only:
        command.append("--read-only")
    command.extend([
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "-v",
        f"{docker_source}:/workspace" + (":ro" if read_only else ":rw"),
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        script,
        "hermes-preflight",
        str(expected_inode),
        sentinel,
    ])
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
        initial = os.stat(canonical, follow_symlinks=False)
        if not stat.S_ISDIR(initial.st_mode):
            raise ValueError("workspace is not a stable directory")
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
            final = os.stat(canonical, follow_symlinks=False)
            if (final.st_dev, final.st_ino) != (initial.st_dev, initial.st_ino):
                raise ValueError("workspace filesystem object changed during preflight")
            return WorkspaceCapability(
                True,
                profile,
                str(canonical),
                read_only=False,
                runtime_path=str(canonical),
                device=final.st_dev,
                inode=final.st_ino,
            )
        if terminal.get("docker_mount_cwd_to_workspace") is not True:
            raise ValueError("Docker profile does not mount the assigned workspace")

        from tools.environments.docker import (
            _readonly_tree_digest,
            _resolve_cwd_mount_source,
        )

        canonical_source, docker_source = _resolve_cwd_mount_source(
            str(canonical),
            allowed_roots=terminal.get("docker_cwd_allowed_roots", []),
            path_mappings=terminal.get("docker_cwd_path_mappings", {}),
        )
        source_before = os.stat(canonical_source, follow_symlinks=False)
        if not stat.S_ISDIR(source_before.st_mode):
            raise ValueError("Docker workspace source is not a stable directory")
        read_only = terminal.get("docker_cwd_mount_mode", "rw") == "ro"
        runtime_probe(
            docker_source=docker_source,
            image=terminal.get("docker_image")
            or "nikolaik/python-nodejs:python3.11-nodejs20",
            expected_inode=source_before.st_ino,
            read_only=read_only,
            network_enabled=bool(terminal.get("docker_network", True)),
        )
        source_after = os.stat(canonical_source, follow_symlinks=False)
        if (source_after.st_dev, source_after.st_ino) != (
            source_before.st_dev,
            source_before.st_ino,
        ):
            raise ValueError("workspace filesystem object changed during preflight")
        content_sha256 = (
            _readonly_tree_digest(Path(canonical_source), include_root_mode=False)
            if reviewer
            else None
        )
        source_final = os.stat(canonical_source, follow_symlinks=False)
        if (source_final.st_dev, source_final.st_ino) != (
            source_after.st_dev,
            source_after.st_ino,
        ):
            raise ValueError("workspace filesystem object changed during authentication")
        return WorkspaceCapability(
            True,
            profile,
            canonical_source,
            read_only=read_only,
            runtime_path=docker_source,
            device=source_final.st_dev,
            inode=source_final.st_ino,
            content_sha256=content_sha256,
        )
    except Exception as exc:
        return WorkspaceCapability(
            False,
            profile,
            str(requested),
            reason=str(exc) or type(exc).__name__,
        )


def cached_preflight_workspace_for_profile(
    profile: str, workspace: str | Path
) -> WorkspaceCapability:
    """Run/cache a bounded probe by profile, workspace object, and config bytes."""
    requested = Path(workspace).expanduser()
    try:
        canonical = requested.resolve(strict=True)
        identity = os.stat(canonical, follow_symlinks=False)
        config = _profile_config(profile)
        terminal = config.get("terminal") if isinstance(config, dict) else {}
        if not isinstance(terminal, dict):
            terminal = {}
        if _is_reviewer_profile(profile, terminal):
            from tools.environments.docker import _readonly_tree_metadata_digest

            workspace_identity = _readonly_tree_metadata_digest(canonical)
        else:
            workspace_identity = identity.st_mtime_ns
        config_identity = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        key = (
            profile, str(canonical), identity.st_dev, identity.st_ino,
            workspace_identity, config_identity,
        )
    except Exception:
        # The ordinary implementation returns structured failure diagnostics;
        # let it do so rather than making cache-key construction authoritative.
        return preflight_workspace_for_profile(profile, requested)
    now = time.monotonic()
    with _probe_cache_lock:
        cached = _probe_cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
        wait_for = _probe_inflight.get(key)
        owner = wait_for is None
        if owner:
            wait_for = threading.Event()
            _probe_inflight[key] = wait_for
    assert wait_for is not None
    if not owner:
        # A competing tick already scheduled this exact expensive canary. Wait
        # outside both the cache mutex and board lock, then consume its result.
        wait_for.wait(timeout=35)
        with _probe_cache_lock:
            cached = _probe_cache.get(key)
            if cached is not None and cached[0] > time.monotonic():
                return cached[1]
        # The owner timed out or failed before publishing; this caller becomes
        # a bounded retry rather than trusting stale/absent evidence.
    result = preflight_workspace_for_profile(profile, canonical, profile_config=config)
    with _probe_cache_lock:
        if len(_probe_cache) >= _PROBE_CACHE_MAX_ENTRIES:
            oldest = min(_probe_cache, key=lambda item: _probe_cache[item][0])
            _probe_cache.pop(oldest, None)
        _probe_cache[key] = (now + _PROBE_CACHE_TTL_SECONDS, result)
        completed = _probe_inflight.pop(key, None)
        if completed is not None:
            completed.set()
    return result


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
        effective_capability = selected_capability
        if not selected_capability.available:
            failures.append(selected_capability)
            fallback = capability(fallback_profile)
            if not fallback.available:
                raise RuntimeError(
                    f"selected profile {selected!r} cannot access workspace "
                    f"({selected_capability.reason}); fallback profile "
                    f"{fallback_profile!r} is unavailable ({fallback.reason})"
                )
            if (
                profile_uses_restricted_reviewer_runtime(selected)
                and not (
                    fallback.read_only
                    and profile_uses_restricted_reviewer_runtime(fallback_profile)
                )
            ):
                raise RuntimeError(
                    f"reviewer profile {selected!r} cannot access workspace and fallback "
                    f"profile {fallback_profile!r} does not prove reviewer isolation"
                )
            target = fallback_profile
            effective_capability = fallback
        item = dict(child)
        item["assignee"] = target
        if (
            profile_uses_restricted_reviewer_runtime(selected)
            or profile_uses_restricted_reviewer_runtime(target)
        ):
            item["requires_reviewer_isolation"] = True
        if not workspace_capability_matches(effective_capability, workspace_path):
            raise RuntimeError(
                f"profile {target!r} workspace changed after capability preflight"
            )
        item["_workspace_device"] = effective_capability.device
        item["_workspace_inode"] = effective_capability.inode
        routed.append(item)
    return routed, failures
