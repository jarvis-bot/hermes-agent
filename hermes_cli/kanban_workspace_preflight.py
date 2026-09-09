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
    reviewer_isolated: bool = False


_PROBE_CACHE_TTL_SECONDS = 30.0
_PROBE_DEADLINE_SECONDS = 30.0
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
    """Load the authoritative effective configuration for one profile.

    This is the single policy source used by preflight, cache identity,
    classification, and worker spawn.  ``load_config`` supplies defaults,
    environment expansion, managed overlays, and path-keyed profile caching.
    """
    from hermes_cli.profiles import profile_exists, resolve_profile_env

    if not profile_exists(profile):
        raise FileNotFoundError(f"profile {profile!r} does not exist")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import load_config

    profile_home = resolve_profile_env(profile)
    token = set_hermes_home_override(profile_home)
    try:
        config = load_config()
    finally:
        reset_hermes_home_override(token)
    if not isinstance(config, dict):
        raise ValueError(f"profile {profile!r} config is not an object")
    return config


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
    deadline: Optional[float] = None,
) -> WorkspaceCapability:
    """Prove that *profile* can safely access this exact workspace.

    Expected failures are returned as an unavailable capability so callers can
    route to a known-safe owner without creating or claiming a doomed task.
    """
    requested = Path(workspace).expanduser()
    if deadline is None:
        deadline = time.monotonic() + _PROBE_DEADLINE_SECONDS
    # Preserve semantic provenance even when config loading/policy validation
    # fails before a runtime can prove availability.
    reviewer = "reviewer" in profile.lower()
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("workspace preflight deadline exceeded")
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
            _enforce_reviewer_workspace_bounds,
            _review_workspace_content_digest,
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
        if reviewer:
            _enforce_reviewer_workspace_bounds(
                Path(canonical_source), deadline=deadline
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("workspace preflight deadline exceeded")
        runtime_probe(
            docker_source=docker_source,
            image=terminal.get("docker_image")
            or "nikolaik/python-nodejs:python3.11-nodejs20",
            expected_inode=source_before.st_ino,
            read_only=read_only,
            network_enabled=bool(terminal.get("docker_network", True)),
            timeout=max(1, int(remaining)),
        )
        source_after = os.stat(canonical_source, follow_symlinks=False)
        if (source_after.st_dev, source_after.st_ino) != (
            source_before.st_dev,
            source_before.st_ino,
        ):
            raise ValueError("workspace filesystem object changed during preflight")
        content_sha256 = (
            _review_workspace_content_digest(
                Path(canonical_source), deadline=deadline
            )
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
            reviewer_isolated=reviewer,
        )
    except Exception as exc:
        return WorkspaceCapability(
            False,
            profile,
            str(requested),
            reason=str(exc) or type(exc).__name__,
            reviewer_isolated=reviewer,
        )


def cached_preflight_workspace_for_profile(
    profile: str, workspace: str | Path
) -> WorkspaceCapability:
    """Run/cache one deadline-bounded probe, deduplicating before tree traversal."""
    requested = Path(workspace).expanduser()
    deadline = time.monotonic() + _PROBE_DEADLINE_SECONDS
    reviewer = "reviewer" in profile.lower()
    try:
        canonical = requested.resolve(strict=True)
        identity = os.stat(canonical, follow_symlinks=False)
        config = _profile_config(profile)
        reviewer = profile_uses_restricted_reviewer_runtime(
            profile, profile_config=config
        )
        config_identity = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        # This intentionally cheap key enters in-flight dedupe before any
        # candidate-controlled recursive walk. Runtime authentication compares
        # the published content digest, so a same-inode in-place mutation during
        # the short TTL fails closed rather than consuming stale bytes.
        key = (
            profile, str(canonical), identity.st_dev, identity.st_ino,
            identity.st_mtime_ns, identity.st_ctime_ns, config_identity,
        )
    except Exception as exc:
        return WorkspaceCapability(
            False, profile, str(requested),
            reason=f"workspace preflight setup failed: {exc}",
            reviewer_isolated=reviewer,
        )

    with _probe_cache_lock:
        now = time.monotonic()
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
        remaining = max(0.0, deadline - time.monotonic())
        if not wait_for.wait(timeout=remaining):
            return WorkspaceCapability(
                False, profile, str(canonical),
                reason="workspace preflight deadline exceeded while waiting for identical probe",
                reviewer_isolated=reviewer,
            )
        with _probe_cache_lock:
            cached = _probe_cache.get(key)
            if cached is not None and cached[0] > time.monotonic():
                return cached[1]
        return WorkspaceCapability(
            False, profile, str(canonical),
            reason="identical workspace preflight completed without published evidence",
            reviewer_isolated=reviewer,
        )

    try:
        result = preflight_workspace_for_profile(
            profile, canonical, profile_config=config, deadline=deadline
        )
    except Exception as exc:
        result = WorkspaceCapability(
            False, profile, str(canonical),
            reason=f"workspace preflight failed: {exc}",
            reviewer_isolated=reviewer,
        )
    with _probe_cache_lock:
        if len(_probe_cache) >= _PROBE_CACHE_MAX_ENTRIES:
            oldest = min(_probe_cache, key=lambda item: _probe_cache[item][0])
            _probe_cache.pop(oldest, None)
        # TTL starts when evidence is published, not before a slow canary.
        _probe_cache[key] = (time.monotonic() + _PROBE_CACHE_TTL_SECONDS, result)
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
        reviewer_isolation_required = bool(
            child.get("requires_reviewer_isolation")
            or selected_capability.reviewer_isolated
            or profile_uses_restricted_reviewer_runtime(selected)
        )
        if not selected_capability.available:
            failures.append(selected_capability)
            fallback = capability(fallback_profile)
            if not fallback.available:
                raise RuntimeError(
                    f"selected profile {selected!r} cannot access workspace "
                    f"({selected_capability.reason}); fallback profile "
                    f"{fallback_profile!r} is unavailable ({fallback.reason})"
                )
            if reviewer_isolation_required and not (
                fallback.read_only and fallback.reviewer_isolated
            ):
                raise RuntimeError(
                    f"reviewer profile {selected!r} cannot access workspace and fallback "
                    f"profile {fallback_profile!r} does not prove reviewer isolation"
                )
            target = fallback_profile
            effective_capability = fallback
        item = dict(child)
        item["assignee"] = target
        if reviewer_isolation_required or effective_capability.reviewer_isolated:
            item["requires_reviewer_isolation"] = True
        if not workspace_capability_matches(effective_capability, workspace_path):
            raise RuntimeError(
                f"profile {target!r} workspace changed after capability preflight"
            )
        item["_workspace_device"] = effective_capability.device
        item["_workspace_inode"] = effective_capability.inode
        routed.append(item)
    return routed, failures
