"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import zlib

import sys
import uuid
from pathlib import Path
from typing import IO, Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)


# Common Docker Desktop install paths checked when 'docker' is not in PATH.
# macOS Intel: /usr/local/bin, macOS Apple Silicon (Homebrew): /opt/homebrew/bin,
# Docker Desktop app bundle: /Applications/Docker.app/Contents/Resources/bin
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EGRESS_LABEL_KEY = "hermes-egress"
_WORKSPACE_LABEL_KEY = "hermes-workspace"
_TMP_STORAGE_LABEL_KEY = "hermes-tmp-storage"
_POLICY_LABEL_KEY = "hermes-policy"
_MAX_REVIEW_WORKSPACE_NODES = 100_000
_MAX_REVIEW_WORKSPACE_FILES = 100_000
_MAX_REVIEW_WORKSPACE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_REVIEW_GIT_OBJECT_BYTES = 512 * 1024 * 1024
_MAX_REVIEW_LOOSE_OBJECT_COMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_REVIEW_PACKED_REFS_BYTES = 16 * 1024 * 1024
_MAX_REVIEW_GIT_OUTPUT_BYTES = 64 * 1024 * 1024
_REVIEW_PROVENANCE_DEADLINE_SECONDS = 180.0

_RESOURCE_LIMITED_GIT_EXEC = r'''
import os, resource, sys
for name, requested in (
    ("RLIMIT_AS", 1536 * 1024 * 1024),
    # stdout is redirected to a regular temporary file and index/read-tree
    # outputs are regular files too, so this is also a live output ceiling.
    ("RLIMIT_FSIZE", 64 * 1024 * 1024),
    ("RLIMIT_CPU", 150),
    ("RLIMIT_NOFILE", 128),
):
    kind = getattr(resource, name, None)
    if kind is None:
        raise RuntimeError(f"required reviewer resource limit is unavailable: {name}")
    _soft, hard = resource.getrlimit(kind)
    limit = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (limit, limit))
os.execve(sys.argv[1], sys.argv[1:], os.environ)
'''


def _resource_limited_git_command(git_exe: str, args: list[str]) -> list[str]:
    """Run candidate-object Git work behind hard host resource ceilings."""
    if os.name != "posix":
        raise ValueError(
            "exact-SHA reviewer Git authentication requires POSIX resource limits"
        )
    return [sys.executable, "-I", "-c", _RESOURCE_LIMITED_GIT_EXEC, git_exe, *args]


def _run_resource_limited_git(
    git_exe: str,
    args: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    capture: bool = False,
    maximum_output: int = _MAX_REVIEW_GIT_OUTPUT_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    """Execute Git with child resource limits and bounded parent-side output."""
    command = _resource_limited_git_command(git_exe, args)
    if not capture:
        return subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        size = output.tell()
        if size > maximum_output:
            raise ValueError("Git reviewer authentication output exceeds its size limit")
        output.seek(0)
        data = output.read(maximum_output + 1)
    if len(data) > maximum_output:
        raise ValueError("Git reviewer authentication output exceeds its size limit")
    return subprocess.CompletedProcess(result.args, result.returncode, data, b"")


def _check_review_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ValueError("Git reviewer workspace authentication exceeded its deadline")


def _open_nofollow_path(path: Path, flags: int) -> int:
    """Open every absolute path component through anchored no-follow dirfds."""
    if os.name != "posix":
        raise ValueError("reviewer path authentication requires POSIX dirfd support")
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise ValueError("reviewer path authentication primitives are unavailable")
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts[1:]
    if not parts:
        raise ValueError("reviewer file path is invalid")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    descriptor = os.open("/", directory_flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        final_flags = flags | os.O_NOFOLLOW | os.O_CLOEXEC
        result = os.open(parts[-1], final_flags, dir_fd=descriptor)
        os.close(descriptor)
        return result
    except BaseException:
        os.close(descriptor)
        raise


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


# Docker label values must match [a-zA-Z0-9_.-] and stay ≤63 chars to round-trip
# safely through `docker ps --filter label=key=value`. Profile and task names
# can technically contain other characters; sanitize defensively.
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """Coerce *value* into a Docker label-safe form (alnum + ``_.-``, ≤63 chars).

    Empty or all-invalid inputs collapse to ``"unknown"`` so the resulting
    label is always queryable. Used at container-create time; never round-trip
    a sanitized value back into application logic.
    """
    if not isinstance(value, str) or not value:
        return "unknown"
    cleaned = _LABEL_VALUE_OK_RE.sub("_", value)
    if cleaned == value and len(cleaned) <= 63:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix = cleaned[:50] or "unknown"
    return f"{prefix}-{digest}"


_READONLY_TREE_DIGEST_DOMAIN = b"hermes-readonly-tree-v2"


def _update_framed_digest(digest, value: bytes) -> None:
    """Append one unambiguous canonical field to a tree digest."""
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _bounded_tree_inventory(
    root: Path, *, deadline: Optional[float] = None
) -> list[Path]:
    """Collect a sortable tree inventory while enforcing bounds incrementally."""
    _check_review_deadline(deadline)
    if root.is_file():
        return [root]
    paths: list[Path] = [root]
    nodes = files = total_bytes = 0
    iterator = root.rglob("*")
    while True:
        _check_review_deadline(deadline)
        try:
            path = next(iterator)
        except StopIteration:
            break
        _check_review_deadline(deadline)
        nodes += 1
        if nodes > _MAX_REVIEW_WORKSPACE_NODES:
            raise ValueError("reviewer workspace exceeds reviewer node limit")
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            files += 1
            if files > _MAX_REVIEW_WORKSPACE_FILES:
                raise ValueError("reviewer workspace exceeds reviewer file limit")
            total_bytes += info.st_size
            if total_bytes > _MAX_REVIEW_WORKSPACE_BYTES:
                raise ValueError("reviewer workspace exceeds reviewer byte limit")
        paths.append(path)
    _check_review_deadline(deadline)
    return [root, *sorted(paths[1:])]


def _readonly_tree_digest(
    root: Path, *, include_root_mode: bool = True, deadline: Optional[float] = None,
    _inventory: Optional[list[Path]] = None,
) -> str:
    """Hash a complete read-only source tree without following symlinks.

    Every node is a domain-separated sequence of length-framed fields. Regular
    file bytes are represented by their own SHA-256, so content can be streamed
    without allowing one file's bytes to impersonate later path records.
    """
    digest = hashlib.sha256()
    _update_framed_digest(digest, _READONLY_TREE_DIGEST_DOMAIN)
    paths = _inventory or _bounded_tree_inventory(root, deadline=deadline)
    for path in paths:
        _check_review_deadline(deadline)
        relative = "." if path == root else path.relative_to(root).as_posix()
        path_stat = path.lstat()
        relative_bytes = relative.encode("utf-8", errors="surrogateescape")
        mode_bytes = (
            f"{stat.S_IMODE(path_stat.st_mode):04o}".encode("ascii")
            if include_root_mode or path != root else b""
        )
        mode = path_stat.st_mode
        if stat.S_ISLNK(mode):
            kind = b"L"
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif stat.S_ISREG(mode):
            kind = b"F"
            file_digest = hashlib.sha256()
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = _open_nofollow_path(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino, opened.st_size)
                    != (path_stat.st_dev, path_stat.st_ino, path_stat.st_size)
                ):
                    raise ValueError(
                        "read-only workspace file changed during authentication"
                    )
                while chunk := os.read(descriptor, 1024 * 1024):
                    _check_review_deadline(deadline)
                    file_digest.update(chunk)
            finally:
                os.close(descriptor)
            payload = file_digest.digest()
        elif stat.S_ISDIR(mode):
            kind = b"D"
            payload = b""
        else:
            raise ValueError(f"unsupported filesystem node in read-only workspace: {path}")
        for field in (relative_bytes, mode_bytes, kind, payload):
            _update_framed_digest(digest, field)
    return digest.hexdigest()


def _readonly_tree_metadata_digest(
    root: Path, *, deadline: Optional[float] = None,
    _inventory: Optional[list[Path]] = None,
) -> str:
    """Hash mutation-sensitive tree metadata without rereading file contents."""
    digest = hashlib.sha256()
    paths = _inventory or _bounded_tree_inventory(root, deadline=deadline)
    for path in paths:
        _check_review_deadline(deadline)
        relative = "." if path == root else path.relative_to(root).as_posix()
        info = path.lstat()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(
            f"{info.st_dev}:{info.st_ino}:{info.st_mode}:{info.st_size}:"
            f"{info.st_mtime_ns}:{info.st_ctime_ns}".encode("ascii")
        )
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _authenticated_tree_digests(
    root: Path, *, deadline: Optional[float] = None,
    _inventory: Optional[list[Path]] = None,
) -> tuple[str, str, str]:
    """Return content and metadata digests from one stable authentication window.

    The metadata passes bracket the content read so a source that changes while
    it is being authenticated is rejected rather than producing an identity
    assembled from two different tree states.
    """
    inventory = _inventory or _bounded_tree_inventory(root, deadline=deadline)
    metadata_before = _readonly_tree_metadata_digest(
        root, deadline=deadline, _inventory=inventory
    )
    content = _readonly_tree_digest(root, deadline=deadline, _inventory=inventory)
    mounted_content = _readonly_tree_digest(
        root, include_root_mode=False, deadline=deadline, _inventory=inventory
    )
    after_inventory = _bounded_tree_inventory(root, deadline=deadline)
    metadata_after = _readonly_tree_metadata_digest(
        root, deadline=deadline, _inventory=after_inventory
    )
    if metadata_before != metadata_after:
        raise ValueError(f"read-only workspace changed during authentication: {root}")
    return content, mounted_content, metadata_after


def _review_workspace_content_digest(
    root: Path, *, deadline: Optional[float] = None
) -> str:
    """Hash the logical review tree independently of reconstructed ``.git``."""
    inventory = _bounded_tree_inventory(root, deadline=deadline)
    logical_inventory = [
        path
        for path in inventory
        if path == root or path.relative_to(root).parts[0] != ".git"
    ]
    return _readonly_tree_digest(
        root,
        include_root_mode=False,
        deadline=deadline,
        _inventory=logical_inventory,
    )


def _container_tree_digest(
    docker_exe: str, container_id: str, container_path: str
) -> str:
    """Hash the exact tree held by a container bind mount.

    The digest runs inside the container, so it reads through the exact mount
    reference rather than resolving the original host pathname again.  This
    closes the rename-swap window between host authentication and daemon
    bind-mount creation.
    """
    script = r'''
import hashlib, os, stat, sys
from pathlib import Path
root = Path(sys.argv[1])
digest = hashlib.sha256()
def frame(value):
    digest.update(len(value).to_bytes(8, 'big')); digest.update(value)
frame(b'hermes-readonly-tree-v2')
paths = [root] if not root.is_dir() else [root, *sorted(root.rglob("*"))]
for path in paths:
    relative = "." if path == root else path.relative_to(root).as_posix()
    relative_bytes = relative.encode("utf-8", errors="surrogateescape")
    mode_bytes = (b'' if path == root else f"{stat.S_IMODE(path.lstat().st_mode):04o}".encode("ascii"))
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        kind = b'L'; payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
    elif stat.S_ISREG(mode):
        kind = b'F'; file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        payload = file_digest.digest()
    elif stat.S_ISDIR(mode):
        kind = b'D'; payload = b''
    else:
        raise ValueError(f"unsupported filesystem node in read-only workspace: {path}")
    for field in (relative_bytes, mode_bytes, kind, payload): frame(field)
print(digest.hexdigest())
'''
    failures: list[str] = []
    for python_exe in ("python3", "python"):
        result = subprocess.run(
            [
                docker_exe,
                "exec",
                "-w",
                "/",
                container_id,
                python_exe,
                "-I",
                "-c",
                script,
                container_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{64}", output):
            return output
        failures.append((result.stderr or result.stdout or "digest failed").strip())
    raise RuntimeError("; ".join(filter(None, failures)) or "container digest failed")


def _packed_git_ref(
    git_dir: Path, ref_name: str, *, deadline: Optional[float] = None
) -> Optional[str]:
    """Resolve *ref_name* from packed-refs, ignoring comments/peeled lines."""
    packed_refs = git_dir / "packed-refs"
    try:
        lines = _read_bounded_git_text(
            packed_refs,
            maximum_bytes=_MAX_REVIEW_PACKED_REFS_BYTES,
            deadline=deadline,
        ).splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if not line or line.startswith(("#", "^")):
            continue
        value, separator, name = line.partition(" ")
        if separator and name == ref_name:
            return value
    return None


def _git_metadata_dirs(
    git_entry: Path, *, deadline: Optional[float] = None
) -> tuple[Path, Path]:
    """Return the worktree-specific and common Git metadata directories."""
    if git_entry.is_file():
        marker = _read_bounded_git_text(
            git_entry, maximum_bytes=16 * 1024, deadline=deadline
        ).strip()
        if not marker.startswith("gitdir:"):
            raise ValueError(f"invalid Git metadata marker: {git_entry}")
        git_dir_text = marker[7:].strip()
        if not git_dir_text:
            raise ValueError(f"invalid Git metadata marker: {git_entry}")
        git_dir = (git_entry.parent / git_dir_text).resolve(strict=True)
    else:
        git_dir = git_entry.resolve(strict=True)

    common_dir = git_dir
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        common_dir_text = _read_bounded_git_text(
            commondir_file, maximum_bytes=16 * 1024, deadline=deadline
        ).strip()
        if not common_dir_text:
            raise ValueError(f"invalid Git common metadata marker: {commondir_file}")
        common_dir = (git_dir / common_dir_text).resolve(strict=True)
    return git_dir, common_dir


def _validate_local_git_metadata(git_entry: Path) -> None:
    """Reject Git metadata that can redirect authenticated host reads.

    Candidate workspaces do not provide a trusted out-of-band metadata root.
    Consequently linked worktrees, ``commondir``, and symlinks anywhere below
    ``.git`` cannot be authenticated safely and are rejected fail closed.
    """
    if git_entry.is_symlink() or git_entry.is_file():
        raise ValueError("external Git metadata is not allowed")
    if not git_entry.is_dir():
        return
    if (git_entry / "commondir").exists() or (git_entry / "commondir").is_symlink():
        raise ValueError("external Git metadata is not allowed")
    for metadata_path in git_entry.rglob("*"):
        if metadata_path.is_symlink():
            raise ValueError("external Git metadata is not allowed")


def _git_commit_identity(
    git_entry: Path, *, deadline: Optional[float] = None
) -> tuple[str, str]:
    """Return ``(HEAD text, commit)`` for normal and linked Git worktrees."""
    git_dir, common_dir = _git_metadata_dirs(git_entry, deadline=deadline)

    head = _read_bounded_git_text(
        git_dir / "HEAD", maximum_bytes=16 * 1024, deadline=deadline
    ).strip()
    if not head.startswith("ref:"):
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
            raise ValueError(f"invalid detached Git HEAD in {git_dir}")
        return head, head.lower()

    ref_name = head[4:].strip()
    if not ref_name or ref_name.startswith("/") or ".." in Path(ref_name).parts:
        raise ValueError(f"invalid symbolic Git HEAD in {git_dir}")
    for ref_root in dict.fromkeys((git_dir, common_dir)):
        ref_path = ref_root / ref_name
        try:
            value = _read_bounded_git_text(
                ref_path, maximum_bytes=16 * 1024, deadline=deadline
            ).strip()
        except FileNotFoundError:
            value = _packed_git_ref(ref_root, ref_name, deadline=deadline) or ""
        if value:
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", value):
                raise ValueError(f"invalid Git ref {ref_name} in {ref_root}")
            return head, value.lower()
    raise ValueError(f"cannot resolve Git ref {ref_name} in {git_dir}")


def _path_identity(
    path: str,
    *,
    content_digest: bool = False,
    metadata_digest: bool = False,
    deadline: Optional[float] = None,
) -> dict[str, object]:
    """Return stable host-object evidence for an immutable bind source.

    The inode fields detect atomic directory/file replacement at an unchanged
    pathname.  Git metadata additionally detects the common in-place review
    checkout update without invoking Git (which would make container startup
    depend on an optional executable).
    """
    candidate = Path(path)
    identity: dict[str, object] = {"path": str(candidate)}
    try:
        resolved = candidate.resolve(strict=True)
        stat_result = resolved.stat()
    except OSError as exc:
        if content_digest or metadata_digest:
            raise ValueError(
                f"cannot authenticate read-only workspace {path}: {exc}"
            ) from exc
        identity["missing"] = True
        return identity
    identity.update({
        "resolved": str(resolved),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "ctime_ns": stat_result.st_ctime_ns,
    })
    git_entry = resolved / ".git" if resolved.is_dir() else None
    if (content_digest or metadata_digest) and git_entry is not None:
        # A read-only workspace is candidate-controlled input.  A .git file
        # (linked worktree) or symlink can redirect provenance reads and tree
        # hashing to arbitrary host paths outside that authenticated input.
        # The Docker policy API has no separately trusted Git-metadata root,
        # so fail closed rather than deriving one from candidate contents.
        try:
            _validate_local_git_metadata(git_entry)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot authenticate read-only workspace {resolved}: {exc}"
            ) from exc
    authenticated_metadata: Optional[str] = None
    if content_digest:
        try:
            content, mounted_content, authenticated_metadata = (
                _authenticated_tree_digests(resolved, deadline=deadline)
            )
            identity["content_sha256"] = content
            identity["mounted_content_sha256"] = mounted_content
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot authenticate read-only workspace {resolved}: {exc}"
            ) from exc
    if content_digest or metadata_digest:
        try:
            identity["tree_metadata_sha256"] = (
                authenticated_metadata
                if authenticated_metadata is not None
                else _readonly_tree_metadata_digest(resolved, deadline=deadline)
            )
        except OSError as exc:
            raise ValueError(
                f"cannot authenticate read-only workspace metadata {resolved}: {exc}"
            ) from exc
    git_entry = resolved / ".git" if resolved.is_dir() else None
    if git_entry is not None and git_entry.exists():
        try:
            head, commit = _git_commit_identity(git_entry, deadline=deadline)
            identity["git_head"] = head
            identity["git_ref"] = commit
            # A linked worktree's .git file points outside the mounted source,
            # so the source-tree digest above authenticates only the pointer.
            # Authenticate the referenced metadata too: replacement refs,
            # config, index and attributes can all change Git's interpretation
            # while HEAD itself remains unchanged.
            if content_digest and git_entry.is_file():
                git_dir, common_dir = _git_metadata_dirs(
                    git_entry, deadline=deadline
                )
                git_content, _, git_metadata = _authenticated_tree_digests(git_dir)
                common_content, _, common_metadata = _authenticated_tree_digests(common_dir)
                identity["git_metadata_sha256"] = git_content
                identity["git_common_metadata_sha256"] = common_content
                identity["git_metadata_tree_sha256"] = git_metadata
                identity["git_common_metadata_tree_sha256"] = common_metadata
            elif metadata_digest and git_entry.is_file():
                git_dir, common_dir = _git_metadata_dirs(git_entry)
                identity["git_metadata_tree_sha256"] = _readonly_tree_metadata_digest(git_dir)
                identity["git_common_metadata_tree_sha256"] = _readonly_tree_metadata_digest(common_dir)
        except (OSError, ValueError) as exc:
            if content_digest or metadata_digest:
                raise ValueError(
                    f"cannot authenticate read-only workspace Git HEAD at "
                    f"{resolved}: {exc}"
                ) from exc
    return identity


def _volume_source_identities(
    volume_args: list[str], *, canonical_workspace: bool = False
) -> list[dict[str, object]]:
    """Fingerprint bind source objects represented by ``-v`` arguments."""
    identities: list[dict[str, object]] = []
    for index, arg in enumerate(volume_args[:-1]):
        if arg != "-v":
            continue
        spec = volume_args[index + 1]
        source = spec.split(":", 1)[0]
        if source.startswith("/"):
            mode = spec.rsplit(":", 1)[-1].split(",")
            destination = spec.split(":", 2)[1] if ":" in spec else ""
            is_read_only_workspace = "ro" in mode and (
                destination == "/workspace"
                or destination.startswith("/workspace/")
            )
            identity = _path_identity(
                source,
                content_digest=(
                    is_read_only_workspace
                    and not (canonical_workspace and destination == "/workspace")
                ),
            )
            identity["destination"] = destination
            if "ro" not in mode:
                for mutable_field in ("ctime_ns", "git_head", "git_ref"):
                    identity.pop(mutable_field, None)
            identities.append(identity)
    return identities


def _extra_arg_file_identities(extra_args: list[str]) -> list[dict[str, object]]:
    """Authenticate files whose contents affect immutable Docker creation state."""
    identities: list[dict[str, object]] = []
    for index, arg in enumerate(extra_args):
        value = ""
        if arg == "--env-file":
            if index + 1 >= len(extra_args):
                raise ValueError("docker_extra_args --env-file requires a path")
            value = extra_args[index + 1]
        elif arg.startswith("--env-file="):
            value = arg.split("=", 1)[1]
        if value:
            identities.append(_path_identity(value, content_digest=True))
    return identities


def _extra_args_have_host_bind(extra_args: list[str]) -> bool:
    """Return whether opaque raw args contain a host-backed bind mount."""
    mount_flags = {"-v", "--volume", "--mount"}
    for index, arg in enumerate(extra_args):
        if arg in mount_flags:
            value = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        elif any(arg.startswith(f"{flag}=") for flag in mount_flags):
            value = arg.split("=", 1)[1]
        else:
            value = ""
        for candidate in [value, *_attached_short_option_values(arg, "v")]:
            candidate = candidate.lstrip("=").replace('"', "").replace("'", "")
            mount_fields = {
                field.strip().lower() for field in candidate.split(",")
            }
            if "type=bind" in mount_fields:
                return True
            if candidate.startswith("/") and ":/" in candidate:
                return True
    return False


def _extra_args_have_host_security_file(extra_args: list[str]) -> bool:
    """Return whether a security-opt asks Docker to read a host profile file."""
    for index, arg in enumerate(extra_args):
        if arg == "--security-opt":
            value = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        elif arg.startswith("--security-opt="):
            value = arg.split("=", 1)[1]
        else:
            continue
        if value.lower().startswith("seccomp=") and value.split("=", 1)[1] != "unconfined":
            return True
    return False


def _copy_review_node(
    source: Path, destination: Path, deadline: Optional[float]
) -> None:
    """Copy one authenticated review node while enforcing the shared deadline."""
    _check_review_deadline(deadline)
    source_stat = source.lstat()
    mode = source_stat.st_mode
    if stat.S_ISLNK(mode):
        destination.symlink_to(os.readlink(source))
        return
    if stat.S_ISDIR(mode):
        destination.mkdir(mode=stat.S_IMODE(mode))
        for child in source.iterdir():
            _copy_review_node(child, destination / child.name, deadline)
        return
    if not stat.S_ISREG(mode):
        raise ValueError(f"unsupported filesystem node in read-only workspace: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = _open_nofollow_path(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size)
            != (source_stat.st_dev, source_stat.st_ino, source_stat.st_mode, source_stat.st_size)
        ):
            raise ValueError("reviewer workspace file changed during copy")
        with os.fdopen(descriptor, "rb", closefd=False) as reader, destination.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                _check_review_deadline(deadline)
                writer.write(chunk)
    finally:
        os.close(descriptor)
    shutil.copystat(source, destination, follow_symlinks=False)


def _read_bounded_regular_file(
    path: Path, *, maximum_bytes: int, deadline: Optional[float]
) -> bytes:
    """Read a path via a nonblocking, no-follow descriptor with a hard bound."""
    _check_review_deadline(deadline)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
        raise ValueError("reviewer workspace file exceeds its authentication size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = _open_nofollow_path(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError("reviewer workspace file changed during authentication")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total)):
            _check_review_deadline(deadline)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("reviewer workspace file exceeds its authentication size limit")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_bounded_git_text(
    path: Path, *, maximum_bytes: int, deadline: Optional[float]
) -> str:
    """Decode bounded candidate Git metadata read from an authenticated fd."""
    return _read_bounded_regular_file(
        path, maximum_bytes=maximum_bytes, deadline=deadline
    ).decode("utf-8", errors="strict")


class _DeadlineReader:
    """Minimal tarfile reader that checks the shared deadline on every read."""

    def __init__(self, handle: IO[bytes], deadline: Optional[float]) -> None:
        self._handle = handle
        self._deadline = deadline

    def read(self, size: int = -1) -> bytes:
        _check_review_deadline(self._deadline)
        value = self._handle.read(size)
        _check_review_deadline(self._deadline)
        return value


def _readonly_workspace_archive(
    root: Path,
    expected_metadata: str,
    expected_git_sha: Optional[str] = None,
    *,
    provenance_deadline: Optional[float] = None,
) -> tuple[IO[bytes], str]:
    """Create a stable regular/dir/symlink-only archive of an authenticated tree."""
    # General-purpose read-only mounts may legitimately contain dirty Git
    # worktrees.  Their exact bytes are authenticated below.  The stronger Git
    # provenance contract is required only for reviewer workspaces carrying an
    # out-of-band assigned SHA.
    if expected_git_sha is not None:
        if provenance_deadline is None:
            provenance_deadline = time.monotonic() + _REVIEW_PROVENANCE_DEADLINE_SECONDS
        _enforce_reviewer_workspace_bounds(root, deadline=provenance_deadline)
        _verify_git_workspace_provenance(
            root, expected_git_sha, deadline=provenance_deadline
        )
    if (
        _readonly_tree_metadata_digest(root, deadline=provenance_deadline)
        != expected_metadata
    ):
        raise ValueError(f"read-only workspace changed before materialization: {root}")
    if not root.is_dir():
        raise ValueError("authenticated read-only workspace root must be a directory")
    archive_root = root
    staging: Optional[tempfile.TemporaryDirectory[str]] = None
    if expected_git_sha is not None:
        # Candidate-owned Git configuration, hooks, refs and index must never be
        # exposed to reviewers.  They can execute code (for example fsmonitor)
        # or change the meaning of an otherwise authenticated diff.  Preserve
        # Git functionality with a minimal metadata directory built solely from
        # the authenticated commit and its content-addressed object database.
        staging = tempfile.TemporaryDirectory(prefix="hermes-review-workspace-")
        archive_root = Path(staging.name)
        for child in root.iterdir():
            if provenance_deadline is not None and time.monotonic() >= provenance_deadline:
                raise ValueError("Git reviewer workspace authentication exceeded its deadline")
            if child.name == ".git":
                continue
            destination = archive_root / child.name
            assert provenance_deadline is not None
            _copy_review_node(child, destination, provenance_deadline)
        source_git = root / ".git"
        trusted_git = archive_root / ".git"
        trusted_git.mkdir(mode=0o755)
        _copy_trusted_git_objects(
            source_git / "objects", trusted_git / "objects", deadline=provenance_deadline
        )
        _head_text, commit = _git_commit_identity(
            source_git, deadline=provenance_deadline
        )
        if commit != expected_git_sha:
            raise ValueError("reviewer workspace HEAD changed during materialization")
        # Branch/ref identity is candidate-controlled and can make ordinary
        # reviewer comparisons misleading. Trust only the out-of-band commit.
        trusted_ref = trusted_git / "refs" / "heads" / "hermes-assigned-review"
        trusted_ref.parent.mkdir(parents=True)
        trusted_ref.write_text(f"{commit}\n", encoding="ascii")
        (trusted_git / "HEAD").write_text(
            "ref: refs/heads/hermes-assigned-review\n", encoding="ascii"
        )
        (trusted_git / "config").write_text(
            "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n"
            "\tbare = false\n\tlogallrefupdates = true\n",
            encoding="utf-8",
        )
        git_exe = shutil.which("git")
        if git_exe is None:
            raise ValueError("git is required to build trusted reviewer metadata")
        index_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
        }
        index = _run_resource_limited_git(
            git_exe,
            [
                "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=/dev/null",
                f"--git-dir={trusted_git}",
                f"--work-tree={archive_root}",
                "read-tree", commit,
            ],
            timeout=max(
                0.1,
                min(
                    30.0,
                    (provenance_deadline - time.monotonic())
                    if provenance_deadline is not None
                    else 30.0,
                ),
            ),
            env=index_env,
        )
        if index.returncode != 0:
            raise ValueError("trusted reviewer Git index could not be materialized")
        (trusted_git / "HEAD").write_text(f"{commit}\n", encoding="ascii")
        trusted_ref.unlink()
        # Authenticate the actual staged bytes against the assigned commit too.
        # Source metadata checks detect swap/restore attacks today, but this
        # independent proof keeps the final snapshot fail-closed even if a host
        # filesystem does not report an intermediate rename in inode metadata.
        _verify_git_workspace_provenance(
            archive_root, expected_git_sha, deadline=provenance_deadline
        )

    try:
        archive_inventory = _bounded_tree_inventory(
            archive_root, deadline=provenance_deadline
        )
        mounted_digest = _authenticated_tree_digests(
            archive_root, deadline=provenance_deadline,
            _inventory=archive_inventory,
        )[1]
        archive_file = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024)
        try:
            with tarfile.open(fileobj=archive_file, mode="w") as archive:
                for path in archive_inventory[1:]:
                    _check_review_deadline(provenance_deadline)
                    path_stat = path.lstat()
                    mode = path_stat.st_mode
                    if not any(check(mode) for check in (stat.S_ISREG, stat.S_ISDIR, stat.S_ISLNK)):
                        raise ValueError(
                            f"unsupported filesystem node in read-only workspace: {path}"
                        )
                    info = archive.gettarinfo(
                        str(path), arcname=path.relative_to(archive_root).as_posix()
                    )
                    if stat.S_ISREG(mode):
                        flags = (
                            os.O_RDONLY
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_NONBLOCK", 0)
                        )
                        descriptor = _open_nofollow_path(path, flags)
                        try:
                            opened = os.fstat(descriptor)
                            if (
                                not stat.S_ISREG(opened.st_mode)
                                or (opened.st_dev, opened.st_ino, opened.st_size)
                                != (path_stat.st_dev, path_stat.st_ino, path_stat.st_size)
                            ):
                                raise ValueError(
                                    "reviewer workspace file changed during archiving"
                                )
                            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                                archive.addfile(
                                    info, _DeadlineReader(handle, provenance_deadline)
                                )
                        finally:
                            os.close(descriptor)
                    else:
                        archive.addfile(info)
            if (
                _readonly_tree_metadata_digest(root, deadline=provenance_deadline)
                != expected_metadata
            ):
                raise ValueError(f"read-only workspace changed during materialization: {root}")
            archive_file.seek(0)
            return archive_file, mounted_digest
        except BaseException:
            archive_file.close()
            raise
    finally:
        if staging is not None:
            staging.cleanup()


def _enforce_reviewer_workspace_bounds(
    root: Path, *, deadline: Optional[float] = None
) -> None:
    """Bound candidate-controlled host work before reviewer isolation exists."""
    _bounded_tree_inventory(root, deadline=deadline)


def _validated_loose_git_object_bytes(
    path: Path, expected_id: str, *, deadline: Optional[float] = None
) -> bytes:
    """Read a loose SHA-1 Git object and authenticate its content address."""
    try:
        _check_review_deadline(deadline)
        compressed = _read_bounded_regular_file(
            path,
            maximum_bytes=_MAX_REVIEW_LOOSE_OBJECT_COMPRESSED_BYTES,
            deadline=deadline,
        )
        inflater = zlib.decompressobj()
        canonical = inflater.decompress(compressed, _MAX_REVIEW_GIT_OBJECT_BYTES + 1)
        if len(canonical) > _MAX_REVIEW_GIT_OBJECT_BYTES or inflater.unconsumed_tail:
            raise ValueError("reviewer workspace loose Git object exceeds reviewer size limit")
        canonical += inflater.flush()
    except (OSError, zlib.error) as exc:
        raise ValueError(
            "reviewer workspace loose Git object failed independent validation"
        ) from exc
    if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("reviewer workspace loose Git object failed independent validation")
    header, separator, payload = canonical.partition(b"\0")
    object_type, size_separator, raw_size = header.partition(b" ")
    if (
        not separator
        or not size_separator
        or object_type not in {b"blob", b"tree", b"commit", b"tag"}
        or not raw_size.isdigit()
        or int(raw_size) != len(payload)
        or hashlib.sha1(canonical).hexdigest() != expected_id
    ):
        raise ValueError("reviewer workspace loose Git object failed independent validation")
    return compressed


def _validate_loose_git_objects(
    source: Path, *, deadline: Optional[float] = None
) -> None:
    """Reject loose objects whose candidate-selected path does not match its bytes."""
    if not source.exists() and not source.is_symlink():
        return
    if source.is_symlink() or not source.is_dir():
        raise ValueError("reviewer workspace Git object database is unsafe")
    loose_directory = re.compile(r"[0-9a-f]{2}")
    loose_object = re.compile(r"[0-9a-f]{38}")
    for child in source.iterdir():
        _check_review_deadline(deadline)
        if not loose_directory.fullmatch(child.name):
            continue
        if child.is_symlink() or not child.is_dir():
            raise ValueError("reviewer workspace loose Git objects are unsafe")
        for obj in child.iterdir():
            _check_review_deadline(deadline)
            if (
                not loose_object.fullmatch(obj.name)
                or obj.is_symlink()
                or not obj.is_file()
            ):
                raise ValueError("reviewer workspace loose Git objects are unsafe")
            _validated_loose_git_object_bytes(
                obj, child.name + obj.name, deadline=deadline
            )


def _copy_trusted_git_objects(
    source: Path, destination: Path, *, deadline: Optional[float] = None
) -> None:
    """Copy only content-addressed Git object files, excluding semantic caches.

    Commit graphs, multi-pack indexes, bitmaps and other auxiliary files are
    candidate-selected interpretations of the object database. Reviewers need
    only loose objects and checksum-addressed pack bytes; pack indexes are
    independently rebuilt from those trusted primitives.
    """
    if source.is_symlink() or not source.is_dir():
        raise ValueError("reviewer workspace Git object database is unsafe")
    destination.mkdir(mode=0o755)
    loose_directory = re.compile(r"[0-9a-f]{2}")
    loose_object = re.compile(r"[0-9a-f]{38}")
    pack_file = re.compile(r"pack-([0-9a-f]{40})\.pack")
    copied_packs: list[tuple[Path, str]] = []
    for child in source.iterdir():
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("Git reviewer workspace authentication exceeded its deadline")
        if loose_directory.fullmatch(child.name):
            if child.is_symlink() or not child.is_dir():
                raise ValueError("reviewer workspace loose Git objects are unsafe")
            target_dir = destination / child.name
            target_dir.mkdir(mode=0o755)
            for obj in child.iterdir():
                if (
                    not loose_object.fullmatch(obj.name)
                    or obj.is_symlink()
                    or not obj.is_file()
                ):
                    raise ValueError("reviewer workspace loose Git objects are unsafe")
                authenticated = _validated_loose_git_object_bytes(
                    obj, child.name + obj.name, deadline=deadline
                )
                (target_dir / obj.name).write_bytes(authenticated)
        elif child.name == "pack":
            if child.is_symlink() or not child.is_dir():
                raise ValueError("reviewer workspace packed Git objects are unsafe")
            target_pack = destination / "pack"
            target_pack.mkdir(mode=0o755)
            for packed in child.iterdir():
                match = pack_file.fullmatch(packed.name)
                if match is None:
                    # Exclude candidate-selected indexes, commit-graph chains,
                    # MIDX, bitmaps, reverse indexes and temporary artifacts.
                    continue
                if packed.is_symlink() or not packed.is_file():
                    raise ValueError("reviewer workspace packed Git objects are unsafe")
                copied = target_pack / packed.name
                _copy_review_node(packed, copied, deadline)
                copied_packs.append((copied, match.group(1)))
        # Deliberately omit objects/info and every unknown auxiliary entry.

    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
    }
    for copied_pack, expected_checksum in copied_packs:
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("Git reviewer workspace authentication exceeded its deadline")
        # A pack index selects which object and offset Git associates with an
        # object ID. Rebuild it from authenticated pack bytes rather than
        # preserving the candidate-selected index. The isolated Python exec
        # wrapper applies hard memory, output-file, CPU, and descriptor ceilings
        # before Git can expand candidate-selected pack objects on the host.
        git_exe = shutil.which("git")
        if git_exe is None:
            raise ValueError("git is required to authenticate a Git reviewer workspace")
        result = _run_resource_limited_git(
            git_exe,
            ["index-pack", "--strict", str(copied_pack)],
            capture=True,
            maximum_output=128,
            timeout=(
                120
                if deadline is None
                else max(0.1, min(120.0, deadline - time.monotonic()))
            ),
            env=safe_env,
        )
        generated_index = copied_pack.with_suffix(".idx")
        if (
            result.returncode != 0
            or result.stdout.decode("ascii", errors="replace").strip() != expected_checksum
            or not generated_index.is_file()
            or generated_index.is_symlink()
        ):
            raise ValueError("reviewer workspace Git pack failed independent validation")


def _verify_git_workspace_provenance(
    root: Path,
    expected_git_sha: Optional[str] = None,
    *,
    deadline: Optional[float] = None,
) -> None:
    """Authenticate a reviewer tree using a promptly reclaimed object store."""
    with tempfile.TemporaryDirectory(prefix="hermes-review-git-auth-") as staging:
        _verify_git_workspace_provenance_in_staging(
            root,
            expected_git_sha,
            deadline=deadline,
            trusted_git=Path(staging),
        )


def _verify_git_workspace_provenance_in_staging(
    root: Path,
    expected_git_sha: Optional[str] = None,
    *,
    deadline: Optional[float] = None,
    trusted_git: Path,
) -> None:
    """Require a Git workspace to be the complete clean tree of its real HEAD.

    Merely hashing candidate-supplied ``.git/HEAD`` text does not establish
    provenance: a fabricated ref can claim an assigned SHA without supplying
    the corresponding object.  Resolve the commit through Git's object store,
    require the tracked worktree to match it, reject every untracked path
    (including ignored paths), and reject unexpanded gitlinks whose contents
    are not authenticated by the parent commit.
    """
    git_entry = root / ".git"
    if not git_entry.exists() and not git_entry.is_symlink():
        if expected_git_sha is not None:
            raise ValueError("reviewer workspace is missing Git metadata for the assigned SHA")
        return
    if deadline is None:
        deadline = time.monotonic() + _REVIEW_PROVENANCE_DEADLINE_SECONDS
    _validate_local_git_metadata(git_entry)
    _validate_loose_git_objects(git_entry / "objects", deadline=deadline)
    _head_text, commit = _git_commit_identity(git_entry, deadline=deadline)
    if expected_git_sha is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", expected_git_sha):
            raise ValueError("expected reviewer Git SHA must be 40 lowercase hex characters")
        if commit != expected_git_sha:
            raise ValueError("reviewer workspace HEAD does not match the assigned SHA")
    git_exe = shutil.which("git")
    if git_exe is None:
        raise ValueError("git is required to authenticate a Git reviewer workspace")

    # Never let candidate-selected pack indexes participate in authentication.
    # Copy only content-addressed loose/pack bytes and rebuild every pack index
    # before resolving the assigned commit, tree, or blobs.
    (trusted_git / "refs").mkdir()
    (trusted_git / "HEAD").write_text(f"{commit}\n", encoding="ascii")
    (trusted_git / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n",
        encoding="utf-8",
    )
    source_objects = git_entry / "objects"
    if not source_objects.exists() and not source_objects.is_symlink():
        raise ValueError("reviewer workspace HEAD commit object is missing or invalid")
    _copy_trusted_git_objects(
        source_objects, trusted_git / "objects", deadline=deadline
    )
    base = [
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=/dev/null",
        f"--git-dir={trusted_git}",
        f"--work-tree={root}",
    ]
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
    }

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise ValueError("Git reviewer workspace authentication exceeded its deadline")

    def run(
        args: list[str], *, capture: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        check_deadline()
        try:
            return _run_resource_limited_git(
                git_exe,
                [*base, *args],
                capture=capture,
                timeout=max(0.1, min(30.0, deadline - time.monotonic())),
                env=safe_env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Git reviewer workspace authentication failed") from exc

    object_check = run(
        ["cat-file", "-e", f"{commit}^{{commit}}"], capture=False
    )
    if object_check.returncode != 0:
        raise ValueError("reviewer workspace HEAD commit object is missing or invalid")
    connectivity = run(
        ["fsck", "--strict", "--connectivity-only", "--no-dangling", commit],
        capture=False,
    )
    if connectivity.returncode != 0:
        raise ValueError("reviewer workspace Git history is incomplete or invalid")
    tree = run(["ls-tree", "-r", "-z", "--full-tree", commit])
    if tree.returncode != 0:
        raise ValueError("reviewer workspace commit tree cannot be authenticated")
    expected_paths: set[str] = set()
    for record in tree.stdout.split(b"\0"):
        check_deadline()
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise ValueError("reviewer workspace commit tree is malformed")
        mode, object_type, object_id = fields
        if mode == b"160000" or object_type == b"commit":
            raise ValueError("reviewer workspace contains unauthenticated Git submodules")
        if object_type != b"blob" or mode not in {b"100644", b"100755", b"120000"}:
            raise ValueError("reviewer workspace contains unsupported Git tree entries")
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("reviewer workspace commit tree contains an unsafe path")
        expected_paths.add(relative)
        candidate = root / relative
        try:
            candidate_mode = candidate.lstat().st_mode
        except OSError as exc:
            raise ValueError("reviewer workspace tracked files do not match HEAD") from exc
        if mode == b"120000":
            if not stat.S_ISLNK(candidate_mode):
                raise ValueError("reviewer workspace tracked files do not match HEAD")
            actual = os.readlink(candidate).encode("utf-8", errors="surrogateescape")
            blob_digest = hashlib.sha1(f"blob {len(actual)}\0".encode("ascii"))
            blob_digest.update(actual)
        else:
            if not stat.S_ISREG(candidate_mode):
                raise ValueError("reviewer workspace tracked files do not match HEAD")
            executable = bool(candidate_mode & stat.S_IXUSR)
            if executable != (mode == b"100755"):
                raise ValueError("reviewer workspace tracked file modes do not match HEAD")
            try:
                candidate_stat = candidate.lstat()
                size = candidate_stat.st_size
                blob_digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                descriptor = _open_nofollow_path(candidate, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or (opened.st_dev, opened.st_ino, opened.st_size)
                        != (candidate_stat.st_dev, candidate_stat.st_ino, size)
                    ):
                        raise ValueError(
                            "reviewer workspace tracked files changed during authentication"
                        )
                    while chunk := os.read(descriptor, 1024 * 1024):
                        check_deadline()
                        blob_digest.update(chunk)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise ValueError("reviewer workspace tracked files do not match HEAD") from exc
        if blob_digest.hexdigest().encode("ascii") != object_id:
            raise ValueError("reviewer workspace tracked files do not match HEAD")

    # Do not consult the candidate-controlled index for untracked detection.
    # Compare the filesystem leaves directly to the authenticated commit tree.
    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for top_level in root.iterdir():
        check_deadline()
        if top_level.name == ".git":
            continue
        candidates = [top_level]
        if top_level.is_dir() and not top_level.is_symlink():
            candidates.extend(top_level.rglob("*"))
        for candidate in candidates:
            if candidate.is_dir() and not candidate.is_symlink():
                actual_directories.add(candidate.relative_to(root).as_posix())
            else:
                actual_paths.add(candidate.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("reviewer workspace contains missing or untracked files")
    expected_directories = {
        parent.as_posix()
        for relative in expected_paths
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_directories != expected_directories:
        raise ValueError("reviewer workspace contains untracked directories")

    # Replacement refs alter ordinary reviewer Git commands even though the
    # authentication commands above disable them. Never preserve that
    # candidate-selected interpretation in the copied reviewer workspace.
    replace_ref_root = git_entry / "refs" / "replace"
    if replace_ref_root.exists() or replace_ref_root.is_symlink():
        raise ValueError("reviewer workspace contains Git replacement refs")
    packed_refs = git_entry / "packed-refs"
    if packed_refs.exists() or packed_refs.is_symlink():
        try:
            if not packed_refs.is_file():
                raise ValueError("reviewer workspace Git refs cannot be authenticated")
            packed_ref_bytes = _read_bounded_regular_file(
                packed_refs,
                maximum_bytes=_MAX_REVIEW_PACKED_REFS_BYTES,
                deadline=deadline,
            )
        except OSError as exc:
            raise ValueError("reviewer workspace Git refs cannot be authenticated") from exc
        if any(
            line
            and not line.startswith((b"#", b"^"))
            and line.partition(b" ")[2].startswith(b"refs/replace/")
            for line in packed_ref_bytes.splitlines()
        ):
            raise ValueError("reviewer workspace contains Git replacement refs")
    for semantic_metadata in (
        git_entry / "info" / "grafts",
        git_entry / "shallow",
        git_entry / "objects" / "info" / "alternates",
        git_entry / "objects" / "info" / "http-alternates",
    ):
        if semantic_metadata.exists() or semantic_metadata.is_symlink():
            raise ValueError(
                "reviewer workspace contains candidate-selected Git history metadata"
            )



def _best_effort_close(handle: IO[bytes]) -> None:
    """Close a reviewer staging stream without masking its primary failure."""
    try:
        handle.close()
    except OSError as exc:
        logger.warning("Reviewer staging stream cleanup failed: %s", exc)


def _best_effort_docker_cleanup(docker_exe: str, args: list[str]) -> None:
    """Run bounded cleanup without masking the failure that required it."""
    try:
        subprocess.run(
            [docker_exe, *args],
            capture_output=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Reviewer Docker cleanup failed for %s: %s", args, exc)


def _materialize_readonly_workspace(
    docker_exe: str,
    image: str,
    source: str,
    expected: dict[str, object],
    expected_git_sha: Optional[str] = None,
    expected_content_sha256: Optional[str] = None,
    *,
    disposable: bool = False,
    provenance_deadline: Optional[float] = None,
) -> str:
    """Materialize authenticated bytes in a daemon-owned volume.

    Assigned-reviewer snapshots are uniquely named and explicitly reclaimed,
    so concurrent reviewers never populate the same mutable volume.
    """
    if expected_git_sha is not None and expected_content_sha256 is not None:
        logical_content = _review_workspace_content_digest(
            Path(source), deadline=provenance_deadline
        )
        if logical_content != expected_content_sha256:
            raise ValueError("reviewer workspace content changed after dispatcher preflight")
    archive, content = _readonly_workspace_archive(
        Path(source),
        str(expected["tree_metadata_sha256"]),
        expected_git_sha,
        provenance_deadline=provenance_deadline,
    )
    if (
        expected_git_sha is None
        and expected_content_sha256 is not None
        and content != expected_content_sha256
    ):
        _best_effort_close(archive)
        raise ValueError("reviewer workspace content changed after dispatcher preflight")
    volume = (
        f"hermes-ro-{content[:16]}-{uuid.uuid4().hex[:8]}"
        if disposable
        else f"hermes-ro-{content[:24]}"
    )
    created_volume = False
    try:
        inspect = subprocess.run(
            [docker_exe, "volume", "inspect", volume],
            capture_output=True, timeout=30, check=False, stdin=subprocess.DEVNULL,
        )
    except BaseException:
        _best_effort_close(archive)
        raise
    if disposable or inspect.returncode != 0:
        try:
            subprocess.run(
                [docker_exe, "volume", "create", "--label", "hermes-agent=1", volume],
                capture_output=True, timeout=30, check=True, stdin=subprocess.DEVNULL,
            )
            created_volume = True
        except BaseException:
            # The daemon may have created the deterministically known volume
            # before the client timed out or lost its connection. Reviewer
            # volumes are disposable, so always attempt removal by name, and
            # never let cleanup mask the original failure or skip archive close.
            try:
                if disposable:
                    _best_effort_docker_cleanup(
                        docker_exe, ["volume", "rm", "-f", volume]
                    )
            finally:
                _best_effort_close(archive)
            raise
        script = r'''
import hashlib, os, pathlib, shutil, stat, sys, tarfile
root = pathlib.Path('/workspace')
for child in root.iterdir():
    shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
with tarfile.open(fileobj=sys.stdin.buffer, mode='r|*') as archive:
    for member in archive:
        target = pathlib.PurePosixPath(member.name)
        if target.is_absolute() or '..' in target.parts or not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError('unsafe workspace archive member')
        archive.extract(member, root, filter='data')
        if member.isfile() or member.isdir():
            os.chmod(root / member.name, member.mode & 0o777)
digest = hashlib.sha256()
def frame(value):
    digest.update(len(value).to_bytes(8, 'big')); digest.update(value)
frame(b'hermes-readonly-tree-v2')
for path in [root, *sorted(root.rglob('*'))]:
    relative = '.' if path == root else path.relative_to(root).as_posix()
    relative_bytes = relative.encode('utf-8', errors='surrogateescape')
    mode_bytes = b'' if path == root else f'{stat.S_IMODE(path.lstat().st_mode):04o}'.encode('ascii')
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode): kind = b'L'; payload = os.readlink(path).encode('utf-8', errors='surrogateescape')
    elif stat.S_ISREG(mode):
        kind = b'F'; file_digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''): file_digest.update(chunk)
        payload = file_digest.digest()
    elif stat.S_ISDIR(mode): kind = b'D'; payload = b''
    else: raise ValueError(f'unsupported filesystem node: {path}')
    for field in (relative_bytes, mode_bytes, kind, payload): frame(field)
print(digest.hexdigest())
'''
        populate_name = f"{volume}-populate"
        try:
            populated = subprocess.run(
                [docker_exe, "run", "--name", populate_name, "--rm", "-i",
                 "--network=none", "--cap-drop",
                 "ALL", "-v", f"{volume}:/workspace", "--entrypoint", "python3",
                 image, "-I", "-c", script],
                stdin=archive, capture_output=True, timeout=120, check=False,
            )
        except BaseException:
            _best_effort_docker_cleanup(
                docker_exe, ["rm", "-f", "-v", populate_name]
            )
            if created_volume:
                _best_effort_docker_cleanup(
                    docker_exe, ["volume", "rm", "-f", volume]
                )
            raise
        finally:
            _best_effort_close(archive)
        output = populated.stdout.decode("utf-8", errors="replace").strip()
        if populated.returncode != 0 or output != content:
            _best_effort_docker_cleanup(
                docker_exe, ["rm", "-f", "-v", populate_name]
            )
            _best_effort_docker_cleanup(
                docker_exe, ["volume", "rm", "-f", volume]
            )
            detail = populated.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"materialized read-only workspace failed authentication: {detail}"
            )
    else:
        _best_effort_close(archive)
    verifier_name = f"{volume}-verify"
    try:
        verifier = subprocess.run(
            [docker_exe, "run", "--name", verifier_name, "-d", "--network=none",
             "--cap-drop", "ALL", "-v", f"{volume}:/workspace:ro", "--entrypoint",
             "sleep", image, "120"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, check=True, stdin=subprocess.DEVNULL,
        )
        verifier_id = verifier.stdout.strip()
        try:
            if _container_tree_digest(docker_exe, verifier_id, "/workspace") != content:
                raise RuntimeError(
                    "materialized read-only workspace failed digest authentication"
                )
        finally:
            _best_effort_docker_cleanup(
                docker_exe, ["rm", "-f", "-v", verifier_name]
            )
        expected["mounted_content_sha256"] = content
        return volume
    except BaseException:
        # A daemon can create the verifier even when the client times out before
        # returning its ID. The deterministic name lets cleanup detach it before
        # reclaiming the disposable volume. Cleanup failures never mask the
        # primary verifier error or skip the volume-removal attempt.
        _best_effort_docker_cleanup(
            docker_exe, ["rm", "-f", "-v", verifier_name]
        )
        if created_volume:
            _best_effort_docker_cleanup(
                docker_exe, ["volume", "rm", "-f", volume]
            )
        raise


def _resolve_image_identity(docker_exe: str, image: str) -> str:
    """Resolve a mutable image reference to the daemon's immutable image ID."""
    try:
        result = subprocess.run(
            [docker_exe, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False, stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"cannot resolve Docker image identity for {image}: {exc}") from exc
    identity = result.stdout.strip()
    if result.returncode != 0:
        try:
            pull = subprocess.run(
                [docker_exe, "pull", image],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, check=False, stdin=subprocess.DEVNULL,
            )
            if pull.returncode == 0:
                result = subprocess.run(
                    [docker_exe, "image", "inspect", "--format", "{{.Id}}", image],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=30, check=False, stdin=subprocess.DEVNULL,
                )
                identity = result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RuntimeError(
                f"cannot pull or resolve Docker image identity for {image}: {exc}"
            ) from exc
    if result.returncode != 0 or not identity:
        raise RuntimeError(
            f"cannot resolve Docker image identity for {image}: {result.stderr.strip()}"
        )
    return identity


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``"default"`` on any error.

    Resolved at container-create time so a single container is permanently
    tagged with the profile that created it. Profile switches inside the
    same process don't retroactively relabel running containers.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def reap_orphan_containers(
    *,
    max_age_seconds: int = 600,
    profile_filter: str | None = None,
    docker_exe: str | None = None,
) -> int:
    """Remove stale hermes-tagged containers left behind by prior processes.

    Targets containers that match all of:

    * ``label=hermes-agent=1`` (created by this codebase)
    * ``status=exited`` (running containers are NEVER reaped — they may
      belong to a sibling Hermes process whose reuse path will pick them
      up; killing them would crash the sibling mid-command)
    * (optional) ``label=hermes-profile=<profile_filter>`` (sweep only the
      caller's profile by default; a hermes process in profile A must not
      tear down profile B's containers)
    * ``State.FinishedAt`` older than *max_age_seconds* ago (so a sibling
      process that just exited and is about to be replaced doesn't get
      its container yanked out from under it)

    Returns the number of containers removed. Best-effort: any failure
    (docker daemon unreachable, slow inspect, parse error) is logged at
    debug level and the function returns whatever it managed before the
    failure. Safe to call repeatedly; idempotent.

    Issue #20561 — this is the safety net for SIGKILL / OOM / crashed
    terminal exits that bypass the ``atexit`` cleanup hook. Without it,
    even with the cleanup-fix in the prior commit, a hard-killed Hermes
    process leaves its container behind permanently because there's no
    subsequent Hermes process scheduled to reuse that exact (task, profile)
    pair.
    """
    docker = docker_exe or find_docker() or "docker"
    filters = ["--filter", "label=hermes-agent=1", "--filter", "status=exited"]
    if profile_filter:
        filters.extend(["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"])

    try:
        listing = subprocess.run(
            [docker, "ps", "-a", *filters, "--format", "{{.ID}}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker ps failed: %s", e)
        return 0
    if listing.returncode != 0:
        logger.debug(
            "orphan reaper docker ps returned %d: %s",
            listing.returncode, listing.stderr.strip(),
        )
        return 0

    candidate_ids = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()]
    if not candidate_ids:
        return 0

    # Inspect each candidate to get FinishedAt; reap only those exited
    # long enough ago.  Doing this per-container (rather than bulk inspect)
    # keeps the failure blast radius to one container at a time.
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for cid in candidate_ids:
        finished_at = _container_finished_at(docker, cid)
        if finished_at is None:
            # Couldn't determine age — be conservative and leave it alone.
            continue
        age = (now - finished_at).total_seconds()
        if age < max_age_seconds:
            continue
        try:
            result = subprocess.run(
                [docker, "rm", "-f", cid],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                removed += 1
                logger.info(
                    "Reaped orphan container %s (exited %d seconds ago)",
                    cid[:12], int(age),
                )
            else:
                logger.debug(
                    "docker rm -f %s failed: %s",
                    cid[:12], result.stderr.strip(),
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("orphan reaper docker rm %s failed: %s", cid[:12], e)
    return removed


def _container_finished_at(docker_exe: str, container_id: str):
    """Parse ``docker inspect`` FinishedAt for *container_id*.

    Returns a timezone-aware datetime, or ``None`` if the field is missing,
    unparseable, or the zero-value ``0001-01-01T00:00:00Z`` Docker emits
    for never-finished containers. ``None`` means "don't reap" — the caller
    leaves the container alone.
    """
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{.State.FinishedAt}}", container_id],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker inspect %s failed: %s", container_id[:12], e)
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    # Docker emits RFC3339 with nanoseconds (e.g. "2026-05-28T13:45:00.123456789Z").
    # Python's fromisoformat handles microseconds but not nanoseconds; trim.
    import re as _re
    raw = _re.sub(r"(\.\d{6})\d+", r"\1", raw)
    raw = raw.replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        logger.debug("could not parse FinishedAt %r for %s: %s", raw, container_id[:12], e)
        return None


def find_docker() -> Optional[str]:
    """Locate the docker (or podman) CLI binary.

    Resolution order:
    1. ``HERMES_DOCKER_BINARY`` env var — explicit override (e.g. ``/usr/bin/podman``)
    2. ``docker`` on PATH via ``shutil.which``
    3. ``podman`` on PATH via ``shutil.which``
    4. Well-known macOS Docker Desktop install locations

    Returns the absolute path, or ``None`` if neither runtime can be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    # 1. Explicit override via env var (e.g. for Podman on immutable distros)
    override = os.getenv("HERMES_DOCKER_BINARY")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _docker_executable = override
        logger.info("Using HERMES_DOCKER_BINARY override: %s", override)
        return override

    # 2. docker on PATH
    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

    # 3. podman on PATH (drop-in compatible for our use case)
    found = shutil.which("podman")
    if found:
        _docker_executable = found
        logger.info("Using podman as container runtime: %s", found)
        return found

    # 4. Well-known macOS Docker Desktop locations
    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info("Found docker at non-PATH location: %s", path)
            return path

    return None


# Security flags applied to every container.
# The container itself is the security boundary (isolated from host).
# We drop all capabilities then add back the minimum needed:
#   DAC_OVERRIDE - root can write to bind-mounted dirs owned by host user
#   CHOWN/FOWNER - package managers (pip, npm, apt) need to set file ownership
#   SETUID/SETGID - the image's init drops from root to the 'hermes'
#       user (via `s6-setuidgid` in the bundled image, or whatever
#       privilege-drop helper a user image uses), which requires these
#       caps. Combined with `no-new-privileges`, the dropped process
#       still cannot escalate back to root, so the security posture is
#       preserved. Omitted entirely when the container starts as a
#       non-root user via --user, since no privilege drop is needed
#       in that mode.
# Block privilege escalation.
# /tmp is size-limited and nosuid by default but allows exec (needed by
# pip/npm builds). Profiles with large writable review copies may opt into the
# container's disk-backed writable layer instead.
#
# Note: ``--pids-limit`` is *not* in this list — it lives in ``resource_args``
# and is gated on ``_cgroup_limits_available(image)`` because it requires the
# ``pids`` cgroup controller to be delegated, which is not the case on hosts
# such as unprivileged LXCs. ``--cpus``/``--memory`` are gated for the same
# reason.
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
]
_TMP_TMPFS_ARGS = ["--tmpfs", "/tmp:rw,nosuid,size=512m"]

# Default per-container PID limit. Applied as ``--pids-limit`` only when the
# cgroup ``pids`` controller is available (see ``_cgroup_limits_available``).
_DEFAULT_PIDS_LIMIT = "256"

# /run is split out from _BASE_SECURITY_ARGS because s6-overlay images need it
# mounted ``exec``: s6 stage0 later runs ``exec /run/s6/basedir/bin/init``, which
# fails with "Permission denied" (exit 126) on a ``noexec`` mount. For all other
# images we keep the hardened ``noexec`` default.
_RUN_TMPFS_NOEXEC = "--tmpfs", "/run:rw,noexec,nosuid,size=64m"
_RUN_TMPFS_EXEC = "--tmpfs", "/run:rw,exec,nosuid,size=64m"

# Extra caps needed when the container starts as root and an init/entrypoint
# must drop privileges (via `s6-setuidgid`, `gosu`, `su`, or similar).
# Skipped when --user is passed because the container already starts
# unprivileged and never needs to switch.
_PRIVDROP_CAP_ARGS = [
    "--cap-add", "SETUID",
    "--cap-add", "SETGID",
]


def _egress_proxy_args_for_docker() -> tuple[list[str], dict[str, str], list[str]]:
    """Build the docker mount/env/host args needed to route a sandbox through
    the iron-proxy egress firewall.

    Returns ``(volume_args, env_overrides, host_args)``:

    * ``volume_args`` — read-only bind mount of the CA cert into the container
      (extends docker's ``-v`` argv list)
    * ``env_overrides`` — env vars to set on container creation: ``HTTPS_PROXY``,
      ``HTTP_PROXY``, ``NO_PROXY`` (loopback only), Python/Node/curl CA-bundle
      paths, and one ``HERMES_PROXY_TOKEN_<NAME>`` per minted mapping
    * ``host_args`` — extra ``--add-host`` flags so the container can reach the
      host-side proxy (Linux needs ``host.docker.internal:host-gateway``;
      Docker Desktop populates this automatically on macOS/Windows)

    Returns three empty containers when the proxy is disabled, not yet set up,
    or not currently running.  If ``proxy.enforce_on_docker`` is true and the
    proxy is enabled-but-not-running, raises ``RuntimeError`` so the docker
    backend refuses to start the sandbox.
    """

    # Narrow except: ImportError is the only legitimate failure here.
    # Bare ``except Exception`` would hide AttributeError, SyntaxError in
    # the config module, etc. and silently start the sandbox without
    # proxy enforcement.  We let unexpected exceptions propagate so the
    # docker backend visibly fails rather than degrading silently.
    try:
        from hermes_cli.config import load_config
        from agent.proxy_sources import iron_proxy as ip
    except ImportError as exc:
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return ([], {}, [])

    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        return ([], {}, [])

    status = ip.get_status()
    enforce = bool(proxy_cfg.get("enforce_on_docker", True))

    if not status.configured:
        msg = (
            "proxy.enabled is true but iron-proxy is not configured. "
            "Run `hermes egress setup` to mint tokens and write proxy.yaml."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    if not (status.pid and status.listening):
        msg = (
            f"iron-proxy is enabled but not running on port {status.tunnel_port}. "
            "Start it with `hermes egress start`."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    if status.ca_cert_path is None or not status.ca_cert_path.exists():
        # status.configured was True a moment ago but the CA file has
        # disappeared.  Treat this with the same enforce semantics as the
        # other failure branches — silently dropping the CA mount would
        # leave the sandbox with proxy env vars pointing at iron-proxy
        # but no trust anchor, so every TLS handshake would 5xx; or
        # worse, with enforce_on_docker=false we'd drop both the proxy
        # vars AND any other isolation, opening the sandbox.
        msg = (
            f"iron-proxy CA cert vanished from {status.ca_cert_path}. "
            "Re-run `hermes egress setup` to regenerate it."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    # Corrupt or empty mappings.json is a silent failure mode that's
    # indistinguishable from an upstream outage from inside the sandbox
    # (every request returns 403).  Refuse to mount with empty mappings
    # rather than ship a broken sandbox.
    mappings = ip.load_mappings()
    if not mappings:
        msg = (
            "iron-proxy is configured but mappings.json is empty or "
            "corrupt.  Re-run `hermes egress setup` to mint provider "
            "tokens before starting a sandbox."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    container_ca = "/etc/ssl/certs/hermes-egress-ca.crt"
    volume_args = ["-v", f"{status.ca_cert_path}:{container_ca}:ro"]

    # tunnel_port serves CONNECT (HTTPS); the plain-HTTP forward listener
    # is on tunnel_port + 1 (see build_proxy_config's listener-role notes).
    proxy_url = f"http://host.docker.internal:{status.tunnel_port}"
    plain_http_url = f"http://host.docker.internal:{status.tunnel_port + 1}"
    env_overrides: dict[str, str] = {
        # HTTPS_PROXY / HTTP_PROXY are respected by curl, requests, urllib,
        # httpx, node fetch, go default transport, etc.  Lowercase variants
        # are also set because some tools only look at one casing.
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": plain_http_url,
        "http_proxy": plain_http_url,
        # Loopback-only NO_PROXY so localhost dev servers inside the sandbox
        # (test fixtures, local LLMs) don't get sent through the proxy.
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        # CA bundle locations for the major language runtimes.  iron-proxy
        # presents a leaf cert signed by our CA on every MITM'd connection.
        #
        # CRITICAL ASYMMETRY: Python (REQUESTS_CA_BUNDLE / SSL_CERT_FILE)
        # and curl (CURL_CA_BUNDLE) REPLACE the system CA store.
        # NODE_EXTRA_CA_CERTS ADDS to it.  A Node.js process that
        # bypasses HTTPS_PROXY by using a raw socket would still see the
        # system CA store and succeed where Python/curl fail validation.
        # We additionally set NODE_OPTIONS=--use-openssl-ca to force Node
        # through the OpenSSL store that SSL_CERT_FILE controls, narrowing
        # the asymmetry.  Not a complete fix — see the docs caveat — but
        # closes the easy case.
        "REQUESTS_CA_BUNDLE": container_ca,   # Python `requests`
        "SSL_CERT_FILE": container_ca,         # Python ssl module / OpenSSL
        "CURL_CA_BUNDLE": container_ca,        # curl
        "NODE_EXTRA_CA_CERTS": container_ca,   # Node.js: adds to system store
        # NOTE: NODE_OPTIONS is intentionally NOT placed in env_overrides
        # here as a flat assignment.  We need to APPEND --use-openssl-ca
        # to whatever the user already has in NODE_OPTIONS (e.g.
        # --max-old-space-size=4096), not clobber it.  The append-merge
        # happens in DockerEnvironment._merge_node_options below.
        # For the agent inside the sandbox to identify itself as proxy-aware.
        "HERMES_EGRESS_PROXY": "1",
        # Sentinel that DockerEnvironment uses to do the NODE_OPTIONS
        # append-merge.  Stripped from the final env before docker run.
        "_HERMES_EGRESS_NODE_OPTIONS_APPEND": "--use-openssl-ca",
    }

    # Surface the per-provider proxy tokens under the standard provider env
    # names so existing SDKs and provider clients work unchanged inside the
    # sandbox.  Alias env names (e.g. GOOGLE_API_KEY for GEMINI_API_KEY)
    # receive the same token so SDKs reading either name authenticate
    # through the proxy.  Keep the HERMES_PROXY_TOKEN_* aliases for
    # diagnostics.
    for m in mappings:
        env_overrides[m.real_env_name] = m.proxy_token
        env_overrides[f"HERMES_PROXY_TOKEN_{m.real_env_name}"] = m.proxy_token
        for alias in getattr(m, "alias_env_names", ()) or ():
            env_overrides[alias] = m.proxy_token

    # On Linux, host.docker.internal isn't populated by default — Docker Desktop
    # adds it on macOS/Windows; on Linux we need an explicit --add-host with
    # host-gateway.  On Desktop this is a no-op (harmless duplicate).
    host_args: list[str] = ["--add-host", "host.docker.internal:host-gateway"]

    return (volume_args, env_overrides, host_args)


def _egress_reuse_fingerprint(
    volume_args: list[str],
    env_overrides: dict[str, str],
    host_args: list[str],
) -> str:
    """Stable Docker-label value for the egress posture of a container."""
    if not (volume_args or env_overrides or host_args):
        return "off"
    payload = json.dumps(
        {
            "volume_args": volume_args,
            "env_overrides": env_overrides,
            "host_args": host_args,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _egress_enforce_on_docker(default: bool = True) -> bool:
    """Read proxy.enforce_on_docker with fail-safe defaulting."""
    try:
        from hermes_cli.config import load_config as _load_cfg

        return bool((_load_cfg().get("proxy") or {}).get("enforce_on_docker", default))
    except (ImportError, OSError):
        return default
    except Exception:
        return default


def _critical_egress_env_names(env_overrides: dict[str, str]) -> set[str]:
    """Env names that would weaken or bypass enforced egress if overridden."""
    critical = {
        "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
        "NO_PROXY", "no_proxy",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS", "NODE_OPTIONS",
    }
    critical.update(
        key for key in env_overrides
        if key.endswith("_API_KEY") or key.endswith("_TOKEN")
    )
    return critical


def _extra_args_egress_collisions(
    extra_args: list[str], critical_names: set[str],
) -> list[str]:
    """Return docker_extra_args entries that can override egress controls."""
    collisions: list[str] = []
    env_flags = {"-e", "--env", "--env-file"}
    network_flags = {"--network", "--net"}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        nxt = extra_args[i + 1] if i + 1 < len(extra_args) else ""
        if arg in env_flags:
            if arg == "--env-file":
                collisions.append(arg)
            else:
                name = nxt.split("=", 1)[0]
                if name in critical_names:
                    collisions.append(name)
            i += 2
            continue
        if any(arg.startswith(f"{flag}=") for flag in env_flags):
            if arg.startswith("--env-file="):
                collisions.append("--env-file")
            else:
                name = arg.split("=", 1)[1].split("=", 1)[0]
                if name in critical_names:
                    collisions.append(name)
        elif arg in network_flags or any(arg.startswith(f"{flag}=") for flag in network_flags):
            collisions.append(arg)
        for attached in _short_option_values(extra_args, i, "e"):
            name = attached.lstrip("=").split("=", 1)[0]
            if name in critical_names:
                collisions.append(name)
        i += 1
    return sorted(set(collisions))


def _extra_args_reserved_label_collisions(extra_args: list[str]) -> list[str]:
    """Return labels in ``docker_extra_args`` reserved for Hermes identity."""
    reserved = {
        "hermes-agent",
        "hermes-task-id",
        "hermes-profile",
        _EGRESS_LABEL_KEY,
        _WORKSPACE_LABEL_KEY,
        _TMP_STORAGE_LABEL_KEY,
        _POLICY_LABEL_KEY,
    }
    collisions: list[str] = []
    for index, arg in enumerate(extra_args):
        value = ""
        if arg in {"--label-file"} or arg.startswith("--label-file="):
            # A file can define any reserved key and is intentionally not read
            # here: config validation must not turn arbitrary paths into I/O.
            collisions.append("--label-file")
            continue
        if arg in {"-l", "--label"}:
            if index + 1 < len(extra_args):
                value = extra_args[index + 1]
        elif arg.startswith(("-l=", "--label=")):
            value = arg.split("=", 1)[1]
        attached_values = _short_option_values(extra_args, index, "l")
        values = [value, *attached_values]
        for candidate in values:
            if candidate:
                name = candidate.lstrip("=").split("=", 1)[0]
                if name in reserved:
                    collisions.append(name)
    return sorted(set(collisions))


def _build_security_args(
    run_as_host_user: bool,
    run_exec: bool = False,
    tmp_storage: str = "tmpfs",
) -> list[str]:
    """Return the security/cap/tmpfs args tailored to the privilege mode.

    ``run_exec`` mounts ``/run`` with ``exec`` instead of the hardened
    ``noexec`` default. This is required for s6-overlay images whose ``/init``
    entrypoint execs ``/run/s6/basedir/bin/init`` during startup; see
    ``_image_uses_init_entrypoint``.
    """
    if tmp_storage not in {"tmpfs", "disk"}:
        raise ValueError("docker_tmp_storage must be 'tmpfs' or 'disk'")
    tmp_args = list(_TMP_TMPFS_ARGS) if tmp_storage == "tmpfs" else []
    run_tmpfs = list(_RUN_TMPFS_EXEC if run_exec else _RUN_TMPFS_NOEXEC)
    args = list(_BASE_SECURITY_ARGS) + tmp_args + run_tmpfs
    if run_as_host_user:
        return args
    return args + list(_PRIVDROP_CAP_ARGS)


def _image_uses_init_entrypoint(docker_exe: str, image: str) -> bool:
    """Return True if ``image``'s entrypoint is the s6-overlay ``/init``.

    Such images (e.g. anything built on ``s6-overlay``, including
    ``hermes-agent:latest``) already provide their own PID-1 init and execute
    ``/run/s6/basedir/bin/init`` during stage0 startup. They are incompatible
    with Docker's ``--init`` (two competing PID-1 inits) and with a ``noexec``
    ``/run`` mount. Detection is best-effort: on any inspection failure we
    return False and keep the hardened defaults.
    """
    try:
        result = subprocess.run(
            [docker_exe, "image", "inspect", image,
             "--format", "{{json .Config.Entrypoint}}"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Docker: could not inspect entrypoint for %s: %s", image, e)
        return False
    if result.returncode != 0:
        # Image may not be pulled yet; the run will pull it. Defaults are safe
        # for non-s6 images, so don't block on this.
        logger.debug(
            "Docker: image inspect for %s returned %d (stderr=%s)",
            image, result.returncode, result.stderr.strip(),
        )
        return False
    raw = (result.stdout or "").strip()
    if not raw or raw == "null":
        return False
    try:
        entrypoint = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if not isinstance(entrypoint, list) or not entrypoint:
        return False
    first = str(entrypoint[0]).strip()
    return first in ("/init", "/package/admin/s6-overlay/command/init")


def _resolve_host_user_spec() -> Optional[str]:
    """Return ``<uid>:<gid>`` for the current host user, or ``None`` on platforms
    where this is not meaningful (e.g. Windows without posix ids).

    We intentionally read ``os.getuid()``/``os.getgid()`` directly rather than
    going through ``getpass``/``pwd`` so this stays cheap and never raises on
    nameless UIDs (nss lookups can fail inside sandboxed launchers).
    """
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if get_uid is None or get_gid is None:
        return None
    try:
        return f"{get_uid()}:{get_gid()}"
    except Exception:  # pragma: no cover - defensive
        return None


_storage_opt_ok: Optional[bool] = None  # cached result across instances
_cgroup_limits_ok: Optional[bool] = None  # cached result across instances


def _cgroup_limits_available(image: str) -> bool:
    """Probe whether cgroup resource limits work in this environment.

    Tests ``--cpus``, ``--memory`` and ``--pids-limit`` together by spawning
    a throwaway container from *image* (the same sandbox image we are about
    to use for real, so no extra pull and no dependency on a public
    registry). The container runs ``sleep 0`` — sleep is guaranteed to be
    present because the sandbox itself uses ``sleep 2h`` as its long-lived
    entrypoint.

    On hosts where the corresponding cgroup controllers are not delegated
    to this process (typical inside unprivileged LXCs and some rootless
    setups) these flags cause every container start to fail with ``OCI
    runtime error`` / exit 126. The probe runs once per process and the
    result — which is host-wide, not image-specific — is cached.
    """
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe or not image:
        _cgroup_limits_ok = False
        return False

    try:
        result = subprocess.run(
            [docker_exe, "run", "--rm", "--network=none", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "--entrypoint", "sleep",
             "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "0"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate the cpu, "
                "memory and pids cgroup controllers to this container. "
                "Probe stderr: %s",
                (result.stderr or "").strip()[:500],
            )
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)

    return _cgroup_limits_ok


def _ensure_docker_available() -> None:
    """Best-effort check that the docker CLI is available before use.

    Reuses ``find_docker()`` so this preflight stays consistent with the rest of
    the Docker backend, including known non-PATH Docker Desktop locations.
    """
    docker_exe = find_docker()
    if not docker_exe:
        logger.error(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the "
            "CLI is available."
        )
        raise RuntimeError(
            "Docker executable not found in PATH or known install locations. "
            "Install Docker and ensure the 'docker' command is available."
        )

    try:
        result = subprocess.run(
            [docker_exe, "version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error(
            "Docker backend selected but the resolved docker executable '%s' could "
            "not be executed.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker executable could not be executed. Check your Docker installation."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Docker backend selected but '%s version' timed out. "
            "The Docker daemon may not be running.",
            docker_exe,
            exc_info=True,
        )
        raise RuntimeError(
            "Docker daemon is not responding. Ensure Docker is running and try again."
        )
    except Exception:
        logger.error(
            "Unexpected error while checking Docker availability.",
            exc_info=True,
        )
        raise
    else:
        if result.returncode != 0:
            logger.error(
                "Docker backend selected but '%s version' failed "
                "(exit code %d, stderr=%s)",
                docker_exe,
                result.returncode,
                result.stderr.strip(),
            )
            raise RuntimeError(
                "Docker command is available but 'docker version' failed. "
                "Check your Docker installation."
            )


def _is_path_within(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or a descendant (component-bounded)."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_cwd_mount_source(
    host_cwd: str,
    *,
    allowed_roots: list | None,
    path_mappings: dict | None,
) -> tuple[str, str]:
    """Validate a container-visible cwd and return canonical + Docker-host paths."""
    try:
        canonical = Path(host_cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Docker cwd mount source cannot be resolved: {host_cwd!r}") from exc
    if not canonical.is_dir():
        raise ValueError(f"Docker cwd mount source is not a directory: {canonical}")

    if allowed_roots is None:
        allowed_roots = []
    if not isinstance(allowed_roots, list) or any(
        not isinstance(root, str) or not os.path.isabs(root) for root in allowed_roots
    ):
        raise ValueError("docker_cwd_allowed_roots must be a list of absolute paths")
    canonical_roots = [Path(root).expanduser().resolve(strict=False) for root in allowed_roots]
    if canonical_roots and not any(_is_path_within(canonical, root) for root in canonical_roots):
        raise ValueError(
            f"Docker cwd mount source {canonical} is outside every allowed workspace root"
        )

    if path_mappings is None:
        path_mappings = {}
    if not isinstance(path_mappings, dict):
        raise ValueError("docker_cwd_path_mappings must be a mapping of absolute paths")

    matches: list[tuple[Path, Path]] = []
    for source, destination in path_mappings.items():
        if (
            not isinstance(source, str)
            or not isinstance(destination, str)
            or not os.path.isabs(source)
            or not os.path.isabs(destination)
        ):
            raise ValueError(
                "docker_cwd_path_mappings must map absolute container paths "
                "to absolute Docker-host paths"
            )
        source_path = Path(source).expanduser().resolve(strict=False)
        if _is_path_within(canonical, source_path):
            matches.append((source_path, Path(destination)))

    if path_mappings and not matches:
        raise ValueError(f"Docker cwd mount source {canonical} has no matching path mapping")
    if not matches:
        return str(canonical), str(canonical)

    # Longest matching source wins, so a specific nested map safely overrides
    # a broader parent map without string-prefix ambiguity.
    source_path, destination = max(matches, key=lambda item: len(item[0].parts))
    translated = destination / canonical.relative_to(source_path)
    return str(canonical), str(translated)


def _volume_mounts_path(volume: str, container_path: str) -> bool:
    """Recognize a Docker ``-v`` entry targeting a path or its subtree."""
    # Match from the right so Windows drive-letter sources (``C:\\...``) do
    # not confuse destination parsing. Canonicalize the destination because the
    # container runtime resolves repeated separators and ``..`` components.
    match = re.search(r":(/[^:]*)(?::[^:]*)?$", volume)
    return bool(
        match and _container_path_is_at_or_below(match.group(1), container_path)
    )


def _volume_targets_exact_path(volume: str, container_path: str) -> bool:
    """Recognize a Docker ``-v`` entry targeting exactly one container path."""
    match = re.search(r":(/[^:]*)(?::[^:]*)?$", volume)
    if not match:
        return False
    candidate = posixpath.normpath("/" + match.group(1).strip().lstrip("/"))
    expected = posixpath.normpath("/" + container_path.strip().lstrip("/"))
    return candidate == expected


def _container_path_is_at_or_below(candidate: str, protected: str) -> bool:
    """Return whether a container path resolves to a protected path/subtree."""
    candidate = candidate.strip().replace('"', "").replace("'", "")
    if not candidate.startswith("/"):
        return False
    # posixpath intentionally preserves exactly two leading slashes; Docker
    # does not provide a separate namespace there, so collapse all of them.
    canonical = posixpath.normpath("/" + candidate.lstrip("/"))
    root = posixpath.normpath("/" + protected.lstrip("/"))
    if root == "/":
        return canonical.startswith("/")
    return canonical == root or canonical.startswith(root + "/")


def _mount_spec_targets_path(spec: str, container_path: str) -> bool:
    """Recognize a Docker ``--mount`` CSV destination after normalization."""
    normalized = spec.replace('"', "").replace("'", "")
    for field in normalized.split(","):
        key, separator, value = field.partition("=")
        if separator and key.strip().lower() in {"dst", "destination", "target"}:
            if _container_path_is_at_or_below(value.strip(), container_path):
                return True
    return False


def _attached_short_option_values(arg: str, option: str) -> list[str]:
    """Return a Docker short-option value bundled in one token.

    Docker accepts both ``-vVALUE`` and bundles such as ``-itvVALUE``. Only
    no-value flags may precede the value-taking option; once another
    value-taking option such as ``-e`` starts, later characters are its value
    rather than more bundled flags.
    """
    if not arg.startswith("-") or arg.startswith("--"):
        return []
    token = arg[1:]
    for index, char in enumerate(token):
        if char == option:
            return [token[index + 1:]]
        # Docker run's value-free short flags may be bundled before another
        # short option.  Stop at every value-taking flag so characters in that
        # option's value are never reinterpreted as flags.
        if char not in {"d", "i", "P", "q", "t"}:
            break
    return []


def _short_option_values(extra_args: list[str], index: int, option: str) -> list[str]:
    """Return attached or following values for a bundled Docker short option."""
    values = _attached_short_option_values(extra_args[index], option)
    if values == [""] and index + 1 < len(extra_args):
        return [extra_args[index + 1]]
    return values


def _volume_mounts_workspace(volume: str) -> bool:
    """Recognize a Docker ``-v`` entry targeting /workspace or its subtree."""
    return _volume_mounts_path(volume, "/workspace")


def _extra_args_mount_workspace(extra_args: list[str]) -> bool:
    """Return whether raw Docker flags can add or replace a /workspace mount."""
    mount_flags = {"-v", "--volume", "--mount", "--tmpfs"}
    for index, arg in enumerate(extra_args):
        if arg == "--volumes-from" or arg.startswith("--volumes-from="):
            # A donor can contain a writable /workspace destination and its
            # mutable mount set cannot participate in our reuse fingerprint.
            return True
        if arg in mount_flags:
            value = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        elif any(arg.startswith(f"{flag}=") for flag in mount_flags):
            value = arg.split("=", 1)[1]
        else:
            value = ""
        values = [value, *_attached_short_option_values(arg, "v")]
        for candidate in values:
            candidate = candidate.lstrip("=")
            if _volume_mounts_workspace(candidate):
                return True
            # Docker also accepts destination-only anonymous volumes and
            # destination-only --tmpfs values.
            direct_target = candidate.split(":", 1)[0]
            if _container_path_is_at_or_below(direct_target, "/workspace"):
                return True
            if _mount_spec_targets_path(candidate, "/workspace"):
                return True
    return False


def _extra_args_mount_tmp(extra_args: list[str]) -> bool:
    """Return whether raw Docker flags can replace /tmp storage policy."""
    mount_flags = {"-v", "--volume", "--mount", "--tmpfs"}
    for index, arg in enumerate(extra_args):
        if arg == "--volumes-from" or arg.startswith("--volumes-from="):
            # The donor's destinations cannot be validated without inspecting
            # mutable external container state, so fail closed.
            return True
        if arg in mount_flags:
            value = extra_args[index + 1] if index + 1 < len(extra_args) else ""
        elif any(arg.startswith(f"{flag}=") for flag in mount_flags):
            value = arg.split("=", 1)[1]
        else:
            value = ""
        values = [value, *_attached_short_option_values(arg, "v")]
        for candidate in values:
            candidate = candidate.lstrip("=")
            if _volume_mounts_path(candidate, "/tmp"):
                return True
            tmpfs_target = candidate.split(":", 1)[0]
            if _container_path_is_at_or_below(tmpfs_target, "/tmp"):
                return True
            if _mount_spec_targets_path(candidate, "/tmp"):
                return True
    return False


def _extra_args_override_network(extra_args: list[str]) -> bool:
    """Return whether raw Docker flags select a network mode."""
    return any(
        arg in {"--network", "--net"}
        or arg.startswith(("--network=", "--net="))
        for arg in extra_args
    )


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution with resource limits and persistence.

    Security: all capabilities dropped, no privilege escalation, PID limits,
    size-limited tmpfs for scratch dirs. The container itself is the security
    boundary — the filesystem inside is writable so agents can install packages
    (pip, npm, apt) as needed. Writable workspace via tmpfs or bind mounts.

    Persistence: when enabled, bind mounts preserve /workspace and /root
    across container restarts.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        host_cwd: Optional[str] = None,
        auto_mount_cwd: bool = False,
        cwd_mount_mode: str = "rw",
        cwd_path_mappings: dict | None = None,
        cwd_allowed_roots: list | None = None,
        run_as_host_user: bool = False,
        extra_args: list = None,
        persist_across_processes: bool = True,
        tmp_storage: str = "tmpfs",
        expected_git_sha: Optional[str] = None,
        reviewer_mode: bool = False,
        expected_content_sha256: Optional[str] = None,
    ):
        if cwd == "~":
            cwd = "/root"
        if not isinstance(tmp_storage, str) or tmp_storage not in {"tmpfs", "disk"}:
            raise ValueError("docker_tmp_storage must be exactly 'tmpfs' or 'disk'")
        reviewer_mode = bool(reviewer_mode or expected_git_sha is not None)
        if reviewer_mode:
            # Exact-SHA reviews need disposable scratch larger than the hardened
            # 512 MiB default. Make this a runtime invariant, not a model-reported
            # or profile-configuration-only property: Docker's effective-mount
            # verification below then proves /tmp is on the writable layer.
            tmp_storage = "disk"
        requested_cwd = cwd
        effective_cwd = "/tmp/review" if reviewer_mode else cwd
        startup_cwd = "/tmp" if reviewer_mode else cwd
        super().__init__(cwd=effective_cwd, timeout=timeout)
        self._reviewer_mode = reviewer_mode
        self._expected_git_sha = expected_git_sha
        # Reviewer scratch state must remain inside the disposable container.
        # Never bind a host-backed /root or reuse its mutable writable layer,
        # even when global terminal defaults request persistence.
        self._persistent = persistent_filesystem and not reviewer_mode
        self._persist_across_processes = persist_across_processes and not reviewer_mode
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        if reviewer_mode:
            if network:
                raise ValueError("assigned reviewer workspaces require docker_network=false")
            if self._forward_env or self._env:
                raise ValueError(
                    "assigned reviewer workspaces cannot receive forwarded or configured environment variables"
                )
            if volumes:
                raise ValueError(
                    "assigned reviewer workspaces cannot receive configured Docker volumes"
                )
            if extra_args:
                raise ValueError(
                    "assigned reviewer workspaces cannot receive raw Docker arguments"
                )
        self._container_id: Optional[str] = None
        self._labels: dict[str, str] = {}
        self._image: str = ""
        self._container_name: str = ""
        self._image_uses_s6_init: bool = False
        self._all_run_args: list[str] = []
        self._network_enabled = network
        self._tmp_storage = tmp_storage
        self._workspace_requires_ro = False
        self._readonly_workspace_sources: list[
            tuple[str, str, dict[str, object]]
        ] = []
        self._snapshot_volumes: list[str] = []
        logger.info(f"DockerEnvironment volumes: {volumes}")
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning(f"docker_volumes config is not a list: {volumes!r}")
            volumes = []

        # Fail fast if Docker is not available.
        _ensure_docker_available()

        # Build resource limit args (gated by cgroup availability probe so
        # they degrade gracefully on hosts without controller delegation,
        # e.g. unprivileged LXCs). The probe runs once per process and is
        # cached host-wide.
        resource_args = []
        if cpu > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--memory", f"{memory}m"])
        if _cgroup_limits_available(image):
            resource_args.extend(["--pids-limit", _DEFAULT_PIDS_LIMIT])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")

        # Persistent workspace via bind mounts from a configurable host directory
        # (TERMINAL_SANDBOX_DIR, default ~/.hermes/sandboxes/). Non-persistent
        # mode uses tmpfs (ephemeral, fast, gone on cleanup).
        from tools.environments.base import get_sandbox_dir

        # User-configured volume mounts (from config.yaml docker_volumes)
        volume_args = []
        workspace_explicitly_mounted = False
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning(f"Docker volume entry is not a string: {vol!r}")
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" in vol:
                if _volume_mounts_path(vol, "/tmp"):
                    raise ValueError(
                        "docker_volumes cannot mount /tmp or its subdirectories; "
                        "use docker_tmp_storage to select the /tmp policy"
                    )
                volume_args.extend(["-v", vol])
                if _volume_mounts_workspace(vol):
                    mode = vol.rsplit(":", 1)[-1].split(",")
                    source = vol.split(":", 1)[0]
                    if "ro" in mode and not source.startswith("/"):
                        raise ValueError(
                            "docker_volumes read-only /workspace requires an authenticated host bind "
                            "with an absolute POSIX source path; named and Windows-style "
                            "volume sources cannot prove exact reviewer contents"
                        )
                    workspace_explicitly_mounted = True
            else:
                logger.warning(f"Docker volume '{vol}' missing colon, skipping")

        canonical_host_cwd = ""
        docker_host_cwd = ""
        bind_host_cwd = auto_mount_cwd and bool(host_cwd)
        canonical_workspace_identity: Optional[dict[str, object]] = None
        if auto_mount_cwd and workspace_explicitly_mounted:
            raise ValueError(
                "docker_volumes already mounts /workspace while "
                "docker_mount_cwd_to_workspace is enabled; remove the explicit "
                "workspace mount or disable the automatic mount"
            )
        reviewer_provenance_deadline = (
            time.monotonic() + _REVIEW_PROVENANCE_DEADLINE_SECONDS
            if reviewer_mode
            else None
        )
        if bind_host_cwd:
            assert host_cwd is not None
            if cwd_mount_mode not in {"ro", "rw"}:
                raise ValueError("docker_cwd_mount_mode must be 'ro' or 'rw'")
            canonical_host_cwd, docker_host_cwd = _resolve_cwd_mount_source(
                host_cwd,
                allowed_roots=cwd_allowed_roots,
                path_mappings=cwd_path_mappings,
            )
            if cwd_mount_mode == "ro":
                if reviewer_mode:
                    _enforce_reviewer_workspace_bounds(
                        Path(canonical_host_cwd), deadline=reviewer_provenance_deadline
                    )
                canonical_workspace_identity = _path_identity(
                    canonical_host_cwd,
                    content_digest=True,
                    deadline=reviewer_provenance_deadline,
                )
                self._readonly_workspace_sources.append(
                    (canonical_host_cwd, "/workspace", canonical_workspace_identity)
                )

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        writable_args = []
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args.extend([
                "-v", f"{self._home_dir}:/root",
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args.extend([
                    "-v", f"{self._workspace_dir}:/workspace",
                ])
        else:
            if not bind_host_cwd and not workspace_explicitly_mounted:
                writable_args.extend([
                    "--tmpfs", "/workspace:rw,exec,size=10g",
                ])
            writable_args.extend([
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ])

        workspace_label = "managed-persistent" if self._persistent else "managed-ephemeral"
        if workspace_explicitly_mounted:
            workspace_specs = sorted(
                volume_args[index + 1]
                for index, arg in enumerate(volume_args[:-1])
                if arg == "-v" and _volume_mounts_workspace(volume_args[index + 1])
            )
            workspace_label = hashlib.sha256(
                "\0".join(workspace_specs).encode("utf-8")
            ).hexdigest()[:16]
            self._workspace_requires_ro = any(
                "ro" in spec.rsplit(":", 1)[-1].split(",")
                for spec in workspace_specs
            )
        if bind_host_cwd:
            mount_spec = f"{docker_host_cwd}:/workspace"
            if cwd_mount_mode == "ro":
                mount_spec += ":ro"
            workspace_label = hashlib.sha256(mount_spec.encode("utf-8")).hexdigest()[:16]
            logger.info(
                "Mounting configured cwd %s via Docker host path %s to /workspace:%s",
                canonical_host_cwd,
                docker_host_cwd,
                cwd_mount_mode,
            )
            volume_args = ["-v", mount_spec, *volume_args]
            self._workspace_requires_ro = cwd_mount_mode == "ro"

        # Mount credential files (OAuth tokens, etc.) declared by skills.
        # Read-only so the container can authenticate but not modify host creds.
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            for mount_entry in ([] if reviewer_mode else get_credential_file_mounts()):
                src = Path(mount_entry["host_path"])
                if src.is_dir():
                    # Docker-in-Docker: Docker auto-created the source path as
                    # a directory when it didn't exist on the host.  Mounting a
                    # directory over a file destination causes exit 125.
                    logger.warning(
                        "Docker: skipping credential mount — source is a directory "
                        "(likely Docker-in-Docker auto-creation): %s",
                        src,
                    )
                    continue
                if not src.is_file():
                    logger.warning(
                        "Docker: skipping credential mount — source not found: %s", src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount skill directories (local + external) so skill
            # scripts/templates are available inside the container.
            for skills_mount in ([] if reviewer_mode else get_skills_directory_mount()):
                src = Path(skills_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping skills mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            # Mount host-side cache directories (documents, images, audio,
            # screenshots) so the agent can access uploaded files and other
            # cached media from inside the container.  Read-only — the
            # container reads these but the host gateway manages writes.
            for cache_mount in ([] if reviewer_mode else get_cache_directory_mounts()):
                src = Path(cache_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping cache mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Docker: could not load credential file mounts: %s", e)

        # Egress credential-injection proxy (iron-proxy) — when configured,
        # mount the CA cert into the sandbox and set HTTPS_PROXY + CA-bundle
        # env vars so outbound traffic routes through the host-side proxy.
        # The sandbox receives PROXY tokens instead of real API keys.
        if reviewer_mode:
            egress_volume_args, egress_env_overrides, egress_host_args = [], {}, []
        else:
            egress_volume_args, egress_env_overrides, egress_host_args = (
                _egress_proxy_args_for_docker()
            )
        egress_label = _egress_reuse_fingerprint(
            egress_volume_args, egress_env_overrides, egress_host_args,
        )
        _enforce_egress = _egress_enforce_on_docker()
        _critical_egress_names = _critical_egress_env_names(egress_env_overrides)
        if egress_env_overrides:
            _forward_collisions = sorted(
                key for key in self._forward_env if key in _critical_egress_names
            )
            if _forward_collisions:
                _msg = (
                    f"docker_forward_env would inject real egress-protected "
                    f"variables {_forward_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these names from docker_forward_env "
                        "or disable enforce_on_docker to opt out of egress isolation."
                    )
                logger.warning(
                    "%s  Explicit docker_forward_env values will override egress tokens.",
                    _msg,
                )
        volume_args.extend(egress_volume_args)
        # egress env overrides are merged in further below alongside the
        # other env_args computation.

        # Explicit environment variables (docker_env config) — set at container
        # creation so they're available to all processes (including entrypoint).
        # Egress proxy env vars (HTTPS_PROXY, CA-bundle paths, proxy tokens)
        # are merged below.  Precedence policy:
        #
        # - When egress enforcement is on AND the user's docker_env tries
        #   to override one of the proxy-control vars (HTTPS_PROXY,
        #   SSL_CERT_FILE, etc.), fail-loud rather than silently inverting
        #   the isolation.  The CA mount + tokens would still ship while
        #   traffic leaves the sandbox direct with real credentials —
        #   exactly what enforce_on_docker is meant to prevent.
        # - When enforcement is off, the user's docker_env wins (current
        #   behavior) but we log a warning naming both config sources.
        # - When the user override is identical to the egress value, no-op.
        if egress_env_overrides:
            try:
                from hermes_cli.config import load_config as _load_cfg_for_collision
                _proxy_cfg = (_load_cfg_for_collision().get("proxy") or {})
            except (ImportError, OSError):
                _proxy_cfg = {}
            except Exception as _e:  # noqa: BLE001 — narrowed below via yaml import
                # yaml.YAMLError from a malformed config.yaml.  We import
                # lazily because PyYAML is a soft dep in some test envs.
                try:
                    import yaml  # noqa: F401
                except ImportError:
                    raise
                logger.warning(
                    "Could not read proxy config for egress collision check: %s",
                    _e,
                )
                _proxy_cfg = {}
            _enforce_egress = bool(_proxy_cfg.get("enforce_on_docker", True))
            # Egress-controlling env vars that affect the proxy posture.
            _critical_proxy_control = {
                "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "NO_PROXY", "no_proxy",
                "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
                "NODE_EXTRA_CA_CERTS",
            }
            # stephenschoettler #2: also block docker_env from injecting
            # real provider keys.  `docker_env: {OPENROUTER_API_KEY: sk-real}`
            # in config.yaml puts the live secret into the sandbox while
            # egress is nominally enforced — defeats the entire feature.
            # Pull the mapped real_env_name from each token mapping at
            # call time so this stays in sync with whatever the operator
            # has configured.
            _critical_provider_keys: set[str] = set()
            try:
                from agent.proxy_sources import iron_proxy as _ip_for_mappings
                _critical_provider_keys = {
                    m.real_env_name for m in _ip_for_mappings.load_mappings()
                }
            except Exception:  # noqa: BLE001 — best-effort collision check
                pass
            _critical = _critical_proxy_control | _critical_provider_keys
            _collisions = sorted(
                k for k in _critical
                if k in self._env
                and (
                    k not in egress_env_overrides
                    or self._env[k] != egress_env_overrides[k]
                )
                # For provider keys, ANY override is a collision (the egress
                # path mints proxy tokens; a real key in docker_env bypasses
                # the swap regardless of whether the egress dict happens to
                # carry it).
                and (
                    k in _critical_provider_keys
                    or (k in egress_env_overrides
                        and self._env[k] != egress_env_overrides[k])
                )
            )
            if _collisions:
                _msg = (
                    f"docker_env in config.yaml overrides egress-proxy "
                    f"variables {_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these keys from docker_env or "
                        "disable enforce_on_docker to opt out of egress "
                        "isolation."
                    )
                logger.warning(
                    "%s  Falling back to docker_env values; sandbox traffic "
                    "will NOT route through the proxy.", _msg,
                )

        # When enforce_on_docker is true, egress overrides win.  When
        # false, docker_env wins (back-compat for users who deliberately
        # opt out).  In both cases the collision check above has already
        # surfaced any disagreement.
        try:
            from hermes_cli.config import load_config as _load_cfg_for_precedence
            _enforce_egress_merge = bool(
                (_load_cfg_for_precedence().get("proxy") or {})
                .get("enforce_on_docker", True)
            )
        except (ImportError, OSError):
            _enforce_egress_merge = True
        except Exception:  # noqa: BLE001 — yaml.YAMLError or similar
            # Malformed config.yaml; fail-safe to enforced.
            _enforce_egress_merge = True

        if _enforce_egress_merge and egress_env_overrides:
            merged_env = dict(self._env)
            merged_env.update(egress_env_overrides)
        else:
            merged_env = dict(egress_env_overrides)
            merged_env.update(self._env)

        # arshkumarsingh #1: NODE_OPTIONS append-merge.  The egress path
        # wants ``--use-openssl-ca`` so Node routes through the OpenSSL
        # CA store ``SSL_CERT_FILE`` controls.  But the operator's
        # ``docker_env: {NODE_OPTIONS: "--max-old-space-size=8192"}``
        # MUST be preserved — replacing it would silently drop their
        # tuning.  We carry the egress flag in a sentinel key
        # ``_HERMES_EGRESS_NODE_OPTIONS_APPEND`` and merge here.
        _egress_node_append = merged_env.pop(
            "_HERMES_EGRESS_NODE_OPTIONS_APPEND", None,
        )
        if _egress_node_append:
            existing_node = merged_env.get("NODE_OPTIONS", "")
            existing_tokens = existing_node.split()
            # maxpetrusenko P1: dedupe is not enough — the operator may have set
            # a CONFLICTING CA-mode flag (e.g. --use-bundled-ca) that would
            # otherwise survive alongside our --use-openssl-ca, leaving Node's
            # final trust behavior dependent on option order / Node parsing.
            # Egress isolation requires our flag to win deterministically, so
            # strip any known-conflicting CA-mode flags before appending.
            _CA_MODE_FLAGS = {"--use-openssl-ca", "--use-bundled-ca"}
            append_token = _egress_node_append.strip()
            if append_token in _CA_MODE_FLAGS:
                dropped = [t for t in existing_tokens if t in _CA_MODE_FLAGS and t != append_token]
                if dropped:
                    logger.warning(
                        "Overriding conflicting NODE_OPTIONS CA-mode flag(s) %s "
                        "with egress-required %s to keep Node routed through the "
                        "egress CA store.", dropped, append_token,
                    )
                existing_tokens = [t for t in existing_tokens if t not in _CA_MODE_FLAGS or t == append_token]
            # De-dup: only add if not already present (the operator may
            # have set the same flag themselves).
            if append_token not in existing_tokens:
                existing_tokens.append(append_token)
            merged_env["NODE_OPTIONS"] = " ".join(existing_tokens).strip()
            if not merged_env["NODE_OPTIONS"]:
                merged_env.pop("NODE_OPTIONS", None)

        env_args = []
        for key in sorted(merged_env):
            env_args.extend(["-e", f"{key}={merged_env[key]}"])

        # Optional: run the container as the host user so files written into
        # bind-mounted dirs (/workspace, /root, docker_volumes entries) are
        # owned by that user on the host instead of by root. Skip cleanly on
        # platforms without POSIX uid/gid (e.g. native Windows Docker).
        user_args: list[str] = []
        if run_as_host_user:
            user_spec = _resolve_host_user_spec()
            if user_spec is not None:
                user_args = ["--user", user_spec]
                logger.info("Docker: running container as host user %s", user_spec)
            else:
                logger.warning(
                    "docker_run_as_host_user is enabled but this platform does "
                    "not expose POSIX uid/gid; container will start as its "
                    "image default user."
                )
                # Fall back to the full cap set — without --user, an image's
                # init may still need s6-setuidgid/gosu/su to drop privileges.

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"
        image_identity = _resolve_image_identity(self._docker_exe, image)
        self._image_identity = image_identity

        # s6-overlay images (e.g. hermes-agent:latest) already use /init as PID 1
        # and exec /run/s6/basedir/bin/init during startup. For those images we
        # must (a) skip Docker's --init (two competing PID-1 inits) and (b) mount
        # /run with exec instead of noexec, or s6 stage0 dies with exit 126
        # "Permission denied". Detected once here; defaults are kept on any
        # inspection failure. See issue #34628.
        image_uses_s6_init = _image_uses_init_entrypoint(
            self._docker_exe, image_identity
        )
        if image_uses_s6_init:
            logger.info(
                "Docker: image %s uses /init (s6-overlay) as entrypoint — "
                "skipping --init and mounting /run with exec.",
                image,
            )
        security_args = _build_security_args(
            run_as_host_user and bool(user_args),
            run_exec=image_uses_s6_init,
            tmp_storage=tmp_storage,
        )
        if reviewer_mode:
            security_args.append("--read-only")

        logger.info(f"Docker volume_args: {volume_args}")
        # User-supplied extra docker run flags (docker_extra_args in config.yaml).
        # Appended last so they can override defaults if needed.
        validated_extra = []
        for arg in (extra_args or []):
            if not isinstance(arg, str):
                logger.warning("Ignoring non-string docker_extra_args entry: %r", arg)
                continue
            validated_extra.append(arg)
        if _extra_args_have_host_security_file(validated_extra):
            raise ValueError(
                "docker_extra_args host-backed security-opt files are unsupported "
                "because their contents cannot be authenticated"
            )
        if any(
            arg == "--volumes-from" or arg.startswith("--volumes-from=")
            for arg in validated_extra
        ):
            raise ValueError(
                "docker_extra_args mounts /workspace or /tmp through an opaque "
                "donor with --volumes-from; this cannot participate in container "
                "reuse isolation. use docker_tmp_storage and explicit "
                "docker_volumes instead"
            )
        if _extra_args_mount_workspace(validated_extra):
            raise ValueError(
                "docker_extra_args mounts /workspace or its subdirectories; "
                "this is not allowed in raw arguments. "
                "use docker_volumes or docker_mount_cwd_to_workspace so the mount "
                "participates in container reuse isolation"
            )
        if _extra_args_mount_tmp(validated_extra):
            raise ValueError(
                "docker_extra_args cannot mount /tmp or its subdirectories; "
                "use docker_tmp_storage to select the /tmp policy"
            )
        if _extra_args_have_host_bind(validated_extra):
            raise ValueError(
                "docker_extra_args host bind sources are unsupported because raw "
                "mounts cannot participate in policy authentication; use docker_volumes"
            )
        if _extra_args_override_network(validated_extra):
            raise ValueError(
                "docker_extra_args cannot select a network mode; use "
                "terminal.docker_network so network policy participates in reuse isolation"
            )
        reserved_label_collisions = _extra_args_reserved_label_collisions(validated_extra)
        if reserved_label_collisions:
            raise ValueError(
                "docker_extra_args cannot override reserved Hermes labels: "
                + ", ".join(reserved_label_collisions)
            )
        if egress_env_overrides:
            _extra_collisions = _extra_args_egress_collisions(
                validated_extra, _critical_egress_names,
            )
            if _extra_collisions:
                _msg = (
                    f"docker_extra_args would override egress-proxy controls "
                    f"{_extra_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these args or disable enforce_on_docker "
                        "to opt out of egress isolation."
                    )
                logger.warning(
                    "%s  Extra Docker args may bypass egress isolation.", _msg,
                )

        bind_source_identities = _volume_source_identities(
            volume_args,
            canonical_workspace=canonical_workspace_identity is not None,
        )
        if canonical_workspace_identity is None:
            for identity in bind_source_identities:
                if "content_sha256" in identity:
                    self._readonly_workspace_sources.append(
                        (
                            str(identity["path"]),
                            str(identity["destination"]),
                            identity,
                        )
                    )

        if expected_git_sha is not None and not self._readonly_workspace_sources:
            raise ValueError(
                "an assigned reviewer Git SHA requires an authenticated read-only workspace"
            )

        # Exact-SHA reviewers require immutable bytes. Ordinary read-only mounts
        # retain their historical live-bind semantics.
        sources_to_materialize = (
            list(self._readonly_workspace_sources) if reviewer_mode else []
        )
        for source, destination, expected in sources_to_materialize:
            snapshot_volume = _materialize_readonly_workspace(
                self._docker_exe,
                image_identity,
                source,
                expected,
                expected_git_sha,
                expected_content_sha256,
                disposable=reviewer_mode,
                provenance_deadline=reviewer_provenance_deadline,
            )
            if reviewer_mode:
                self._snapshot_volumes.append(snapshot_volume)
            for index, arg in enumerate(volume_args[:-1]):
                if arg != "-v":
                    continue
                spec = volume_args[index + 1]
                if _volume_targets_exact_path(spec, destination) and "ro" in spec.rsplit(
                    ":", 1
                )[-1].split(","):
                    volume_args[index + 1] = f"{snapshot_volume}:{destination}:ro"
                    break
            else:
                raise RuntimeError(
                    f"cannot locate authenticated read-only workspace mount: {destination}"
                )
        if reviewer_mode:
            self._readonly_workspace_sources = [
                ("", destination, expected)
                for _, destination, expected in self._readonly_workspace_sources
            ]
        else:
            # A general read-only bind remains a live view by design. Its source
            # identity still participates in reuse fingerprinting, but subsequent
            # host-side updates are not treated as reviewer provenance failures.
            self._readonly_workspace_sources = []

        all_run_args = (
            security_args
            + user_args
            + writable_args
            + resource_args
            + egress_host_args
            + volume_args
            + env_args
            + validated_extra
        )
        extra_file_identities = _extra_arg_file_identities(validated_extra)
        policy_payload = {
            "version": 1,
            "image": image,
            "image_identity": image_identity,
            "cwd": effective_cwd,
            "requested_cwd": requested_cwd,
            "image_uses_s6_init": image_uses_s6_init,
            "persistent_filesystem": self._persistent,
            "reviewer_mode": reviewer_mode,
            "expected_git_sha": expected_git_sha,
            "run_args": all_run_args,
            "bind_sources": bind_source_identities,
            "canonical_workspace": canonical_workspace_identity,
            "extra_files": extra_file_identities,
        }
        policy_label = hashlib.sha256(
            json.dumps(
                policy_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:24]
        logger.info(f"Docker run_args: {all_run_args}")

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        # Labels make hermes-created containers identifiable to:
        #   * the orphan reaper (`hermes-agent=1` for the global sweep filter)
        #   * future cross-process reuse (`hermes-task-id`, `hermes-profile`)
        #   * operators running `docker ps --filter label=hermes-agent=1`
        # Values are limited to the safe character set defined by
        # _sanitize_label_value(); the active Hermes profile is captured at
        # container-start time and never changes for the container's lifetime.
        profile_name = _sanitize_label_value(_get_active_profile_name())
        task_label = _sanitize_label_value(task_id)
        label_args = [
            "--label", "hermes-agent=1",
            "--label", f"hermes-task-id={task_label}",
            "--label", f"hermes-profile={profile_name}",
            "--label", f"{_EGRESS_LABEL_KEY}={egress_label}",
            "--label", f"{_WORKSPACE_LABEL_KEY}={workspace_label}",
            "--label", f"{_TMP_STORAGE_LABEL_KEY}={tmp_storage}",
            "--label", f"{_POLICY_LABEL_KEY}={policy_label}",
        ]
        # Save args for container recreation on "No such container" recovery.
        # Recovery and initial creation both use the authenticated immutable ID;
        # the human-readable reference remains only in the policy payload/logs.
        self._image = image_identity
        self._container_name = container_name
        self._image_uses_s6_init = image_uses_s6_init
        self._all_run_args = all_run_args

        self._labels = {
            "hermes-agent": "1",
            "hermes-task-id": task_label,
            "hermes-profile": profile_name,
            _EGRESS_LABEL_KEY: egress_label,
            _WORKSPACE_LABEL_KEY: workspace_label,
            _TMP_STORAGE_LABEL_KEY: tmp_storage,
            _POLICY_LABEL_KEY: policy_label,
        }

        # Cross-process container reuse (issue #20561 — docs claim "ONE long-lived
        # container shared across sessions").  If a prior Hermes process
        # already started a container for this (task_id, profile) and it
        # still exists, attach to it instead of starting a fresh one.  This
        # restores the documented contract; opt out via
        # ``terminal.docker_persist_across_processes: false``.
        #
        # Reuse matches on labels only.  The egress posture gets its own label
        # because env vars, CA mounts, and host mappings are immutable after
        # container creation — reusing a pre-egress or pre-rotation container
        # would silently bypass the credential firewall.
        reused = False
        if self._persist_across_processes:
            existing = self._find_reusable_container(
                task_label, profile_name, egress_label, workspace_label, tmp_storage,
                policy_label,
            )
            if existing is not None:
                container_id, state = existing
                # Network-mode guard: reuse must not silently defeat an
                # egress lockdown.  A container created before the operator
                # set ``docker_network: false`` keeps its original bridge
                # NetworkMode, so label-only reuse would hand the agent a
                # networked container despite the config.  On mismatch we
                # remove the stale container and start fresh — leaving it in
                # place would let the next label-based reuse pick it up again.
                # Raw network selection is rejected, so the effective mode must
                # match in both directions: enabled must not reuse ``none``, and
                # disabled must reuse only ``none``.
                actual_mode = self._container_network_mode(container_id)
                mode_mismatch = (
                    actual_mode is None
                    or (not network and actual_mode != "none")
                    or (network and actual_mode == "none")
                )
                if mode_mismatch:
                    logger.warning(
                        "Existing container %s has NetworkMode=%s but the requested "
                        "docker_network policy requires %s — removing it and starting "
                        "fresh (task=%s, profile=%s).",
                        container_id[:12], actual_mode or "unknown",
                        "network access" if network else "an air-gapped container",
                        task_label, profile_name,
                    )
                    try:
                        subprocess.run(
                            [self._docker_exe, "rm", "-f", container_id],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=30,
                            check=False,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.TimeoutExpired, OSError) as e:
                        logger.warning("Failed to remove mismatched container %s: %s", container_id[:12], e)
                    existing = None
            if existing is not None:
                container_id, state = existing
                self._container_id = container_id
                if state != "running":
                    try:
                        subprocess.run(
                            [self._docker_exe, "start", container_id],
                            capture_output=True,
                            text=True, encoding='utf-8', errors='replace',
                            timeout=30,
                            check=True,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(
                            "Failed to start existing container %s (state=%s): "
                            "%s — falling back to a fresh container.",
                            container_id[:12], state, e,
                        )
                        self._container_id = None
                if self._container_id:
                    logger.info(
                        "Reusing container %s (task=%s, profile=%s, prior state=%s)",
                        container_id[:12], task_label, profile_name, state,
                    )
                    reused = True

        if not reused:
            # tini/catatonit as PID 1 reaps zombie children — but s6-overlay
            # images already provide their own /init PID 1, so adding --init
            # there creates two competing inits and breaks startup (#34628).
            init_args = [] if image_uses_s6_init else ["--init"]
            run_cmd = [
                self._docker_exe, "run", "-d",
                *init_args,
                "--name", container_name,
                *label_args,
                "-w", startup_cwd,
                *all_run_args,
                image_identity,
                "sleep", "infinity",  # no fixed lifetime — idle reaper handles cleanup
            ]
            logger.debug(f"Starting container: {' '.join(run_cmd)}")
            try:
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=120,  # image pull may take a while
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except BaseException as e:
                # Docker may create the container object before `docker run`
                # fails to start it, including client-side OSError failures.
                # Remove by known name because no container id may be returned.
                logger.warning(
                    "docker run failed for %s, cleaning up orphaned container: %s",
                    container_name, e,
                )
                try:
                    subprocess.run(
                        [self._docker_exe, "rm", "-f", container_name],
                        capture_output=True, timeout=10, check=False,
                        stdin=subprocess.DEVNULL,
                    )
                finally:
                    self._remove_snapshot_volumes()
                raise
            self._container_id = result.stdout.strip()
            logger.info(f"Started container {container_name} ({self._container_id[:12]})")

        policy_error = self._effective_policy_violation(self._container_id)
        if policy_error:
            container_id = self._container_id
            assert container_id is not None
            if self._remove_rejected_container(container_id):
                self._container_id = None
            raise RuntimeError(policy_error)

        if reviewer_mode:
            self._initialize_reviewer_copy(self._container_id, expected_git_sha)

        # Build the init-time env forwarding args (used only by init_session
        # to inject host env vars into the snapshot; subsequent commands get
        # them from the snapshot file).
        self._init_env_args = self._build_init_env_args()

        # Initialize session snapshot inside the container. Construction has
        # already acquired a running container and (for reviewers) a uniquely
        # owned snapshot volume, so a late failure must roll both back here:
        # callers never receive an environment on which they could call cleanup.
        try:
            self.init_session()
        except BaseException:
            # Exact-SHA reviewer resources are always newly created and
            # disposable. Ordinary environments may be attached to a reused
            # persistent container; preserve their historical cleanup contract
            # rather than destroying an existing container on an init hiccup.
            if not reviewer_mode:
                raise
            container_id = self._container_id
            try:
                if container_id:
                    subprocess.run(
                        [self._docker_exe, "rm", "-f", "-v", container_id],
                        capture_output=True, timeout=30, check=False,
                        stdin=subprocess.DEVNULL,
                    )
                    self._container_id = None
            finally:
                self._remove_snapshot_volumes()
            raise

    def _build_init_env_args(self) -> list[str]:
        """Build -e KEY=VALUE args for injecting host env vars into init_session.

        These are used once during init_session() so that export -p captures
        them into the snapshot.  Subsequent execute() calls don't need -e flags.
        """
        if getattr(self, "_reviewer_mode", False):
            return []
        exec_env: dict[str, str] = dict(self._env)

        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        try:
            from tools.env_passthrough import get_all_passthrough
            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass
        # Explicit docker_forward_env entries are an intentional opt-in and must
        # win over the generic Hermes secret blocklist. Only implicit passthrough
        # keys are filtered. Also strip Hermes-internal dynamic secrets
        # (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_* auth) that the
        # name-based blocklist doesn't cover — see _is_hermes_internal_secret.
        _implicit_forward = {
            k for k in passthrough_keys if not _is_hermes_internal_secret(k)
        }
        forward_keys = explicit_forward_keys | (_implicit_forward - _HERMES_PROVIDER_ENV_BLOCKLIST)
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        for key in sorted(forward_keys):
            value = os.getenv(key)
            if not value:
                value = hermes_env.get(key)
            if value:
                exec_env[key] = value

        args = []
        for key in sorted(exec_env):
            args.extend(["-e", f"{key}={exec_env[key]}"])
        return args

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Docker container."""
        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")

        # Only inject -e env args during init_session (login=True).
        # Subsequent commands get env vars from the snapshot.
        if login:
            cmd.extend(self._init_env_args)

        cmd.extend([self._container_id])

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    # ------------------------------------------------------------------
    # "No such container" recovery (issue #36266)
    # ------------------------------------------------------------------

    _NO_CONTAINER_PATTERNS = (
        "No such container",
        "is not running",
        "no such container",
    )

    def _is_container_gone(self, output: str) -> bool:
        """Return True if the output indicates the container no longer exists."""
        return any(p in output for p in self._NO_CONTAINER_PATTERNS)

    def _recreate_container(self) -> bool:
        """Recreate the container after it was removed out-of-band.

        Tries label-based reuse first; if no existing container is found,
        starts a fresh one with the same image and run-args.  Returns True
        on success, False if recreation fails (caller should surface the
        original error).
        """
        old_id = (self._container_id or "")[:12]
        try:
            current_image_identity = _resolve_image_identity(
                self._docker_exe, self._image
            )
        except RuntimeError as exc:
            logger.error("Recovery cannot authenticate image identity: %s", exc)
            return False
        if current_image_identity != self._image_identity:
            logger.error(
                "Recovery rejected mutable image %s: identity changed from %s to %s",
                self._image, self._image_identity, current_image_identity,
            )
            return False
        logger.warning(
            "Container %s appears to be gone — attempting recovery", old_id,
        )
        self._container_id = None

        # 1. Try label-based reuse (another process may have recreated it).
        task_label = self._labels.get("hermes-task-id", "")
        profile_label = self._labels.get("hermes-profile", "")
        existing = self._find_reusable_container(
            task_label, profile_label, self._labels.get(_EGRESS_LABEL_KEY, "off"),
            self._labels.get(_WORKSPACE_LABEL_KEY, "off"),
            self._labels.get(_TMP_STORAGE_LABEL_KEY, "tmpfs"),
            self._labels.get(_POLICY_LABEL_KEY, ""),
        )
        if existing is not None:
            cid, state = existing
            if state == "running":
                self._container_id = cid
                logger.info("Recovery: reusing running container %s", cid[:12])
            else:
                try:
                    subprocess.run(
                        [self._docker_exe, "start", cid],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, check=True,
                        stdin=subprocess.DEVNULL,
                    )
                    self._container_id = cid
                    logger.info("Recovery: restarted container %s", cid[:12])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning("Recovery: failed to start container %s: %s", cid[:12], e)

        # 2. No reusable container — create a fresh one.
        if not self._container_id:
            if not self._image:
                logger.error("Recovery: no saved image name, cannot recreate container")
                return False
            try:
                import uuid as _uuid
                new_name = f"hermes-{_uuid.uuid4().hex[:8]}"
                init_args = [] if self._image_uses_s6_init else ["--init"]
                label_args = []
                for k, v in self._labels.items():
                    label_args.extend(["--label", f"{k}={v}"])
                run_cmd = [
                    self._docker_exe, "run", "-d",
                    *init_args,
                    "--name", new_name,
                    *label_args,
                    "-w", self.cwd,
                    *self._all_run_args,
                    self._image,
                    "sleep", "infinity",
                ]
                result = subprocess.run(
                    run_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                )
                self._container_id = result.stdout.strip()
                self._container_name = new_name
                logger.info(
                    "Recovery: created fresh container %s (%s)",
                    new_name, self._container_id[:12],
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                logger.error("Recovery: failed to create new container: %s", e)
                return False

        # 3. Revalidate immutable policies before using the recovered container.
        mode = self._container_network_mode(self._container_id)
        mode_mismatch = (
            mode is None
            or (not self._network_enabled and mode != "none")
            or (self._network_enabled and mode == "none")
        )
        if mode_mismatch:
            logger.error(
                "Recovery rejected container %s with NetworkMode=%s",
                self._container_id[:12], mode or "unknown",
            )
            self._remove_rejected_container(self._container_id)
            self._container_id = None
            return False
        policy_error = self._effective_policy_violation(self._container_id)
        if policy_error:
            logger.error("Recovery rejected container: %s", policy_error)
            self._remove_rejected_container(self._container_id)
            self._container_id = None
            return False

        # 4. Re-initialize session snapshot in the (re)created container.
        try:
            self._snapshot_ready = False
            self.init_session()
        except Exception as e:
            logger.error("Recovery: init_session failed in new container: %s", e)
            return False

        logger.info("Recovery successful — new container %s", (self._container_id or "")[:12])
        return True

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Execute a command, auto-recovering from dead containers.

        If the container was removed out-of-band (idle reaper, docker prune,
        OOM kill, daemon restart), detect the error and recreate the container
        transparently before retrying once.
        """
        # Mutable host binds require before/after checks. Materialized snapshots
        # are authenticated once at creation/reuse and mounted from a daemon
        # volume read-only, so hashing a large repository twice per command is
        # unnecessary and makes every terminal/file operation O(tree size).
        revalidate_each_command = any(
            source for source, _destination, _expected in self._readonly_workspace_sources
        )
        if revalidate_each_command:
            workspace_error = self._readonly_workspace_identity_violation()
            if workspace_error:
                return {"output": workspace_error, "returncode": 126}
        result = super().execute(command, cwd, **kwargs)
        if (
            result.get("returncode", 0) != 0
            and self._is_container_gone(result.get("output", ""))
            and self._persist_across_processes
        ):
            if self._recreate_container():
                result = super().execute(command, cwd, **kwargs)
        if revalidate_each_command:
            workspace_error = self._readonly_workspace_identity_violation()
            if workspace_error:
                return {
                    "output": f"{workspace_error}; changed during command execution",
                    "returncode": 126,
                }
        return result

    def _readonly_workspace_identity_violation(self) -> Optional[str]:
        """Detect host-side mutation of an authenticated read-only bind.

        Docker's ``ro`` protects the source from the container, not from other
        host processes. Re-authenticate immediately before every command so a
        long-lived reviewer container cannot silently observe a different tree
        under the same creation-policy label.
        """
        for source, destination, expected in self._readonly_workspace_sources:
            if source:
                try:
                    current = _path_identity(source, metadata_digest=True)
                except (OSError, ValueError) as exc:
                    return f"cannot authenticate read-only workspace {source}: {exc}"
                if any(expected.get(key) != value for key, value in current.items()):
                    return (
                        "read-only workspace changed after container policy "
                        f"authentication: {source}"
                    )
            if self._container_id and "content_sha256" in expected:
                try:
                    mounted_digest = _container_tree_digest(
                        self._docker_exe,
                        self._container_id,
                        destination,
                    )
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    return (
                        "cannot authenticate mounted read-only workspace "
                        f"{destination}: {exc}"
                    )
                if mounted_digest != expected["mounted_content_sha256"]:
                    return (
                        "mounted read-only workspace differs from authenticated "
                        f"source: {destination}"
                    )
        return None

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = subprocess.run(
                [docker, "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
                stdin=subprocess.DEVNULL,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                [docker, "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5,
                                   stdin=subprocess.DEVNULL)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def _container_network_mode(self, container_id: str) -> Optional[str]:
        """Return the container's ``HostConfig.NetworkMode`` (e.g. ``bridge``,
        ``none``, ``host``), or ``None`` when inspection fails.

        Used by the reuse path to make sure a persisted container's network
        mode still matches the operator's ``docker_network`` setting; callers
        treat ``None`` (unknown) as a mismatch when lockdown was requested,
        so a failed inspect fails closed rather than open.
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "inspect",
                    "--format", "{{.HostConfig.NetworkMode}}",
                    container_id,
                ],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker inspect NetworkMode failed: %s", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker inspect NetworkMode returned %d: %s",
                result.returncode, result.stderr.strip(),
            )
            return None
        mode = result.stdout.strip()
        return mode or None

    def _container_has_mount_at_or_below(
        self,
        container_id: str,
        container_path: str,
        *,
        include_root: bool = True,
        include_ancestors: bool = False,
        writable_only: bool = False,
    ) -> bool:
        """Fail closed if effective Docker mounts overlap a protected path.

        ``include_ancestors`` is needed for symlink-resolved protected paths:
        a mount at ``/workspace`` also controls a resolved path such as
        ``/workspace/tmp`` even though its destination is not below that path.
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe,
                    "inspect",
                    "--format",
                    "{{json .Mounts}}",
                    container_id,
                ],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=30,
                check=True,
                stdin=subprocess.DEVNULL,
            )
            mounts = json.loads(result.stdout)
            if not isinstance(mounts, list):
                return True
            root = posixpath.normpath("/" + container_path.lstrip("/"))
            for mount in mounts:
                if not isinstance(mount, dict):
                    continue
                destination = mount.get("Destination")
                if not isinstance(destination, str):
                    continue
                canonical = posixpath.normpath("/" + destination.lstrip("/"))
                if _container_path_is_at_or_below(
                    destination, container_path
                ) or (
                    include_ancestors
                    and _container_path_is_at_or_below(container_path, destination)
                ):
                    if include_root or canonical != root:
                        if not writable_only or mount.get("RW") is not False:
                            return True
            return False
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
            json.JSONDecodeError,
        ) as e:
            logger.warning(
                "Could not verify effective mounts for container %s: %s",
                container_id[:12],
                e,
            )
            return True

    def _container_resolved_path(
        self, container_id: str, container_path: str
    ) -> Optional[str]:
        """Resolve a protected path inside the running image, failing closed."""
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "exec", container_id,
                    "readlink", "-f", "--", container_path,
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, check=False, stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Could not resolve %s in container: %s", container_path, exc)
            return None
        resolved = result.stdout.strip()
        if result.returncode != 0 or not resolved.startswith("/"):
            logger.warning(
                "Could not resolve %s in container %s: %s",
                container_path, container_id[:12], result.stderr.strip(),
            )
            return None
        return posixpath.normpath(resolved)

    def _initialize_reviewer_copy(self, container_id: str, expected_git_sha: str) -> None:
        """Replace stale reviewer state and verify a writable exact-SHA copy."""
        script = r'''
import os, pathlib, shutil, subprocess, sys
source = pathlib.Path('/workspace')
target = pathlib.Path('/tmp/review')
if target.is_symlink() or target.is_file():
    target.unlink()
elif target.exists():
    shutil.rmtree(target)
target.mkdir(mode=0o700)
for child in source.iterdir():
    destination = target / child.name
    if child.is_dir() and not child.is_symlink():
        shutil.copytree(child, destination, symlinks=True)
    elif child.is_symlink():
        destination.symlink_to(os.readlink(child))
    else:
        shutil.copy2(child, destination, follow_symlinks=False)
safe_env = {
    'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C',
    'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_NO_LAZY_FETCH': '1',
}
result = subprocess.run(
    ['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
     'rev-parse', '--verify', 'HEAD^{commit}'],
    cwd=target, capture_output=True, text=True, encoding='utf-8',
    errors='replace', timeout=30, check=False, stdin=subprocess.DEVNULL,
    env=safe_env,
)
if result.returncode != 0 or result.stdout.strip() != sys.argv[1]:
    raise RuntimeError('writable reviewer copy does not match assigned SHA')
clean = subprocess.run(
    ['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
     'status', '--porcelain=v1', '--untracked-files=all'],
    cwd=target, capture_output=True, text=True, encoding='utf-8',
    errors='replace', timeout=30, check=False, stdin=subprocess.DEVNULL,
    env=safe_env,
)
if clean.returncode != 0 or clean.stdout:
    raise RuntimeError('writable reviewer copy is not the clean assigned tree')
'''
        try:
            result = subprocess.run(
                [self._docker_exe, "exec", container_id, "python3", "-I", "-c",
                 script, expected_git_sha],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, check=False, stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            if self._remove_rejected_container(container_id):
                self._container_id = None
            raise RuntimeError("failed to create writable reviewer copy") from exc
        if result.returncode != 0:
            detail = result.stderr.strip()
            if self._remove_rejected_container(container_id):
                self._container_id = None
            raise RuntimeError(
                f"failed to create writable reviewer copy at assigned SHA: {detail}"
            )

    def _unexpected_reviewer_writable_mount(self, container_id: str) -> Optional[str]:
        """Return a writable mount outside the fixed disposable scratch roots."""
        allowed = {"/tmp", "/var/tmp", "/run", "/home", "/root"}
        try:
            result = subprocess.run(
                [self._docker_exe, "inspect", "--format", "{{json .Mounts}}", container_id],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, check=True, stdin=subprocess.DEVNULL,
            )
            mounts = json.loads(result.stdout)
            if not isinstance(mounts, list):
                return "unknown"
            for mount in mounts:
                if not isinstance(mount, dict) or mount.get("RW") is False:
                    continue
                destination = mount.get("Destination")
                if not isinstance(destination, str):
                    return "unknown"
                canonical = posixpath.normpath("/" + destination.lstrip("/"))
                if canonical not in allowed:
                    return canonical
                # Image-provided symlinks must not turn an allowlisted scratch
                # mount (for example /root) into a writable mount inside the
                # authoritative /workspace tree.  Require every allowlisted
                # destination to resolve to itself, just as /tmp is checked
                # below for the general storage policy.
                resolved = self._container_resolved_path(container_id, canonical)
                if resolved != canonical:
                    return canonical
            return None
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
            json.JSONDecodeError,
        ):
            return "unknown"

    def _effective_policy_violation(self, container_id: str) -> Optional[str]:
        """Return a fail-closed effective-mount policy error, if any."""
        workspace_error = self._readonly_workspace_identity_violation()
        if workspace_error:
            return workspace_error
        if self._reviewer_mode:
            unexpected_mount = self._unexpected_reviewer_writable_mount(container_id)
            if unexpected_mount is not None:
                return (
                    "assigned reviewer container has unexpected writable mount: "
                    f"{unexpected_mount}"
                )
        if self._tmp_storage == "disk":
            resolved_tmp = self._container_resolved_path(container_id, "/tmp")
            # Requiring the path itself (not merely its current final target) to
            # live on the writable layer closes a mutable-symlink race.  An
            # intermediate link inside an image-declared volume could otherwise
            # be redirected after this one-time policy check.
            if resolved_tmp != "/tmp" or self._container_has_mount_at_or_below(
                container_id, "/tmp", include_ancestors=True
            ):
                return (
                    "docker_tmp_storage=disk requires /tmp on the container "
                    "writable layer, but the effective image/container declares "
                    "a mount at /tmp or resolves /tmp through a symlink"
                )
        else:
            resolved_tmp = self._container_resolved_path(container_id, "/tmp")
            if resolved_tmp is None or self._container_has_mount_at_or_below(
                container_id, "/tmp", include_root=False
            ) or (
                resolved_tmp != "/tmp"
                and self._container_has_mount_at_or_below(
                    container_id,
                    resolved_tmp,
                    include_root=False,
                    include_ancestors=True,
                )
            ):
                return "effective container mounts bypass the hardened /tmp tmpfs"
        if self._workspace_requires_ro and self._container_has_mount_at_or_below(
            container_id,
            "/workspace",
            include_root=False,
            writable_only=True,
        ):
            return "effective container mounts bypass the read-only /workspace"
        if self._workspace_requires_ro:
            resolved_workspace = self._container_resolved_path(container_id, "/workspace")
            if resolved_workspace != "/workspace":
                return "effective container path bypasses the read-only /workspace"
        return None

    def _remove_rejected_container(self, container_id: str) -> bool:
        """Best-effort bounded cleanup that never masks a policy failure."""
        try:
            result = subprocess.run(
                [self._docker_exe, "rm", "-f", "-v", container_id],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error(
                "Failed to remove rejected container %s: %s", container_id[:12], e
            )
            return False
        if result.returncode != 0:
            logger.error(
                "Failed to remove rejected container %s: %s",
                container_id[:12], result.stderr.strip(),
            )
            return False
        self._remove_snapshot_volumes()
        return True

    def _remove_snapshot_volumes(self) -> None:
        """Best-effort removal of uniquely owned reviewer snapshot volumes."""
        volumes = list(getattr(self, "_snapshot_volumes", []))
        self._snapshot_volumes = []
        for volume in volumes:
            try:
                subprocess.run(
                    [self._docker_exe, "volume", "rm", "-f", volume],
                    capture_output=True, timeout=30, check=False,
                    stdin=subprocess.DEVNULL,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("docker volume rm -f %s failed: %s", volume, exc)

    def _find_reusable_container(
        self,
        task_label: str,
        profile_label: str,
        egress_label: str,
        workspace_label: str = "off",
        tmp_storage: str = "tmpfs",
        policy_label: str = "",
    ) -> Optional[tuple[str, str]]:
        """Look for an existing container labeled for this (task, profile).

        Returns ``(container_id, state)`` on hit, ``None`` on miss / on any
        failure (including ``docker ps`` itself failing). State is one of the
        values Docker reports via ``{{.State}}`` — e.g. ``running``, ``exited``,
        ``created``, ``paused``, ``restarting``, ``dead``. The caller decides
        whether the state warrants ``docker start`` before reuse.

        Restricted to the docker-stored label set this class creates; never
        matches containers that happened to be named ``hermes-*`` but were
        started by some other tool.
        """
        try:
            filters = [
                "--filter", "label=hermes-agent=1",
                "--filter", f"label=hermes-task-id={task_label}",
                "--filter", f"label=hermes-profile={profile_label}",
                "--filter", f"label={_TMP_STORAGE_LABEL_KEY}={tmp_storage}",
                # Workspace mounts and their access mode are immutable. Match
                # even "off" so removing a mount cannot reuse stale host access.
                "--filter", f"label={_WORKSPACE_LABEL_KEY}={workspace_label}",
                "--filter", f"label={_POLICY_LABEL_KEY}={policy_label}",
            ]
            if egress_label != "off":
                filters.extend(["--filter", f"label={_EGRESS_LABEL_KEY}={egress_label}"])
                fmt = "{{.ID}}\t{{.State}}"
            else:
                # When egress is off, we widen the probe to find any
                # task+profile container (regardless of egress label), then
                # post-filter in Python: reject containers whose
                # hermes-egress label is present and not "off".  Without
                # this, a container created with egress=on can be silently
                # reused after the operator runs "hermes egress disable",
                # preserving baked-in proxy env and CA mounts.
                fmt = '{{.ID}}\t{{.State}}\t{{.Label "' + _EGRESS_LABEL_KEY + '"}}'
            result = subprocess.run(
                [
                    self._docker_exe, "ps", "-a",
                    *filters,
                    "--format", fmt,
                ],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker ps probe failed: %s — will start a fresh container", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker ps probe returned %d: %s — will start a fresh container",
                result.returncode, result.stderr.strip(),
            )
            return None
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        # Multiple matches are unusual (one (task, profile) should produce one
        # container) but can happen if a previous Hermes process crashed
        # mid-cleanup. Prefer a running one if present; otherwise pick the
        # first listed. Stale duplicates get reaped by the orphan-reaper in a
        # follow-up commit; we don't try to be heroic about them here.
        running = None
        first = None
        for ln in lines:
            if egress_label == "off":
                # Format: ID\tState\tEgressLabel — parse all three fields
                # and reject containers with a non-off egress label.
                parts = ln.split("\t", 2)
                if len(parts) < 3:
                    continue
                cid, state, egress_val = parts[0], parts[1].lower(), parts[2]
                if egress_val not in ("", "<no value>", "off"):
                    logger.debug(
                        "skipping container %s for egress=off reuse: "
                        "label %s=%r", cid, _EGRESS_LABEL_KEY, egress_val,
                    )
                    continue
            else:
                parts = ln.split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, state = parts[0], parts[1].lower()
            if first is None:
                first = (cid, state)
            if state == "running" and running is None:
                running = (cid, state)
        return running or first

    def cleanup(self, *, force_remove: bool = False):
        """Tear down the container according to persist mode and *force_remove*.

        Persist-mode (``persist_across_processes=True``, the default) leaves the
        container **running** untouched. The docs promise "ONE long-lived
        container shared across sessions" and stopping it on every Hermes exit
        breaks that promise:

        * Background processes inside the container (``npm run dev``, watchers,
          long-running pytest) get killed every time the user runs ``/quit``.
        * Every reuse requires ``docker start`` + waiting for the container to
          come back up, adding 1–2s to the first tool call of the new session.
        * The user-visible difference between "ONE long-lived container" and
          "a new container that happens to share state" is exactly this:
          processes survive in the former, die in the latter.

        Resource reclamation for the persist-mode case lives in the
        ``reap_orphan_containers()`` path (see issue #20561 commit 3): if no
        Hermes process touches a labeled container for ``2 × lifetime_seconds``
        it gets ``docker rm -f``'d at the next Hermes startup. That covers the
        SIGKILL / OOM / abandoned-laptop cases without us needing to stop the
        container on every graceful exit.

        Opt-out mode (``persist_across_processes=False``) still does
        ``docker stop`` + ``docker rm -f`` on every cleanup, matching the
        pre-PR behavior for users who explicitly want per-process isolation.

        ``force_remove=True`` overrides persist mode and always tears the
        container down (``docker stop`` + ``docker rm -f``). This is the
        explicit-teardown path for ``/reset``, ``cleanup_vm(task_id)``-driven
        resets, or any caller that wants a guaranteed fresh container on next
        ``DockerEnvironment(task_id=...)``. No current caller passes
        ``force_remove=True``; the parameter is here so the explicit-teardown
        semantics can be wired up later without changing this method's
        signature.

        Cleanup runs on a daemon thread with bounded ``subprocess.run`` calls
        (not the racy ``Popen(... &)`` pattern from before PR #33645). The
        atexit hook in ``tools/terminal_tool.py`` waits up to 15s for the
        thread to finish before the interpreter exits, so ``docker stop`` /
        ``docker rm`` actually completes when we do trigger it.
        """
        container_id = self._container_id
        if not container_id:
            # Construction can fail after a reviewer snapshot is materialized
            # but before a container handle is established. Do not make
            # snapshot cleanup contingent on the container existing.
            self._remove_snapshot_volumes()
            # Still drop the bind-mount dirs if any were allocated and we're
            # NOT in persist mode (persist mode preserves them).
            if not self._persistent:
                for d in (self._workspace_dir, self._home_dir):
                    if d:
                        shutil.rmtree(d, ignore_errors=True)
            return

        # Decide what to actually do. Three cases:
        #
        #   force_remove=True             → stop + rm (explicit teardown)
        #   persist_across_processes=True → no-op (leave container running)
        #   persist_across_processes=False → stop + rm (per-process isolation)
        #
        # The persist-mode no-op is the issue-#20561 contract: the container
        # outlives Hermes processes, processes inside it stay alive, and
        # reuse on next startup is instant.
        if force_remove:
            should_stop = True
            should_remove = True
        elif self._persist_across_processes:
            # No-op for the container. Drop the in-process handle so a fresh
            # __init__ will re-probe via labels (and find the running
            # container) instead of trying to reuse a stale Python reference.
            self._container_id = None
            return
        else:
            should_stop = True
            should_remove = True

        # Capture state needed by the worker before we null out the attrs —
        # the worker thread can outlive ``self``.
        docker_exe = self._docker_exe
        log_id = container_id[:12]
        snapshot_volumes = list(getattr(self, "_snapshot_volumes", []))
        self._snapshot_volumes = []

        def _do_cleanup() -> None:
            if should_stop:
                try:
                    subprocess.run(
                        [docker_exe, "stop", "-t", "10", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker stop %s timed out / failed: %s", log_id, e)
            if should_remove:
                try:
                    subprocess.run(
                        [docker_exe, "rm", "-f", "-v", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker rm -f %s failed: %s", log_id, e)
                for volume in snapshot_volumes:
                    try:
                        subprocess.run(
                            [docker_exe, "volume", "rm", "-f", volume],
                            capture_output=True, timeout=30, check=False,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.TimeoutExpired, OSError) as e:
                        logger.warning("docker volume rm -f %s failed: %s", volume, e)

        # Daemon thread: doesn't block interpreter exit (atexit returns
        # promptly), but unlike the old ``Popen(... &)`` shell trick the
        # Python-level join semantics let the thread actually run to
        # completion if the interpreter is still alive. atexit registers
        # ``_atexit_cleanup`` in terminal_tool.py which waits up to ~60s for
        # outstanding cleanups, so most exits complete the work cleanly.
        import threading
        t = threading.Thread(target=_do_cleanup, daemon=True, name=f"hermes-cleanup-{log_id}")
        t.start()
        self._cleanup_thread = t
        self._container_id = None

        # Bind-mount dir teardown only runs when we actually removed the
        # container (the dirs are the container's filesystem state; keeping
        # them around with no container would orphan the data on disk).
        if should_remove and not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        """Block up to *timeout* seconds for the cleanup worker thread.

        Returns ``True`` if the thread finished (or no thread was started),
        ``False`` on timeout. The atexit hook in terminal_tool.py calls this
        on every active environment so docker stop/rm actually completes
        before the Python process exits — without this, ``hermes /quit``
        races the interpreter shutdown and leaves stopped containers behind.
        """
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
