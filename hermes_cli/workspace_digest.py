"""Canonical, bounded authentication of candidate-controlled workspace bytes.

This module deliberately knows nothing about Git.  A candidate's ``.git`` directory
is semantic, executable metadata and is excluded before traversal; repository root,
branch and commit identity must be supplied by a separately trusted policy layer.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DIGEST_DOMAIN = b"hermes-readonly-tree-v2"
DEFAULT_MAX_NODES = 100_000
DEFAULT_MAX_FILES = 100_000
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024
DEFAULT_DEADLINE_SECONDS = 30.0
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _Node:
    path: Path
    relative: str
    info: os.stat_result


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ValueError("workspace authentication deadline exceeded")


def _frame(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _inventory(
    root: Path,
    *,
    deadline: Optional[float],
    max_nodes: int,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    allow_symlinks: bool,
    require_single_link: bool,
    exclude_git_metadata: bool,
) -> list[_Node]:
    _check_deadline(deadline)
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("workspace root must be a directory")
    nodes = [_Node(root, ".", root_info)]
    pending = [(root, "")]
    node_count = file_count = total_bytes = 0
    while pending:
        directory, prefix = pending.pop()
        _check_deadline(deadline)
        with os.scandir(directory) as entries:
            for entry in entries:
                # ``scandir`` can itself be a slow or adversarial producer.  Check
                # the shared deadline after every yielded entry and charge the node
                # before retaining either its metadata or traversal path.  The
                # context manager closes the iterator on every rejection path.
                _check_deadline(deadline)
                if exclude_git_metadata and not prefix and entry.name == ".git":
                    continue
                node_count += 1
                if node_count > max_nodes:
                    raise ValueError("workspace exceeds node limit")
                path = directory / entry.name
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    if not allow_symlinks:
                        raise ValueError("workspace symlink is not supported")
                elif stat.S_ISDIR(info.st_mode):
                    pending.append((path, relative))
                elif stat.S_ISREG(info.st_mode):
                    if require_single_link and info.st_nlink != 1:
                        raise ValueError("workspace regular files must have a single link")
                    file_count += 1
                    if file_count > max_files:
                        raise ValueError("workspace exceeds file limit")
                    if info.st_size > max_file_bytes:
                        raise ValueError("workspace file byte limit exceeded")
                    total_bytes += info.st_size
                    if total_bytes > max_total_bytes:
                        raise ValueError("workspace aggregate byte limit exceeded")
                else:
                    raise ValueError("workspace contains unsupported filesystem node")
                nodes.append(_Node(path, relative, info))
    return sorted(nodes, key=lambda node: os.fsencode(node.relative))


def canonical_logical_workspace_digest(
    workspace: str | Path,
    *,
    deadline: Optional[float] = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    allow_symlinks: bool = False,
    require_single_link: bool = True,
    exclude_git_metadata: bool = True,
) -> str:
    """Return the stable logical-tree SHA-256 without consulting ``.git``.

    Limits are checked from ``lstat`` metadata before any file allocation. Files are
    streamed through no-follow descriptors and rechecked by descriptor and pathname;
    a second inventory brackets the whole operation to reject additions/removals and
    in-place mutation.
    """
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_DEADLINE_SECONDS
    root = Path(workspace).expanduser().resolve(strict=True)
    inventory = _inventory(
        root,
        deadline=deadline,
        max_nodes=max_nodes,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        allow_symlinks=allow_symlinks,
        require_single_link=require_single_link,
        exclude_git_metadata=exclude_git_metadata,
    )
    digest = hashlib.sha256()
    _frame(digest, DIGEST_DOMAIN)
    for node in inventory:
        _check_deadline(deadline)
        mode = node.info.st_mode
        relative = node.relative.encode("utf-8", errors="surrogateescape")
        mode_bytes = b"" if node.relative == "." else f"{stat.S_IMODE(mode):04o}".encode("ascii")
        if stat.S_ISDIR(mode):
            kind, payload = b"D", b""
        elif stat.S_ISLNK(mode):
            kind = b"L"
            payload = os.readlink(node.path).encode("utf-8", errors="surrogateescape")
            if not _same_file(node.info, node.path.lstat()):
                raise ValueError("workspace symlink changed during authentication")
        else:
            kind = b"F"
            file_digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            fd = os.open(node.path, flags)
            try:
                opened = os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode) or not _same_file(node.info, opened):
                    raise ValueError("workspace file changed during authentication")
                total = 0
                while True:
                    _check_deadline(deadline)
                    chunk = os.read(fd, min(_CHUNK_BYTES, max_file_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_file_bytes or total > node.info.st_size:
                        raise ValueError("workspace file byte limit exceeded")
                    file_digest.update(chunk)
                opened_after = os.fstat(fd)
                current = node.path.lstat()
                if total != node.info.st_size or not _same_file(node.info, opened_after) or not _same_file(node.info, current):
                    raise ValueError("workspace file changed during authentication")
            finally:
                os.close(fd)
            payload = file_digest.digest()
        for field in (relative, mode_bytes, kind, payload):
            _frame(digest, field)
    after = _inventory(
        root,
        deadline=deadline,
        max_nodes=max_nodes,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        allow_symlinks=allow_symlinks,
        require_single_link=require_single_link,
        exclude_git_metadata=exclude_git_metadata,
    )
    if len(after) != len(inventory) or any(
        left.relative != right.relative or not _same_file(left.info, right.info)
        for left, right in zip(inventory, after)
    ):
        raise ValueError("workspace changed during authentication")
    return digest.hexdigest()
