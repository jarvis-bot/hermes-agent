from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from hermes_cli import workspace_digest


class _FakeEntry:
    def __init__(self, name: str):
        self.name = name

    def stat(self, *, follow_symlinks: bool = True):
        assert follow_symlinks is False
        return os.stat_result((stat.S_IFREG | 0o644, 1, 1, 1, 0, 0, 0, 0, 0, 0))


class _ScandirProducer:
    def __init__(self, count: int, *, delay: float = 0.0):
        self.count = count
        self.delay = delay
        self.consumed = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def __iter__(self):
        for index in range(self.count):
            if self.delay:
                time.sleep(self.delay)
            self.consumed += 1
            yield _FakeEntry(f"{index:06d}.txt")


def test_inventory_stops_at_first_node_over_limit_and_closes_scandir(
    monkeypatch, tmp_path
):
    producer = _ScandirProducer(100_000)
    monkeypatch.setattr(workspace_digest.os, "scandir", lambda _path: producer)

    with pytest.raises(ValueError, match="node limit"):
        workspace_digest.canonical_logical_workspace_digest(tmp_path, max_nodes=1)

    assert producer.consumed == 2
    assert producer.closed is True


def test_inventory_checks_deadline_while_consuming_slow_scandir(monkeypatch, tmp_path):
    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(workspace_digest.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", advance)
    producer = _ScandirProducer(100, delay=0.003)
    monkeypatch.setattr(workspace_digest.os, "scandir", lambda _path: producer)

    with pytest.raises(ValueError, match="deadline"):
        workspace_digest.canonical_logical_workspace_digest(tmp_path, deadline=0.010)

    assert producer.consumed == 4
    assert 0.010 <= clock[0] <= 0.015
    assert producer.closed is True


def test_inventory_rejects_very_wide_real_directory_at_limit(tmp_path):
    for index in range(128):
        (tmp_path / f"{index:04d}.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="node limit"):
        workspace_digest.canonical_logical_workspace_digest(tmp_path, max_nodes=16)


def test_inventory_node_limit_is_global_across_nested_directories(tmp_path):
    for directory_name in ("a", "b"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "payload.txt").write_text(directory_name, encoding="utf-8")

    with pytest.raises(ValueError, match="node limit"):
        workspace_digest.canonical_logical_workspace_digest(tmp_path, max_nodes=3)


def test_canonical_digest_is_stable_for_normal_fixture(tmp_path):
    (tmp_path / "z.txt").write_text("last\n", encoding="utf-8")
    nested = tmp_path / "a"
    nested.mkdir(mode=0o750)
    (nested / "b.txt").write_bytes(b"first\x00payload")
    os.chmod(tmp_path / "z.txt", 0o640)
    os.chmod(nested / "b.txt", 0o600)

    assert workspace_digest.canonical_logical_workspace_digest(tmp_path) == (
        "5f803464a5bd1993cf91e5cd0911ca33c57e15918be0f4c2b59c9e7046fc7d48"
    )
