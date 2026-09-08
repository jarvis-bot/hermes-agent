from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_workspace_preflight import WorkspaceCapability


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _capability(workspace: Path, available: set[str]):
    def check(profile: str, candidate: Path) -> WorkspaceCapability:
        assert candidate.resolve() == workspace.resolve()
        return WorkspaceCapability(
            profile in available,
            profile,
            str(candidate),
            "mount denied" if profile not in available else "",
            read_only="reviewer" in profile,
            device=candidate.stat().st_dev,
            inode=candidate.stat().st_ino,
        )

    return check


def test_restart_recovers_capability_block_to_fallback_idempotently(
    kanban_home, tmp_path
):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        root = kb.create_task(
            conn,
            title="root",
            assignee="default",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        child = kb.create_task(
            conn,
            title="review",
            assignee="security-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        kb.link_tasks(conn, parent_id=child, child_id=root)
        kb.block_task(conn, child, reason="mount denied", kind="capability")

        first = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="default",
            capability_fn=_capability(workspace, {"default"}),
        )
        second = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="default",
            capability_fn=_capability(workspace, {"default"}),
        )
        recovered = kb.get_task(conn, child)
        recovery_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'workspace_recovered'",
            (child,),
        ).fetchone()[0]

    assert first == 1
    assert second == 0
    assert recovered is not None
    assert recovered.assignee == "default"
    assert recovered.status == "ready"
    assert recovered.requires_reviewer_isolation is True
    assert recovered.block_kind is None
    assert recovered.block_recurrences == 0
    assert recovery_events == 1


def test_recovered_child_completion_reconciles_parent(kanban_home, tmp_path):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        root = kb.create_task(
            conn,
            title="root",
            assignee="default",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        child = kb.create_task(
            conn,
            title="review",
            assignee="functional-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        kb.link_tasks(conn, parent_id=child, child_id=root)
        kb.block_task(conn, child, reason="mount denied", kind="capability")
        kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="default",
            capability_fn=_capability(workspace, {"default"}),
        )
        claimed = kb.claim_task(conn, child)
        assert claimed is not None
        assert kb.complete_task(conn, child, summary="continued safely") is True
        parent = kb.get_task(conn, root)

    assert parent.status == "ready"


def test_recovery_to_reviewer_fallback_persists_isolation(kanban_home, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="implementer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        kb.block_task(conn, task_id, reason="mount denied", kind="capability")

        recovered = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="quality-reviewer",
            capability_fn=_capability(workspace, {"quality-reviewer"}),
        )
        task = kb.get_task(conn, task_id)

    assert recovered == 1
    assert task is not None
    assert task.assignee == "quality-reviewer"
    assert task.requires_reviewer_isolation is True


def test_dispatch_preflights_before_claim_and_reroutes_to_active_owner(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="quality-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
        spawned = []

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, path: spawned.append((task.assignee, path)) or 1234,
            default_assignee="default",
            workspace_capability_fn=_capability(workspace, {"default"}),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == [(task_id, "default", str(workspace.resolve()))]
    assert spawned == [("default", str(workspace.resolve()))]
    assert task.assignee == "default"
    assert task.requires_reviewer_isolation is True
    assert task.status == "running"


def test_all_unavailable_leaves_existing_owner_unclaimed_for_later_recovery(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="quality-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: (_ for _ in ()).throw(AssertionError("must not spawn")),
            default_assignee="default",
            workspace_capability_fn=_capability(workspace, set()),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert task.status == "ready"
    assert task.assignee == "quality-reviewer"
    assert task.current_run_id is None


def test_networked_writable_worker_can_dispatch_when_it_is_its_own_fallback(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    config = {
        "terminal": {
            "backend": "docker",
            "docker_image": "python:3.11-slim",
            "docker_mount_cwd_to_workspace": True,
            "docker_cwd_mount_mode": "rw",
            "docker_cwd_allowed_roots": [str(tmp_path)],
            "docker_cwd_path_mappings": {},
            "docker_network": True,
        }
    }
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    monkeypatch.setattr(
        "hermes_cli.kanban_workspace_preflight._profile_config", lambda _: config
    )
    monkeypatch.setattr(
        "tools.environments.docker.find_docker", lambda: "/usr/bin/docker"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_workspace_preflight.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=b""),
    )
    spawned = []

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="implementation",
            assignee="default",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, path: spawned.append((task.assignee, path)) or 1234,
            default_assignee="default",
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == [(task_id, "default", str(workspace.resolve()))]
    assert spawned == [("default", str(workspace.resolve()))]
    assert task is not None
    assert task.status == "running"


def test_dispatch_rejects_replaced_workspace_object_after_probe(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="quality-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            before = candidate.stat()
            candidate.rename(tmp_path / "attested-original")
            candidate.mkdir()
            return WorkspaceCapability(
                True,
                profile,
                str(candidate),
                read_only=True,
                device=before.st_dev,
                inode=before.st_ino,
            )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: (_ for _ in ()).throw(
                AssertionError("must not spawn replaced workspace")
            ),
            default_assignee="default",
            workspace_capability_fn=capability,
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None


def test_dispatch_requeues_if_workspace_is_replaced_after_claim(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    original = tmp_path / "attested-original"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="quality-reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)

        def replace_after_claim(task, *, board=None):
            workspace.rename(original)
            workspace.mkdir()
            return str(workspace)

        monkeypatch.setattr(kb, "resolve_workspace", replace_after_claim)
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: (_ for _ in ()).throw(
                AssertionError("must not spawn replaced workspace")
            ),
            default_assignee="default",
            workspace_capability_fn=_capability(
                workspace, {"quality-reviewer", "default"}
            ),
        )
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()

    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None
    assert run["status"] == "workspace_changed"
    assert run["outcome"] == "workspace_changed"


def test_recovery_does_not_apply_probe_after_workspace_changes(
    kanban_home, tmp_path
):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="security-reviewer",
            workspace_kind="dir",
            workspace_path=str(original),
        )
        kb.block_task(conn, task_id, reason="mount denied", kind="capability")

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            assert profile == "security-reviewer"
            assert candidate == original
            kb.set_workspace_path(conn, task_id, replacement)
            return WorkspaceCapability(True, profile, str(candidate), read_only=True)

        recovered = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="default",
            capability_fn=capability,
        )
        task = kb.get_task(conn, task_id)

    assert recovered == 0
    assert task is not None
    assert task.status == "blocked"
    assert task.workspace_path == str(replacement)


def test_dispatch_does_not_claim_after_workspace_changes_during_probe(
    kanban_home, tmp_path, monkeypatch
):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review",
            assignee="quality-reviewer",
            workspace_kind="dir",
            workspace_path=str(original),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            assert profile == "quality-reviewer"
            assert candidate == original
            kb.set_workspace_path(conn, task_id, replacement)
            return WorkspaceCapability(True, profile, str(candidate), read_only=True)

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: (_ for _ in ()).throw(AssertionError("must not spawn")),
            default_assignee="default",
            workspace_capability_fn=capability,
        )
        task = kb.get_task(conn, task_id)
        claimed_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'claimed'",
            (task_id,),
        ).fetchone()[0]
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.current_run_id is None
    assert task.workspace_path == str(replacement)
    assert claimed_events == 0
    assert run_count == 0
