from __future__ import annotations

from pathlib import Path

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
    assert recovered.assignee == "default"
    assert recovered.status == "ready"
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
