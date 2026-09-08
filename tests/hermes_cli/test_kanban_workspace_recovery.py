from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_workspace_preflight import (
    WorkspaceCapability,
    preflight_workspace_for_profile,
)


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
            reviewer_isolated="reviewer" in profile,
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
        kb.block_task(conn, child, reason="mount denied", kind="workspace_capability")

        first = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="fallback-reviewer",
            capability_fn=_capability(workspace, {"fallback-reviewer"}),
        )
        second = kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="fallback-reviewer",
            capability_fn=_capability(workspace, {"fallback-reviewer"}),
        )
        recovered = kb.get_task(conn, child)
        recovery_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'workspace_recovered'",
            (child,),
        ).fetchone()[0]

    assert first == 1
    assert second == 0
    assert recovered is not None
    assert recovered.assignee == "fallback-reviewer"
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
        kb.block_task(conn, child, reason="mount denied", kind="workspace_capability")
        kb.recover_workspace_capability_tasks(
            conn,
            fallback_profile="fallback-reviewer",
            capability_fn=_capability(workspace, {"fallback-reviewer"}),
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
        kb.block_task(conn, task_id, reason="mount denied", kind="workspace_capability")

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


def test_generic_capability_block_is_never_workspace_recovered(kanban_home, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="publish image", assignee="default",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        kb.block_task(
            conn, task_id, reason="missing registry credential", kind="capability"
        )
        recovered = kb.recover_workspace_capability_tasks(
            conn, fallback_profile="default",
            capability_fn=_capability(workspace, {"default"}),
        )
        task = kb.get_task(conn, task_id)

    assert recovered == 0
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "capability"


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
            default_assignee="fallback-reviewer",
            workspace_capability_fn=_capability(workspace, {"fallback-reviewer"}),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == [(task_id, "fallback-reviewer", str(workspace.resolve()))]
    assert spawned == [("fallback-reviewer", str(workspace.resolve()))]
    assert task.assignee == "fallback-reviewer"
    assert task.requires_reviewer_isolation is True
    assert task.status == "running"


def test_reviewer_isolation_never_downgrades_to_ordinary_fallback(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="security review", assignee="security-reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_: (_ for _ in ()).throw(AssertionError("must not spawn")),
            default_assignee="default",
            workspace_capability_fn=_capability(workspace, {"default"}),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert task is not None
    assert task.assignee == "security-reviewer"
    assert task.status == "ready"
    assert task.current_run_id is None


@pytest.mark.parametrize("initial_status", ["ready", "review"])
def test_deleted_selected_assignee_uses_capable_fallback(
    kanban_home, tmp_path, monkeypatch, initial_status
):
    workspace = tmp_path / "deleted-assignee"
    workspace.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda profile: profile == "fallback-reviewer",
    )
    probed = []

    def capability(profile: str, candidate: Path) -> WorkspaceCapability:
        probed.append(profile)
        identity = candidate.stat()
        return WorkspaceCapability(
            profile == "fallback-reviewer", profile, str(candidate),
            reason="profile does not exist" if profile != "fallback-reviewer" else "",
            read_only=profile == "fallback-reviewer",
            device=identity.st_dev, inode=identity.st_ino,
            reviewer_isolated=True,
        )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="legacy review", assignee="deleted-reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        if initial_status == "review":
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
            conn.commit()
        spawned = []
        result = kb.dispatch_once(
            conn, default_assignee="fallback-reviewer",
            workspace_capability_fn=capability,
            spawn_fn=lambda task, path: spawned.append(
                (task.assignee, task.requires_reviewer_isolation, path)
            ) or 1234,
        )
        task = kb.get_task(conn, task_id)

    assert set(probed) == {"deleted-reviewer", "fallback-reviewer"}
    assert result.spawned == [(task_id, "fallback-reviewer", str(workspace.resolve()))]
    assert spawned == [("fallback-reviewer", True, str(workspace.resolve()))]
    assert task is not None and task.status == "running"
    assert task.assignee == "fallback-reviewer"
    assert task.requires_reviewer_isolation is True


def test_failed_production_reviewer_preflight_persists_before_rejecting_fallback(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "reviewer-failure"
    workspace.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)

    def capability(profile: str, candidate: Path) -> WorkspaceCapability:
        if profile == "security-reviewer":
            return preflight_workspace_for_profile(
                profile,
                candidate,
                profile_config={"terminal": {"backend": "local"}},
                runtime_probe=lambda **_: None,
            )
        identity = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), read_only=False,
            device=identity.st_dev, inode=identity.st_ino,
        )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="legacy reviewer task", assignee="security-reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        result = kb.dispatch_once(
            conn, default_assignee="default", workspace_capability_fn=capability,
            spawn_fn=lambda *_: pytest.fail("ordinary fallback must not spawn"),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert task is not None and task.status == "ready"
    assert task.assignee == "security-reviewer"
    assert task.requires_reviewer_isolation is True


def test_scratch_workspace_is_materialized_then_preflighted_before_claim(
    kanban_home, monkeypatch
):
    calls = []
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="scratch review", assignee="quality-reviewer"
        )

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            calls.append((profile, candidate, kb.get_task(conn, task_id).status))
            identity = candidate.stat()
            return WorkspaceCapability(
                True, profile, str(candidate), read_only=True,
                device=identity.st_dev, inode=identity.st_ino,
                reviewer_isolated=True,
            )

        result = kb.dispatch_once(
            conn, spawn_fn=lambda _task, _path: 1234,
            workspace_capability_fn=capability,
        )
        task = kb.get_task(conn, task_id)

    assert len(calls) == 1
    assert calls[0][1].is_dir()
    assert calls[0][2] == "ready"
    assert result.spawned == [(task_id, "quality-reviewer", str(calls[0][1]))]
    assert task is not None and task.status == "running"


def test_review_column_materializes_and_preflights_before_claim(
    kanban_home, monkeypatch
):
    calls = []
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="review lane", assignee="quality-reviewer"
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            calls.append((profile, candidate, kb.get_task(conn, task_id).status))
            identity = candidate.stat()
            return WorkspaceCapability(
                True, profile, str(candidate), read_only=True,
                device=identity.st_dev, inode=identity.st_ino,
                reviewer_isolated=True,
            )

        result = kb.dispatch_once(
            conn, spawn_fn=lambda _task, _path: 1234,
            workspace_capability_fn=capability,
        )
        task = kb.get_task(conn, task_id)

    assert len(calls) == 1
    assert calls[0][1].is_dir()
    assert calls[0][2] == "review"
    assert result.spawned == [(task_id, "quality-reviewer", str(calls[0][1]))]
    assert task is not None and task.status == "running"


@pytest.mark.parametrize("assignee", [None, "unavailable-reviewer"])
def test_review_column_uses_capable_default_fallback_with_cas(
    kanban_home, tmp_path, monkeypatch, assignee
):
    workspace = tmp_path / "review-fallback"
    workspace.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    spawned = []
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="review fallback", assignee=assignee,
            workspace_kind="dir", workspace_path=str(workspace),
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()

        result = kb.dispatch_once(
            conn,
            default_assignee="fallback-reviewer",
            workspace_capability_fn=_capability(workspace, {"fallback-reviewer"}),
            spawn_fn=lambda task, path: spawned.append(
                (task.assignee, task.requires_reviewer_isolation, path)
            ) or 1234,
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == [(task_id, "fallback-reviewer", str(workspace.resolve()))]
    assert spawned == [("fallback-reviewer", True, str(workspace.resolve()))]
    assert task is not None
    assert task.assignee == "fallback-reviewer"
    assert task.requires_reviewer_isolation is True
    assert task.status == "running"


def test_review_column_ordinary_assignee_cannot_spawn_unrestricted(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "ordinary-in-review"
    workspace.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    observed = []

    def capability(profile: str, candidate: Path) -> WorkspaceCapability:
        identity = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), read_only=profile == "fallback-reviewer",
            device=identity.st_dev, inode=identity.st_ino,
            reviewer_isolated=profile == "fallback-reviewer",
        )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="candidate content", assignee="coder",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()
        result = kb.dispatch_once(
            conn,
            default_assignee="fallback-reviewer",
            workspace_capability_fn=capability,
            spawn_fn=lambda task, _path: observed.append(
                (task.assignee, task.requires_reviewer_isolation)
            ) or 1234,
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned
    assert observed == [("fallback-reviewer", True)]
    assert task is not None and task.requires_reviewer_isolation is True


def test_linked_worktree_review_dispatches_from_standalone_runtime_snapshot(
    kanban_home, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    linked = repo / ".worktrees" / "linked"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "candidate.txt").write_text("exact candidate\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    (repo / "candidate.txt").write_text("exact candidate v2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "candidate"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "review/test", str(linked)],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    observed = []

    def runtime_capability(profile: str, candidate: Path) -> WorkspaceCapability:
        from tools.environments.docker import _path_identity
        identity = _path_identity(str(candidate), content_digest=True)
        assert (candidate / ".git").is_dir()
        assert identity["git_ref"] == commit
        from tools.environments.docker import _verify_git_workspace_provenance
        _verify_git_workspace_provenance(candidate, commit)
        stat_result = candidate.stat()
        return WorkspaceCapability(
            True, profile, str(candidate), read_only=True,
            device=stat_result.st_dev, inode=stat_result.st_ino,
            content_sha256=str(identity["mounted_content_sha256"]),
            reviewer_isolated=True,
        )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="linked review", assignee="quality-reviewer",
            workspace_kind="worktree", workspace_path=str(linked),
            branch_name="review/test",
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()
        result = kb.dispatch_once(
            conn, workspace_capability_fn=runtime_capability,
            spawn_fn=lambda task, path, **_: observed.append((task, path)) or 1234,
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned
    assert task is not None
    assert task.workspace_kind == "dir"
    assert task.expected_workspace_sha == commit
    assert task.requires_reviewer_isolation is True
    assert Path(task.workspace_path or "") != linked
    assert (Path(task.workspace_path or "") / ".git").is_dir()
    assert observed[0][1] == task.workspace_path


def test_linked_review_snapshot_rejects_candidate_gitdir_redirect_and_sha_mismatch(
    kanban_home, tmp_path
):
    trusted = tmp_path / "trusted"
    private = tmp_path / "private"
    for repo, payload in ((trusted, "public\n"), (private, "private data\n")):
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / "payload.txt").write_text(payload, encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    expected = subprocess.run(
        ["git", "-C", str(trusted), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    candidate = trusted / ".worktrees" / "task-malicious"
    candidate.mkdir(parents=True)
    (candidate / ".git").write_text(
        f"gitdir: {private / '.git'}\n", encoding="utf-8"
    )
    task = SimpleNamespace(
        id="task-malicious", expected_workspace_sha=expected,
        project_id=None, workspace_path=str(candidate),
    )

    with pytest.raises(RuntimeError, match="trusted repository|Git metadata"):
        kb._materialize_immutable_review_snapshot(task, candidate)

    legitimate = trusted / ".worktrees" / "task-legitimate"
    subprocess.run(
        ["git", "-C", str(trusted), "worktree", "add", "-qb", "review-test", str(legitimate)],
        check=True,
    )
    wrong_sha = subprocess.run(
        ["git", "-C", str(private), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    mismatch_task = SimpleNamespace(
        id="task-legitimate", expected_workspace_sha=wrong_sha,
        project_id=None, workspace_path=str(legitimate),
    )
    with pytest.raises(RuntimeError, match="HEAD does not match assigned SHA"):
        kb._materialize_immutable_review_snapshot(mismatch_task, legitimate)
    assert mismatch_task.expected_workspace_sha == wrong_sha

    snapshots = kb.workspaces_root() / ".review-snapshots"
    assert not snapshots.exists() or not any(snapshots.iterdir())


def test_linked_review_snapshot_rebuilds_tampered_existing_checkout(
    kanban_home, tmp_path
):
    repo = tmp_path / "trusted"
    linked = repo / ".worktrees" / "linked"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("trusted\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "trusted"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "review/tamper", str(linked)],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    task = SimpleNamespace(
        id="tamper", expected_workspace_sha=commit, project_id=None,
        workspace_path=str(linked),
    )

    snapshot, _ = kb._materialize_immutable_review_snapshot(task, linked)
    (snapshot / "tracked.txt").write_text("tampered\n", encoding="utf-8")
    rebuilt, _ = kb._materialize_immutable_review_snapshot(task, linked)

    assert rebuilt == snapshot
    assert (rebuilt / "tracked.txt").read_text(encoding="utf-8") == "trusted\n"
    assert subprocess.run(
        ["git", "-C", str(rebuilt), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_linked_review_snapshot_validates_concurrent_file_exists_winner(
    kanban_home, tmp_path, monkeypatch
):
    repo = tmp_path / "trusted"
    linked = repo / ".worktrees" / "linked"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("trusted\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "trusted"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "review/winner", str(linked)],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    task = SimpleNamespace(
        id="winner", expected_workspace_sha=commit, project_id=None,
        workspace_path=str(linked),
    )
    target, _ = kb._materialize_immutable_review_snapshot(task, linked)
    winner = target.with_name("saved-winner")
    target.rename(winner)
    winner_inode = winner.stat().st_ino
    original_rename = Path.rename
    injected = False

    def install_winner_then_lose(path, destination):
        nonlocal injected
        if destination == target and path != winner and not injected:
            injected = True
            original_rename(winner, target)
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", install_winner_then_lose)
    owned = []
    resolved, _ = kb._materialize_immutable_review_snapshot(
        task, linked, created_artifacts=owned
    )

    assert injected is True
    assert resolved.stat().st_ino == winner_inode
    assert owned == []


def test_dispatch_cas_loss_rolls_back_only_new_worktree(
    kanban_home, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    original = kb._resolve_worktree_workspace

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="race", assignee="engineer", workspace_kind="worktree",
            workspace_path=str(repo),
        )

        def materialize_then_move(task, **kwargs):
            result = original(task, **kwargs)
            with kb.connect() as rival:
                rival.execute(
                    "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                    (str(tmp_path / "rival"), task.id),
                )
                rival.commit()
            return result

        monkeypatch.setattr(kb, "_resolve_worktree_workspace", materialize_then_move)
        snapshots = kb._materialize_dispatch_workspace_candidates(
            conn, board=None, default_assignee=None, max_spawn=None,
            max_in_progress=None, max_in_progress_per_profile=None,
        )

    target = repo / ".worktrees" / task_id
    assert snapshots == {}
    assert not target.exists()
    assert not kb._git_branch_exists(repo, f"wt/{task_id}")


def test_review_snapshot_cas_loss_removes_owned_snapshot_but_preserves_linked_worktree(
    kanban_home, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    linked = repo / ".worktrees" / "linked"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "review/race", str(linked)],
        check=True,
    )
    original = kb._materialize_immutable_review_snapshot

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="review race", assignee="quality-reviewer",
            workspace_kind="worktree", workspace_path=str(linked),
            branch_name="review/race",
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()

        def snapshot_then_move(task, workspace, **kwargs):
            result = original(task, workspace, **kwargs)
            with kb.connect() as rival:
                rival.execute(
                    "UPDATE tasks SET expected_workspace_sha = ? WHERE id = ?",
                    ("f" * 40, task.id),
                )
                rival.commit()
            return result

        monkeypatch.setattr(kb, "_materialize_immutable_review_snapshot", snapshot_then_move)
        snapshots = kb._materialize_dispatch_workspace_candidates(
            conn, board=None, default_assignee=None, max_spawn=None,
            max_in_progress=None, max_in_progress_per_profile=None,
        )
        raced_task = kb.get_task(conn, task_id)

    snapshot_root = kb.workspaces_root() / ".review-snapshots"
    assert snapshots == {}
    assert raced_task is not None and raced_task.expected_workspace_sha == "f" * 40
    assert linked.is_dir()
    assert not snapshot_root.exists() or not any(snapshot_root.iterdir())


def test_slow_workspace_probe_runs_outside_dispatch_lock(
    kanban_home, tmp_path, monkeypatch
):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    lock_depth = 0
    from contextlib import contextmanager

    @contextmanager
    def tracked_lock(_path):
        nonlocal lock_depth
        lock_depth += 1
        try:
            yield True
        finally:
            lock_depth -= 1

    monkeypatch.setattr(kb, "_dispatch_tick_lock", tracked_lock)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _: True)
    with kb.connect() as conn:
        kb.create_task(
            conn, title="review", assignee="quality-reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )

        def capability(profile: str, candidate: Path) -> WorkspaceCapability:
            assert lock_depth == 0
            identity = candidate.stat()
            return WorkspaceCapability(
                True, profile, str(candidate), read_only=True,
                device=identity.st_dev, inode=identity.st_ino,
                reviewer_isolated=True,
            )

        result = kb.dispatch_once(
            conn, spawn_fn=lambda *_: 1234, workspace_capability_fn=capability
        )

    assert len(result.spawned) == 1


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
        kb.block_task(conn, task_id, reason="mount denied", kind="workspace_capability")

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
