"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist and can mount the test workspace."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]

    def route_capable(children, workspace, *, fallback_profile):
        identity = Path(workspace).stat()
        return [
            dict(
                child,
                _workspace_device=identity.st_dev,
                _workspace_inode=identity.st_ino,
                requires_reviewer_isolation="reviewer" in str(child.get("assignee", "")),
            )
            for child in children
        ], []

    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
        patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=route_capable,
        ),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


def test_decompose_preflights_exact_workspace_and_reroutes_before_create(
    kanban_home, tmp_path
):
    workspace = tmp_path / "dynamic" / "ticket" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="review candidate",
            triage=True,
            assignee="orchestrator",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "independent reviews",
        "tasks": [
            {"title": "functional", "body": "review", "assignee": "functional-reviewer", "parents": []},
            {"title": "security", "body": "review", "assignee": "security-reviewer", "parents": []},
        ],
    })
    checked = []

    def route(children, selected_workspace, *, fallback_profile):
        from hermes_cli.kanban_workspace_preflight import WorkspaceCapability

        checked.append((Path(selected_workspace), fallback_profile))
        failures = [
            WorkspaceCapability(False, child["assignee"], str(selected_workspace), "denied")
            for child in children
        ]
        attestation = Path(selected_workspace).stat()
        routed = [
            dict(
                child,
                assignee="orchestrator",
                requires_reviewer_isolation=True,
                _workspace_device=attestation.st_dev,
                _workspace_inode=attestation.st_ino,
            )
            for child in children
        ]
        return routed, failures

    patches = _patch_list_profiles(
        ["orchestrator", "functional-reviewer", "security-reviewer"]
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=route,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is True
    assert checked == [(workspace.resolve(), "orchestrator")]
    with kb.connect() as conn:
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]
    assert [child.assignee for child in children] == ["orchestrator", "orchestrator"]


def test_decompose_materializes_scratch_workspace_before_graph_publication(
    kanban_home
):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="dynamic review", triage=True, assignee="orchestrator"
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "review",
        "tasks": [
            {"title": "quality", "body": "review", "assignee": "quality-reviewer", "parents": []},
        ],
    })
    checked = []

    def route(children, selected_workspace, *, fallback_profile):
        workspace = Path(selected_workspace)
        assert workspace.is_dir()
        checked.append(workspace)
        identity = workspace.stat()
        return [dict(
            children[0], _workspace_device=identity.st_dev,
            _workspace_inode=identity.st_ino,
            requires_reviewer_isolation=True,
        )], []

    patches = _patch_list_profiles(["orchestrator", "quality-reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=route,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is True
    assert len(checked) == 1
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        child = kb.get_task(conn, outcome.child_ids[0])
    assert root is not None and root.workspace_path == str(checked[0])
    assert child is not None and child.workspace_path == str(checked[0])


def test_worktree_decomposition_materializes_and_attests_each_child_before_publish(
    kanban_home, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="fan out", triage=True, assignee="orchestrator",
            workspace_kind="worktree", workspace_path=str(repo),
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "tasks": [
            {"title": "one", "body": "work", "assignee": "engineer", "parents": []},
            {"title": "two", "body": "review", "assignee": "quality-reviewer", "parents": [0]},
        ],
    })
    checked = []

    def route(children, selected_workspace, *, fallback_profile):
        workspace = Path(selected_workspace).resolve(strict=True)
        assert (workspace / ".git").is_file()
        checked.append(workspace)
        identity = workspace.stat()
        return [dict(
            children[0], _workspace_device=identity.st_dev,
            _workspace_inode=identity.st_ino,
            requires_reviewer_isolation="reviewer" in children[0]["assignee"],
        )], []

    patches = _patch_list_profiles(["orchestrator", "engineer", "quality-reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=route,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is True
    assert len(checked) == 2 and len(set(checked)) == 2
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, child_id) for child_id in outcome.child_ids]
    assert root is not None
    assert all(child is not None and child.workspace_path for child in children)
    assert {Path(child.workspace_path) for child in children} == set(checked)
    assert all(Path(child.workspace_path) != Path(root.workspace_path) for child in children)


def test_worktree_decomposition_rolls_back_new_children_after_repeated_second_failure(
    kanban_home, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    subprocess.run(["git", "-C", str(repo), "branch", "preexisting"], check=True)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="fan out", triage=True, assignee="orchestrator",
            workspace_kind="worktree", workspace_path=str(repo),
        )
    payload = jsonlib.dumps({
        "fanout": True,
        "tasks": [
            {"title": "one", "body": "work", "assignee": "engineer", "parents": []},
            {"title": "two", "body": "work", "assignee": "engineer", "parents": []},
        ],
    })
    calls = 0
    fail_second = True

    def route(children, selected_workspace, *, fallback_profile):
        nonlocal calls
        calls += 1
        if fail_second and calls == 2:
            raise RuntimeError("second child rejected")
        identity = Path(selected_workspace).stat()
        return [dict(children[0], _workspace_device=identity.st_dev,
                     _workspace_inode=identity.st_ino)], []

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for item in patches:
        item.start()
    try:
        for _ in range(2):
            calls = 0
            with _patch_aux_client(payload), _patch_extra_body(), patch(
                "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
                side_effect=route,
            ):
                outcome = decomp.decompose_task(tid, author="me")
            assert outcome.ok is False
            assert "second child rejected" in outcome.reason
            listed = subprocess.run(
                ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout
            branches = subprocess.run(
                ["git", "-C", str(repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            assert "preexisting" in branches
            assert listed.count("worktree ") == 2
            assert [branch for branch in branches if branch.startswith("wt/")] == [f"wt/{tid}"]

        fail_second = False
        for publication_result in (ValueError("rejected graph"), None):
            calls = 0
            publication_patch = (
                patch.object(kb, "decompose_triage_task", side_effect=publication_result)
                if isinstance(publication_result, Exception)
                else patch.object(kb, "decompose_triage_task", return_value=None)
            )
            with _patch_aux_client(payload), _patch_extra_body(), patch(
                "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
                side_effect=route,
            ), publication_patch:
                outcome = decomp.decompose_task(tid, author="me")
            assert outcome.ok is False
            listed = subprocess.run(
                ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout
            assert listed.count("worktree ") == 2

        existing_target = repo / ".worktrees" / "preexisting-child"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-qb", "wt/preexisting-child", str(existing_target)],
            check=True,
        )
        artifacts = []
        kb._ensure_git_worktree(
            repo, existing_target, "wt/preexisting-child",
            created_artifacts=artifacts,
        )
        assert artifacts == []
        kb._cleanup_created_worktree_artifacts(artifacts)
        kb._cleanup_created_worktree_artifacts(artifacts)
        assert existing_target.is_dir()
        assert kb._git_branch_exists(repo, "wt/preexisting-child")

        subprocess.run(
            ["git", "-C", str(repo), "branch", "wt/reused-branch"], check=True
        )
        reused_target = repo / ".worktrees" / "reused-branch"
        artifacts = []
        kb._ensure_git_worktree(
            repo, reused_target, "wt/reused-branch", created_artifacts=artifacts
        )
        assert len(artifacts) == 1 and artifacts[0].created_branch is False
        kb._cleanup_created_worktree_artifacts(artifacts)
        assert not reused_target.exists()
        assert kb._git_branch_exists(repo, "wt/reused-branch")
    finally:
        for item in patches:
            item.stop()


def test_decompose_aborts_atomically_when_selected_and_fallback_cannot_mount(
    kanban_home, tmp_path
):
    workspace = tmp_path / "dynamic" / "repo"
    workspace.mkdir(parents=True)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="review candidate",
            triage=True,
            assignee="orchestrator",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "review",
        "tasks": [
            {"title": "security", "body": "review", "assignee": "security-reviewer", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "security-reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=RuntimeError("fallback unavailable"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "workspace preflight failed" in outcome.reason
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        created = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'created' AND task_id != ?",
            (tid,),
        ).fetchone()[0]
    assert root.status == "triage"
    assert root.assignee == "orchestrator"
    assert created == 0


def test_decompose_aborts_if_workspace_changes_after_preflight(
    kanban_home, tmp_path
):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="review candidate",
            triage=True,
            assignee="orchestrator",
            workspace_kind="dir",
            workspace_path=str(original),
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "review",
        "tasks": [
            {"title": "security", "body": "review", "assignee": "security-reviewer", "parents": []},
        ],
    })

    def route(children, selected_workspace, *, fallback_profile):
        assert Path(selected_workspace) == original
        with kb.connect() as conn:
            kb.set_workspace_path(conn, tid, replacement)
        return children, []

    patches = _patch_list_profiles(["orchestrator", "security-reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose.route_children_to_capable_profiles",
            side_effect=route,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        children = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id != ?", (tid,)
        ).fetchone()[0]
    assert root is not None
    assert root.status == "triage"
    assert root.workspace_path == str(replacement)
    assert children == 0


def test_decompose_aborts_if_workspace_object_is_replaced_after_preflight(
    kanban_home, tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="review candidate",
            triage=True,
            assignee="orchestrator",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        before = workspace.stat()
        workspace.rename(tmp_path / "attested-original")
        workspace.mkdir()
        with pytest.raises(ValueError, match="filesystem object changed"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orchestrator",
                children=[{
                    "title": "security review",
                    "assignee": "security-reviewer",
                    "parents": [],
                    "requires_reviewer_isolation": True,
                    "_workspace_device": before.st_dev,
                    "_workspace_inode": before.st_ino,
                }],
                expected_workspace_path=str(workspace),
            )
        root = kb.get_task(conn, tid)
        child_count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id != ?", (tid,)
        ).fetchone()[0]

    assert root is not None
    assert root.status == "triage"
    assert child_count == 0


