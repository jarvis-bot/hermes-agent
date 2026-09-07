"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
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
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
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
        return ([dict(child, assignee="orchestrator") for child in children], failures)

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


