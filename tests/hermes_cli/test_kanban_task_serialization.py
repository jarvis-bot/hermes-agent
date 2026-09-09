from types import SimpleNamespace
from typing import Any, cast

from hermes_cli.kanban import _task_to_dict


def test_task_json_exposes_max_runtime_for_dispatch_authentication() -> None:
    task = SimpleNamespace(
        id="t_runtime",
        title="runtime",
        body="body",
        assignee="reviewer",
        status="ready",
        priority=0,
        tenant=None,
        workspace_kind="dir",
        workspace_path="/workspace",
        branch_name=None,
        expected_workspace_sha="a" * 40,
        project_id=None,
        created_by="user",
        created_at=1,
        started_at=None,
        completed_at=None,
        result=None,
        skills=(),
        max_retries=3,
        max_runtime_seconds=600,
        model_override=None,
        provider_override=None,
        session_id=None,
        workflow_template_id=None,
        current_step_key=None,
    )

    assert _task_to_dict(cast(Any, task))["max_runtime_seconds"] == 600
