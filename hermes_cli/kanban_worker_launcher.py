"""Pre-exec gate for durable Kanban resume launch intents.

This process performs no task work.  It validates the launch generation against the
shared board, records the child PID handshake, and only then execs the real Hermes
worker.  Failed/stale handshakes exit without loading the agent runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TRUSTED_ROOT = str(Path(__file__).resolve().parent.parent)


def _trusted_import_path() -> list[str]:
    """Exclude the candidate cwd while importing the installed Hermes tree."""
    cwd = os.path.realpath(os.getcwd())
    return [_TRUSTED_ROOT] + [
        item
        for item in sys.path
        if item and os.path.realpath(item) not in {cwd, _TRUSTED_ROOT}
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "--" or len(args) == 1:
        return 64
    original_path = list(sys.path)
    sys.path[:] = _trusted_import_path()
    try:
        from hermes_cli.kanban_resume_requests import handshake_resume_launch_intent

        allowed = handshake_resume_launch_intent(
            Path(os.environ["HERMES_KANBAN_DB"]),
            intent_id=os.environ["HERMES_KANBAN_LAUNCH_INTENT_ID"],
            generation=int(os.environ["HERMES_KANBAN_LAUNCH_GENERATION"]),
            task_id=os.environ["HERMES_KANBAN_TASK"],
            run_id=int(os.environ["HERMES_KANBAN_RUN_ID"]),
            claim_lock=os.environ["HERMES_KANBAN_CLAIM_LOCK"],
            worker_pid=os.getpid(),
        )
        if not allowed:
            return 75
        command = args[1:]
        # argv is constructed by the trusted dispatcher; no shell is involved and
        # candidate/request bytes never select the executable.
        exec_env = dict(os.environ)
        # The real CLI may use ``python -m`` as a fallback. Keep the candidate
        # cwd out of that interpreter's import path after the handshake too.
        exec_env["PYTHONSAFEPATH"] = "1"
        exec_env["PYTHONPATH"] = _TRUSTED_ROOT
        os.execvpe(command[0], command, exec_env)  # nosec B606
        return 70  # pragma: no cover - exec never returns
    except (KeyError, TypeError, ValueError, OSError):
        return 75
    finally:
        # Relevant only on failure and in unit tests: successful exec replaces
        # the process before this restoration can run.
        sys.path[:] = original_path


if __name__ == "__main__":
    raise SystemExit(main())
