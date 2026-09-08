"""Reviewed no-agent cron entrypoint; copy to $HERMES_HOME/scripts before activation."""

from __future__ import annotations

import os
from pathlib import Path

from hermes_cli.kanban_resume_observer import main


if __name__ == "__main__":
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser().resolve()
    raise SystemExit(
        main(["--manifest", str(home / "kanban" / "resume-observer.json")])
    )
