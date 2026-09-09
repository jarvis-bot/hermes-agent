"""Reviewed no-agent cron entrypoint; copy to $HERMES_HOME/scripts before activation."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from hermes_cli.kanban_resume_observer import main


if __name__ == "__main__":
    script = Path(__file__).resolve(strict=True)
    info = script.stat(follow_symlinks=False)
    if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise SystemExit(
            "observer entrypoint must be root-owned and not group/world writable"
        )
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser().resolve()
    raise SystemExit(
        main(["--manifest", str(home / "kanban" / "resume-observer.json")])
    )
