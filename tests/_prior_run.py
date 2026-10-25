"""Leave the state an earlier run of lilbee left, written by a real earlier process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from lilbee.runtime.engine_lock import request_keep_warm\n"
    "request_keep_warm(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))\n"
)


def request_keep_warm_in_prior_run(engine_dir: Path, config_root: Path, idle_minutes: int) -> None:
    """Record a keep-warm opt-in from a process that has already exited."""
    subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(engine_dir), str(config_root), str(idle_minutes)],
        check=True,
        timeout=60,
    )
