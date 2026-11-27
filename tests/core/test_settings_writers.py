"""Processes that write config.toml at the same time lose no key."""

import subprocess
import sys
import tomllib
from pathlib import Path

WRITERS = 8
WRITE = (
    "from pathlib import Path; from lilbee.core import settings; "
    "settings.set_value(Path({root!r}), {key!r}, {value})"
)


def test_eight_processes_writing_keys_lose_none(tmp_path: Path) -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", WRITE.format(root=str(tmp_path), key=f"x{i}", value=i)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        for i in range(WRITERS)
    ]
    assert [process.wait(timeout=120) for process in processes] == [0] * WRITERS
    written = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert written == {f"x{i}": i for i in range(WRITERS)}
