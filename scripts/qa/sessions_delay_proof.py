"""Prove the sessions tests survive a slow filter or rename render.

Usage: python scripts/qa/sessions_delay_proof.py [BASE_REF]

Run it with the Python of an environment that has the test dependencies; the
script puts src/ on PYTHONPATH.

The product renders the session list from Input.Changed (the filter) and from
refresh_list (after a rename). Both run in the same message pass as
the key press, so a plain pilot.pause() always sees fresh rows. On a slow
runner the render can land later. This script models that by deferring the
filter apply from Input.Changed and the refresh that ends a rename with a timer, so the render
runs DELAY seconds after the key press returns. It then runs the sessions tests
from BASE_REF (default: the parent of the flake fix, c84fe2389) and the tests in
the working tree, each with the delay on. Old tests must fail, new tests pass.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELAY = 0.15
OLD_NAME = "test_tui_sessions_old.py"

PLUGIN = f"""
import sys

import pytest

DELAY = {DELAY}


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    from lilbee.cli.tui.widgets.session_list import SessionListPanel

    orig_apply = SessionListPanel._apply_filter
    orig_refresh = SessionListPanel.refresh_list

    def apply(self):
        # Only the Input.Changed handler is slow. The catch-up path that the
        # list keys use (_filter_caught_up) still applies the filter at once.
        if sys._getframe(1).f_code.co_name == "_on_filter":
            self.set_timer(DELAY, lambda: orig_apply(self))
        else:
            orig_apply(self)

    def refresh(self):
        # Only the refresh that ends a rename is slow; mount and delete are not.
        if sys._getframe(1).f_code.co_name == "_finish_rename":
            self.set_timer(DELAY, lambda: orig_refresh(self))
        else:
            orig_refresh(self)

    SessionListPanel._apply_filter = apply
    SessionListPanel.refresh_list = refresh
"""


def run(plugin_dir: Path, test_file: str) -> tuple[int, str]:
    env_path = f"{plugin_dir}:{ROOT / 'src'}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-p",
        "delay_plugin",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "-rf",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": env_path},
    )
    lines = proc.stdout.strip().splitlines() or [proc.stderr[-300:]]
    failed = [ln.split(" - ")[0].split("::")[-1] for ln in lines if ln.startswith("FAILED")]
    return proc.returncode, lines[-1] + "".join(f"\n    {name}" for name in failed)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "c84fe2389"
    old_src = subprocess.run(
        ["git", "show", f"{base}:tests/test_tui_sessions.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    old_path = ROOT / "tests" / OLD_NAME
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "delay_plugin.py").write_text(PLUGIN)
        old_path.write_text(old_src)
        try:
            old_rc, old_tail = run(Path(tmp), f"tests/{OLD_NAME}")
            new_rc, new_tail = run(Path(tmp), "tests/test_tui_sessions.py")
        finally:
            old_path.unlink()
    print(f"old tests ({base}) with {DELAY}s delay: rc={old_rc} {old_tail}")
    print(f"new tests (working tree) with {DELAY}s delay: rc={new_rc} {new_tail}")
    return 0 if old_rc != 0 and new_rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
