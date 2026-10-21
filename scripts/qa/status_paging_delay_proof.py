"""Prove the cursor paging test survives a third page that lands with the second.

Usage: python scripts/qa/status_paging_delay_proof.py [BASE_REF]

Run it with the Python of an environment that has the test dependencies; the
script puts src/ on PYTHONPATH.

A scroll event can reach the status screen with the table geometry from before
the second page was added. It then starts a third page load at once. On a slow
runner the third page lands before the test polls, so the row count goes from
PAGE straight past 2 * PAGE and an exact wait for 2 * PAGE never holds. This
script models that: when the second page lands, the third page is appended in
the same call. It runs the paging tests from BASE_REF (default: the parent of
the fix, bd872024e) and from the working tree with that plugin on. The old
cursor test must fail and the new one must pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_NAME = "test_tui_status_paging_old.py"
TEST = "test_moving_the_cursor_near_the_end_loads_the_next_page"

PLUGIN = """
import pytest


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    from lilbee.cli.tui.screens.status import StatusScreen, _DocsPage

    orig = StatusScreen._append_docs_page

    def append(self, page):
        orig(self, page)
        if self._docs_loaded == 2 * len(page.sources) and self._docs_has_more:
            from lilbee.app.services import get_services

            store = get_services().store
            more = store.get_sources(limit=len(page.sources), offset=self._docs_loaded)
            orig(self, _DocsPage(sources=more))

    StatusScreen._append_docs_page = append
"""


def run(plugin_dir: Path, test_file: str) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": f"{plugin_dir}:{ROOT / 'src'}"}
    cmd = [
        sys.executable, "-m", "pytest", f"{test_file}::{TEST}", "-p", "delay_plugin",
        "-q", "--no-cov", "-p", "no:cacheprovider",
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    lines = proc.stdout.strip().splitlines() or [proc.stderr[-300:]]
    return proc.returncode, lines[-1]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "bd872024e"
    old_src = subprocess.run(
        ["git", "show", f"{base}:tests/test_tui_status_paging.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    old_path = ROOT / "tests" / OLD_NAME
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "delay_plugin.py").write_text(PLUGIN)
        old_path.write_text(old_src)
        try:
            old_rc, old_tail = run(Path(tmp), f"tests/{OLD_NAME}")
            new_rc, new_tail = run(Path(tmp), "tests/test_tui_status_paging.py")
        finally:
            old_path.unlink()
    print(f"old test ({base}) with the third page: rc={old_rc} {old_tail}")
    print(f"new test (working tree) with the third page: rc={new_rc} {new_tail}")
    return 0 if old_rc != 0 and new_rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
