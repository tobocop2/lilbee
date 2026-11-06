"""The profile recorder that each release attaches as ``profile-responses.json``.

Clients replay these bodies in their own tests, so the recorder has to keep
running against the current app. It runs in a subprocess because it sets the
home and data root before lilbee is imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.record_profile_responses import _scenario, _scrub

ROOT = Path(__file__).resolve().parent.parent

REFUSALS = {
    "error_missing": 404,
    "error_clash": 409,
    "error_builtin": 400,
    "error_reserved": 400,
    "error_too_large": 400,
    "error_no_route": 404,
}
PLACEHOLDERS = ("<package>/", "<project>/", "<global>/")


def test_recorder_writes_every_profile_response(tmp_path):
    out = tmp_path / "profile-responses.json"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "record_profile_responses.py"), str(out)],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    recorded = json.loads(out.read_text(encoding="utf-8"))

    assert set(recorded) == {label for label, *_ in _scenario() if label is not None}
    successes = {label: entry for label, entry in recorded.items() if label not in REFUSALS}
    assert all(200 <= entry["status"] < 300 for entry in successes.values()), successes
    assert {label: recorded[label]["status"] for label in REFUSALS} == REFUSALS
    assert recorded["export"]["content_disposition"].startswith("attachment")

    names = {profile["name"] for profile in recorded["list"]["body"]["profiles"]}
    assert {"Default", "Court filings"} <= names
    paths = list(_values_of(recorded, "path"))
    assert paths, "the recording names no profile paths to check"
    leaked = [path for path in paths if not path.startswith(PLACEHOLDERS)]
    assert leaked == [], "a throwaway path leaked into the recording"


def test_scrub_swaps_a_windows_root_in_either_separator_form():
    """A Windows root matches with ``\\`` or ``/``, and the result uses ``/``."""
    roots = {"C:\\tmp\\p": "<project>"}
    recorded = {
        "path": "C:\\tmp\\p\\profiles\\mine.toml",
        "detail": "cannot read C:\\tmp\\p\\profiles",
        "posix": "C:/tmp/p/profiles/theirs.toml",
        "rows": [3, None],
    }
    assert _scrub(recorded, roots) == {
        "path": "<project>/profiles/mine.toml",
        "posix": "<project>/profiles/theirs.toml",
        "detail": "cannot read <project>\\profiles",
        "rows": [3, None],
    }


def _values_of(value, key):
    """Every string under ``key`` anywhere in a JSON value."""
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key and isinstance(item, str):
                yield item
            else:
                yield from _values_of(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _values_of(item, key)
