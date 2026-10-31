"""Snap metadata: every snapcraft.yaml passes the snapcraft field limits."""

from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SNAPCRAFT_FILES = sorted((_ROOT / "packaging").glob("snap*/snapcraft.yaml"))
_SUMMARY_LIMIT = 78


def test_snapcraft_files_are_found() -> None:
    assert len(_SNAPCRAFT_FILES) >= 5


@pytest.mark.parametrize("path", _SNAPCRAFT_FILES, ids=lambda p: p.parent.name)
def test_summary_fits_the_snapcraft_limit(path: Path) -> None:
    summary = yaml.safe_load(path.read_text(encoding="utf-8"))["summary"]
    assert len(summary) <= _SUMMARY_LIMIT, f"{len(summary)} characters: {summary}"
