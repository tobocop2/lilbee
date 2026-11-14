"""Every engine recording sets strict, so a recorded run proves the strict steps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ENGINE_RECORDINGS = Path(__file__).parent / "fixtures" / "engines"

_RECORDINGS = sorted(ENGINE_RECORDINGS.glob("**/*.json"))


def test_the_engine_recordings_are_found() -> None:
    assert len(_RECORDINGS) >= 5


@pytest.mark.parametrize("path", _RECORDINGS, ids=lambda path: path.stem)
def test_an_engine_recording_sets_strict(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc.get("strict") is True, f"{path.name} does not set strict"
