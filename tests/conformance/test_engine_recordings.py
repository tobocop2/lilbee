"""Every engine recording sets strict, so a recorded run proves the strict steps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conformance.preset_check import harness_recordings

ENGINE_RECORDINGS = Path(__file__).parent / "fixtures" / "engines"

_RECORDINGS = sorted(ENGINE_RECORDINGS.glob("**/*.json"))
# A harness run has "results" and sets strict. A hand run carries "source" and the exchanges.
_HARNESS_RUNS = harness_recordings()
_HAND_RUNS = [path for path in _RECORDINGS if path not in _HARNESS_RUNS]


def test_the_engine_recordings_are_found() -> None:
    assert len(_RECORDINGS) >= 5


@pytest.mark.parametrize("path", _HARNESS_RUNS, ids=lambda path: path.stem)
def test_an_engine_recording_sets_strict(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc.get("strict") is True, f"{path.name} does not set strict"


def test_the_hand_runs_are_found() -> None:
    assert {path.stem for path in _HAND_RUNS} >= {
        "openvino-cpu",
        "openvino-gpu",
        "lemonade-cpu-thinkpad",
    }


@pytest.mark.parametrize("path", _HAND_RUNS, ids=lambda path: path.stem)
def test_a_hand_run_names_its_source_machine(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["source"].startswith("machine: ")
    assert path.with_suffix(".log").read_text(encoding="utf-8").startswith("source: ")
