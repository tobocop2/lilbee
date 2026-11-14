"""Replays each recorded preset run through the real provider, with no server."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.factory import sdk_library
from tests.conformance.preset_check import (
    FIXTURES,
    Outcome,
    configured,
    error_answers,
    load_fixture,
    recording,
    replaying,
    run_steps,
    steps_for,
)

# The replay drives the real provider; a missing library fails the import.
importlib.import_module(sdk_library().module)

_FILES = sorted(FIXTURES.glob("*.json"))


def test_fixtures_exist() -> None:
    assert _FILES, f"no preset fixture under {FIXTURES}"


_RESULTS_DOC = Path(__file__).resolve().parents[2] / "docs" / "preset-results.md"
_ROW_STEPS = {
    "chat template": "template",
    "tool call": "tool",
    "context overflow": "overflow",
    "embed": "embed",
    "rerank": "rerank",
    "vision": "vision",
    "named": "named",
}
_CITE = re.compile(r"`([a-z0-9-]+)`(?: \((pass|fail)\))?")
_WIRE = re.compile(r" \(wire `[a-z0-9]+`\)$")


def _claims() -> tuple[dict[tuple[str, str], bool], list[str]]:
    """The result each pass or fail row claims for a (fixture, step), and the rows in error."""
    claims: dict[tuple[str, str], bool] = {}
    problems: list[str] = []
    lines = _RESULTS_DOC.read_text(encoding="utf-8").split("\n")
    start = next(n for n, line in enumerate(lines) if line.startswith("| Server | Role |"))
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        server, role, result, _version, cited = (c.strip() for c in line.strip("|").split(" | "))
        verdict = re.match(r"(pass|fail)\b", result)
        if verdict is None:
            if cited:
                problems.append(f"{server} {role}: a row with no result cites {cited}")
            continue
        step = _ROW_STEPS.get(_WIRE.sub("", role))
        cites = _CITE.findall(cited)
        if step is None or not cites:
            problems.append(f"{server} {role}: no step or no fixture for a {verdict[1]} row")
        for name, own in cites:
            if (name, step) in claims:
                problems.append(f"{name} {step}: two rows claim it")
            claims[name, step] = (own or verdict[1]) == "pass"
    return claims, problems


def _replayed(path: Path, data_root: Path) -> tuple[dict[str, Outcome], list[list[Any]], Any]:
    """Replay *path* through a recording proxy; return the results and the error answers the
    client received on the wire."""
    setup, _recorded, exchanges, doc = load_fixture(path)
    with (
        replaying(exchanges) as url,
        recording(url) as (proxy, tape),
        configured(proxy, setup, data_root),
    ):
        results = run_steps(setup)
    return results, error_answers(tape.exchanges), doc


def test_every_row_names_a_fixture_that_exists() -> None:
    claims, problems = _claims()
    assert not problems
    assert claims
    assert {name for name, _step in claims} <= {path.stem for path in _FILES}


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_gives_the_results_of_the_rows(path: Path, tmp_path: Path) -> None:
    claims, _problems = _claims()
    replayed, _served, _doc = _replayed(path, tmp_path)
    unclaimed = [step for step in replayed if (path.stem, step) not in claims]
    assert not unclaimed, f"no row cites {path.stem} for the steps {unclaimed}"
    missing = [step for name, step in claims if name == path.stem and step not in replayed]
    assert not missing, f"a row cites {path.stem} for the steps {missing}, which it does not hold"
    assert {step: o.ok for step, o in replayed.items()} == {
        step: ok for (name, step), ok in claims.items() if name == path.stem
    }


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_serves_the_recorded_error_answers(path: Path, tmp_path: Path) -> None:
    _setup, _recorded, exchanges, _doc = load_fixture(path)
    _results, received, doc = _replayed(path, tmp_path)
    assert "error_answers" in doc, f"{path.name} holds no error_answers field"
    assert doc["error_answers"] == error_answers(exchanges)
    assert received == doc["error_answers"]


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_gives_the_recorded_results(path: Path, tmp_path: Path) -> None:
    setup, recorded, exchanges, _doc = load_fixture(path)
    assert exchanges, "the fixture holds no exchange"
    assert set(recorded) == set(steps_for(setup))
    with replaying(exchanges) as url, configured(url, setup, tmp_path):
        replayed = run_steps(setup)
    assert replayed == recorded
