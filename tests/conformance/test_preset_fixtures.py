"""Replays each recorded preset run through the real provider, with no server."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conformance.preset_check import (
    FIXTURES,
    configured,
    load_fixture,
    replaying,
    run_steps,
    steps_for,
)

_FILES = sorted(FIXTURES.glob("*.json"))


def test_fixtures_exist() -> None:
    assert _FILES, f"no preset fixture under {FIXTURES}"


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_gives_the_recorded_results(path: Path, tmp_path: Path) -> None:
    setup, recorded, exchanges, _doc = load_fixture(path)
    assert exchanges, "the fixture holds no exchange"
    assert set(recorded) == set(steps_for(setup))
    with replaying(exchanges) as url, configured(url, setup, tmp_path):
        replayed = run_steps(setup)
    assert replayed == recorded
