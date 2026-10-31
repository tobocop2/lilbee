"""Replays each recorded engine run through the real provider, with no server."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.factory import sdk_library
from tests.conformance.preset_check import (
    ENGINE_FIXTURES,
    configured,
    load_fixture,
    replaying,
    run_steps,
    steps_for,
)

# The replay drives the real provider, so it runs where the extra is installed:
# the integration job names this directory's tests in `make test-integration`.
pytest.importorskip(sdk_library().module)

_FILES = sorted(ENGINE_FIXTURES.glob("**/*.json"))


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_gives_the_recorded_results(path: Path, tmp_path: Path) -> None:
    setup, recorded, exchanges, doc = load_fixture(path)
    assert doc.get("strict") is True, "an engine recording sets strict"
    assert exchanges, "the fixture holds no exchange"
    assert set(recorded) == set(steps_for(setup))
    with replaying(exchanges) as url, configured(url, setup, tmp_path):
        replayed = run_steps(setup)
    assert replayed == recorded
    assert all(outcome.ok for outcome in recorded.values())
