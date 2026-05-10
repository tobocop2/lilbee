"""T1 CLI self-check-extras. Frozen-binary smoke gate for bundled optional extras."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import EXTRAS_PROBE_TIMEOUT, Lane, run_lilbee

_EXPECTED_EXTRAS = ("litellm", "crawl4ai", "spacy", "graspologic_native")


@pytest.mark.cli
def test_self_check_extras_json_shape(lane: Lane, lilbee_data: Path) -> None:
    """`--json self-check-extras` emits a payload with one bool per known extra."""
    result = run_lilbee(
        lane, ["--json", "self-check-extras"], data_dir=lilbee_data, timeout=EXTRAS_PROBE_TIMEOUT
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(payload, dict), payload
    assert isinstance(payload.get("ok"), bool), payload
    for name in _EXPECTED_EXTRAS:
        assert name in payload, f"extras report missing key '{name}': {payload}"
        assert isinstance(payload[name], bool), payload


@pytest.mark.cli
def test_self_check_extras_passes_on_binary_lane(lane: Lane, lilbee_data: Path) -> None:
    """The release binary bundles all four extras; the binary lane reports ok=True.

    The pypi lane installs the bare wheel without extras, so it is excluded
    here; that lane's expected behavior is exit_code=1 with one or more
    extras reported False, which the JSON-shape test already covers.
    """
    if not lane.is_binary:
        pytest.skip("the bundled-extras invariant only applies to the release binary")
    result = run_lilbee(
        lane, ["--json", "self-check-extras"], data_dir=lilbee_data, timeout=EXTRAS_PROBE_TIMEOUT
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True, payload
    for name in _EXPECTED_EXTRAS:
        assert payload[name] is True, f"extra '{name}' missing from binary: {payload}"
