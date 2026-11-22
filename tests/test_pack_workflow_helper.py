"""The helper that finds the engine-packs workflow fails in CI and skips elsewhere."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.outcomes import Failed, Skipped

from tests._pack_workflow import (
    CELLS_ENV,
    WORKFLOW_ENV,
    cells_below_the_record,
    pack_cells,
    pack_pins,
    pack_workflow,
    published_builds,
    record_after_publish,
)


def test_the_helper_returns_the_named_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKFLOW_ENV, "/packs/build-packs.yml")
    monkeypatch.setenv("CI", "true")
    assert pack_workflow() == Path("/packs/build-packs.yml")


@pytest.mark.parametrize("value", [None, ""])
def test_the_helper_fails_in_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.setenv("CI", "true")
    if value is None:
        monkeypatch.delenv(WORKFLOW_ENV, raising=False)
    else:
        monkeypatch.setenv(WORKFLOW_ENV, value)
    with pytest.raises(Failed, match=WORKFLOW_ENV):
        pack_workflow()


def test_the_helper_skips_outside_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(WORKFLOW_ENV, raising=False)
    with pytest.raises(Skipped, match=WORKFLOW_ENV):
        pack_workflow()


def test_published_builds_reads_the_numbers_of_the_rows_that_name_the_wheel() -> None:
    dist, version = "lilbee_engine_llama_cpp", "1.12032.1"
    builds = published_builds(dist, version, "cpu", "manylinux_2_34_x86_64")
    assert builds
    assert all(isinstance(number, int) for number in builds)
    assert published_builds(dist, version, "cpu", "no_such_platform") == []
    assert published_builds(dist, "0", "cpu", "manylinux_2_34_x86_64") == []


_ROWS = [
    {"file": "lilbee_engine_x-1-2.cpu-py3-none-linux.whl"},
    {"file": "lilbee_engine_x-1-3.cpu-py3-none-linux.whl"},
    {"file": "lilbee_engine_x-1-9.cuda-py3-none-linux.whl"},
]


def test_a_cell_below_the_highest_row_of_its_wheel_name_is_below_the_record() -> None:
    cells = [("old", "cpu", "linux", 2), ("equal", "cpu", "linux", 3), ("next", "cpu", "linux", 4)]
    assert cells_below_the_record("lilbee_engine_x", "1", cells, _ROWS) == ["old"]


def test_a_cell_is_compared_with_the_rows_of_its_own_backend_only() -> None:
    cells = [("cpu", "cpu", "linux", 3), ("cuda", "cuda", "linux", 3)]
    assert cells_below_the_record("lilbee_engine_x", "1", cells, _ROWS) == ["cuda"]


def test_a_cell_with_no_row_is_refused_for_proving_nothing() -> None:
    with pytest.raises(AssertionError, match="orphan"):
        cells_below_the_record("lilbee_engine_x", "1", [("orphan", "cpu", "win", 1)], _ROWS)


def test_the_record_after_a_publish_holds_one_row_for_each_cell_at_its_number() -> None:
    cells = [("a", "cpu", "linux", 4)]
    after = record_after_publish("lilbee_engine_x", "1", cells, _ROWS)
    assert published_builds("lilbee_engine_x", "1", "cpu", "linux", after) == [4]
    assert after[0] == _ROWS[2]
    assert published_builds("lilbee_engine_x", "1", "cpu", "linux", _ROWS) == [2, 3]


_CELLS_FILE = (
    "engines:\n  llama:\n    pins: {TAG: b1}\n"
    "    cells:\n      - {cell: a, build_number: 3}\n      - {cell: b}\n"
)


def test_pack_pins_returns_the_pins_of_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells_file = tmp_path / "cells.yml"
    cells_file.write_text(_CELLS_FILE, encoding="utf-8")
    monkeypatch.setenv(CELLS_ENV, str(cells_file))
    assert pack_pins("llama") == {"TAG": "b1"}


def test_pack_cells_returns_the_cells_of_the_engine_in_file_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells_file = tmp_path / "cells.yml"
    cells_file.write_text(_CELLS_FILE, encoding="utf-8")
    monkeypatch.setenv(CELLS_ENV, str(cells_file))
    assert pack_cells("llama") == [{"cell": "a", "build_number": 3}, {"cell": "b"}]


def test_pack_cells_fails_for_an_engine_the_file_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells_file = tmp_path / "cells.yml"
    cells_file.write_text(_CELLS_FILE, encoding="utf-8")
    monkeypatch.setenv(CELLS_ENV, str(cells_file))
    with pytest.raises(KeyError, match="nope"):
        pack_cells("nope")


@pytest.mark.parametrize("value", [None, ""])
def test_pack_cells_fails_in_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.setenv("CI", "true")
    if value is None:
        monkeypatch.delenv(CELLS_ENV, raising=False)
    else:
        monkeypatch.setenv(CELLS_ENV, value)
    with pytest.raises(Failed, match=CELLS_ENV):
        pack_cells("llama")


def test_pack_cells_skips_outside_ci_when_the_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(CELLS_ENV, raising=False)
    with pytest.raises(Skipped, match=CELLS_ENV):
        pack_cells("llama")


def test_the_checkout_action_sets_both_variables_the_pack_tests_read() -> None:
    action = Path(__file__).parents[1] / ".github/actions/checkout-engine-packs/action.yml"
    text = action.read_text(encoding="utf-8")
    assert f"{WORKFLOW_ENV}=" in text
    assert f"{CELLS_ENV}=" in text
    assert "engine-packs/cells.yml" in text
