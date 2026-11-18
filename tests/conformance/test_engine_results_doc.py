"""Each row of docs/engine-results.md says what its recorded fixture holds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "engine-results.md"
FIXTURES = Path(__file__).parent / "fixtures" / "engines"
NOT_RECORDED = "not recorded"


class Row(NamedTuple):
    recording: str
    roles: str
    machine: str
    result: str
    version: str
    fixture: str


def _rows() -> list[Row]:
    lines = DOC.read_text(encoding="utf-8").split("\n")
    start = next(n for n, line in enumerate(lines) if line.startswith("| Recording | Roles |"))
    rows: list[Row] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(Row._fields), f"a row with {len(cells)} cells: {line}"
        row = Row(*cells)
        rows.append(row._replace(fixture=row.fixture.strip("`")))
    return rows


ROWS = _rows()
RECORDED = [row for row in ROWS if row.fixture]
JSON_ROWS = [row for row in RECORDED if row.fixture.endswith(".json")]


def _doc(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _log_text(name: str) -> str:
    return (FIXTURES / name).with_suffix(".log").read_text(encoding="utf-8")


def _machine(log: str) -> str:
    """The machine a harness log names: the GPU when it has one, else the CPU host."""
    gpu = [line for line in log.split("\n") if line.startswith("gpu: ")]
    return gpu[0].removeprefix("gpu: ").split(",")[0] if gpu else "CPU host"


def test_the_table_has_recorded_and_unrecorded_rows() -> None:
    assert len(RECORDED) >= 8
    assert any(row.result.startswith(NOT_RECORDED) for row in ROWS)


@pytest.mark.parametrize("row", RECORDED, ids=lambda row: row.recording)
def test_a_recorded_row_names_a_fixture_that_exists(row: Row) -> None:
    assert (FIXTURES / row.fixture).is_file(), f"{row.recording} cites {row.fixture}"
    assert not row.result.startswith(NOT_RECORDED)


@pytest.mark.parametrize(
    "row",
    [row for row in JSON_ROWS if "results" in _doc(row.fixture)],
    ids=lambda row: row.recording,
)
def test_a_harness_row_matches_its_recorded_steps(row: Row) -> None:
    doc = _doc(row.fixture)
    steps = doc["results"]
    failed = [name for name, step in steps.items() if not step["ok"]]
    assert steps, f"{row.fixture} holds no step"
    expected = f"pass, {len(steps)} of {len(steps)} steps"
    if failed:
        expected = f"fail: {', '.join(failed)}"
    assert row.result == expected
    assert row.roles == ", ".join(doc["roles"])
    assert row.version == doc["version"]
    assert row.machine == _machine(_log_text(row.fixture))
    assert row.recording == doc["server"]


@pytest.mark.parametrize(
    "row",
    [row for row in JSON_ROWS if "results" not in _doc(row.fixture)],
    ids=lambda row: row.recording,
)
def test_a_hand_run_row_matches_its_exchanges(row: Row) -> None:
    doc = _doc(row.fixture)
    [exchange] = doc["exchanges"]
    assert exchange["path"] == "/v1/chat/completions"
    assert row.result == f"hand run: 1 chat request, HTTP {exchange['status']}, no strict steps"
    assert row.roles == ", ".join(doc["roles"])
    assert row.version == doc["version"]
    assert row.machine in doc["source"]
    assert row.recording == doc["server"]


@pytest.mark.parametrize(
    "row",
    [row for row in RECORDED if row.fixture == "openvino-gpu-no-runtime.log"],
    ids=lambda row: row.recording,
)
def test_a_log_only_row_matches_its_log(row: Row) -> None:
    log = (FIXTURES / row.fixture).read_text(encoding="utf-8")
    assert row.machine in log
    assert row.version in log
    assert row.result.startswith("fail") == ("Aborted (core dumped)" in log)


def test_the_device_list_row_matches_its_log() -> None:
    [row] = [r for r in RECORDED if r.fixture == "openvino-two-gpu.list-devices.log"]
    log = (FIXTURES / row.fixture).read_text(encoding="utf-8")
    assert row.machine in log
    assert log.count("GGML_OPENVINO_DEVICE=") == 3
    assert "[exit 0]" in log
    assert row.result.startswith("pass")


@pytest.mark.parametrize(
    "row",
    [row for row in ROWS if not row.fixture],
    ids=lambda row: row.recording,
)
def test_an_unrecorded_row_has_no_fixture(row: Row) -> None:
    assert row.result.startswith(NOT_RECORDED)
    assert not row.version
    assert not list(FIXTURES.glob(f"{row.recording}*"))


def test_every_fixture_has_a_row() -> None:
    cited = {row.fixture for row in RECORDED}
    stems = {row.fixture.removesuffix(".json") for row in RECORDED if row.fixture.endswith(".json")}
    held = {path.name for path in FIXTURES.iterdir()}
    assert held
    uncited = {n for n in held if n not in cited and n.removesuffix(".log") not in stems}
    assert not uncited, f"no row cites {sorted(uncited)}"


@pytest.mark.parametrize("name", ["engine-integration.md", "presets.md"])
def test_the_results_page_is_linked(name: str) -> None:
    assert "(engine-results.md)" in (REPO / "docs" / name).read_text(encoding="utf-8")
