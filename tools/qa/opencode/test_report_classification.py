"""Unit tests for the supported/unsupported cell classification and run verdict."""

from __future__ import annotations

from harness_config import ExpectedSupport, ScenarioStatus
from report import CellResult
from scenarios import ScenarioResult


def _pass_scenario() -> ScenarioResult:
    return ScenarioResult(name="small godot", status=ScenarioStatus.PASS)


def _fail_scenario() -> ScenarioResult:
    return ScenarioResult(name="small godot", status=ScenarioStatus.FAIL)


def _cell(scenario: ScenarioResult, expected: ExpectedSupport) -> CellResult:
    return CellResult(
        family="x",
        ref="r",
        scenarios=[scenario],
        chat_completions_ok=2,
        expected=expected,
    )


def test_supported_pass_is_pass() -> None:
    cell = _cell(_pass_scenario(), ExpectedSupport.SUPPORTED)
    assert cell.passed
    assert cell.classification == "PASS"
    assert not cell.is_regression


def test_supported_fail_is_regression() -> None:
    cell = _cell(_fail_scenario(), ExpectedSupport.SUPPORTED)
    assert not cell.passed
    assert cell.classification == "REGRESSION"
    assert cell.is_regression


def test_unsupported_fail_is_expected_not_regression() -> None:
    cell = _cell(_fail_scenario(), ExpectedSupport.UNSUPPORTED)
    assert cell.classification == "expected-fail"
    assert not cell.is_regression


def test_unsupported_pass_is_newly_working() -> None:
    cell = _cell(_pass_scenario(), ExpectedSupport.UNSUPPORTED)
    assert cell.passed
    assert cell.classification == "newly-working"
    assert not cell.is_regression


def test_serve_error_downgrades_supported_to_regression() -> None:
    cell = CellResult(
        family="x",
        ref="r",
        scenarios=[_pass_scenario()],
        chat_completions_ok=2,
        serve_errors="WorkerError: boom",
        expected=ExpectedSupport.SUPPORTED,
    )
    assert not cell.passed
    assert cell.is_regression
