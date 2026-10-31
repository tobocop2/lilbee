"""Checks a real engine: --engine-url for a running one, --engine-pin to launch a pinned binary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conformance.harness import (
    CLAIMS,
    Verdict,
    launch,
    run,
    run_managed,
    write_record,
)

_ENGINE_TIMEOUT_S = 180.0
_RUN_TIMEOUT_S = 3600

# A real run takes minutes, so the repo-wide 60 second test timeout does not apply to this module.
pytestmark = pytest.mark.timeout(_RUN_TIMEOUT_S)


def _claims(config: pytest.Config) -> set[str]:
    claims = {c.strip() for c in str(config.getoption("--claims", default="chat")).split(",") if c}
    unknown = claims - CLAIMS
    assert not unknown, f"--claims names {sorted(unknown)}; the roles are {sorted(CLAIMS)}"
    return claims


def _models(config: pytest.Config, claims: set[str]) -> dict[str, str]:
    pairs = str(config.getoption("--models", default="") or "")
    named = dict(pair.split("=", 1) for pair in pairs.split(",") if pair)
    missing = claims - set(named) - {"reasoning"}
    assert not missing, f"--models gives no id for {sorted(missing)}; pass --models role=ID,..."
    return named


def _record_path(config: pytest.Config, tmp_path: Path) -> Path:
    given = config.getoption("--record", default=None)
    return Path(given) if given else tmp_path / "conformance-record.jsonl"


def _assert_supported(verdicts: dict[str, Verdict]) -> None:
    failed = {key: str(value) for key, value in verdicts.items() if value is Verdict.FAIL}
    assert not failed, f"claimed roles fail {failed}"


def test_running_engine_conforms(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    url = pytestconfig.getoption("--engine-url", default=None)
    if url is None:
        pytest.skip("no --engine-url")
    claims = _claims(pytestconfig)
    verdicts = run(
        url,
        claims=claims,
        models=_models(pytestconfig, claims),
        health_path=pytestconfig.getoption("--health-path", default=None),
        quality=True,
        timeout=_ENGINE_TIMEOUT_S,
    )
    write_record(_record_path(pytestconfig, tmp_path), verdicts, None)
    _assert_supported(verdicts)


def test_pinned_engine_conforms(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    path = pytestconfig.getoption("--engine-pin", default=None)
    if path is None:
        pytest.skip("no --engine-pin")
    pin: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    binary = Path(pin["binary"])
    claims = _claims(pytestconfig)
    started = launch(binary, pin, timeout=_ENGINE_TIMEOUT_S)
    assert started is not None, "the launch template did not start an engine that answers /health"
    proc, port = started
    try:
        tier1 = run(
            f"http://127.0.0.1:{port}",
            claims=claims,
            models=_models(pytestconfig, claims),
            health_path=pytestconfig.getoption("--health-path", default=None),
            quality=True,
            timeout=_ENGINE_TIMEOUT_S,
        )
    finally:
        proc.kill()
        proc.wait()
    tier2 = run_managed(binary, pin=pin, timeout=_ENGINE_TIMEOUT_S)
    write_record(_record_path(pytestconfig, tmp_path), {**tier1, **tier2}, pin)
    _assert_supported({**tier1, **tier2})
