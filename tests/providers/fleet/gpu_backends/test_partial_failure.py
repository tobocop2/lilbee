"""A vendor smi tool that prints rows and then exits non-zero still gives its rows.

Each case runs a real fake tool found through ``PATH``, so the exit code and stderr
travel the same path as in production.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from lilbee.providers.fleet.gpu_backends import base as base_mod
from lilbee.providers.fleet.gpu_backends.amd import AmdBackend
from lilbee.providers.fleet.gpu_backends.apple import AppleBackend
from lilbee.providers.fleet.gpu_backends.base import run_smi
from lilbee.providers.fleet.gpu_backends.intel import IntelBackend

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the fake tools are POSIX scripts")

_ROCM_ROWS = '{"card0": {"GPU use (%)": "35", "Temperature (Sensor edge) (C)": "52"}}'
_AMD_ROWS = '[{"gpu": 0, "gfx_activity": 72, "temperature_c": 61}]'
_XPU_ROWS = (
    '{"device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 88},'
    ' {"metrics_type": "XPUM_STATS_GPU_CORE_TEMPERATURE", "value": 73}]}'
)
_IOREG_ROWS = '    | |   "PerformanceStatistics" = {"Device Utilization %"=42}'
_STDERR = "device 1 failed to answer"
_INDICES = frozenset({0})

_VENDORS = {
    "rocm-smi": (lambda: AmdBackend(), _ROCM_ROWS),
    "amd-smi": (lambda: AmdBackend(), _AMD_ROWS),
    "xpu-smi": (lambda: IntelBackend(), _XPU_ROWS),
    "ioreg": (lambda: AppleBackend(), _IOREG_ROWS),
}


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "_warned", {})


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install fake tools by name; the only folder on ``PATH`` is the one that holds them."""
    monkeypatch.setenv("PATH", str(tmp_path))

    def _install(name: str, printed: str, code: int, err: str = "") -> None:
        script = tmp_path / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"sys.stdout.write({printed!r})\n"
            f"sys.stderr.write({err!r})\n"
            f"sys.exit({code})\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

    return _install


@pytest.mark.parametrize("name", sorted(_VENDORS))
def test_rows_printed_before_a_non_zero_exit_are_kept(
    name: str, tools, caplog: pytest.LogCaptureFixture
) -> None:
    backend, rows = _VENDORS[name]
    tools(name, rows, 1, _STDERR)
    with caplog.at_level(logging.WARNING, logger=base_mod.log.name):
        samples = backend().sample(_INDICES)
    assert set(samples) == _INDICES
    assert any(name in r.message and _STDERR in r.message for r in caplog.records)


@pytest.mark.parametrize("name", sorted(_VENDORS))
def test_a_non_zero_exit_with_no_rows_gives_none_and_a_warning(
    name: str, tools, caplog: pytest.LogCaptureFixture
) -> None:
    backend, _ = _VENDORS[name]
    tools(name, "", 1, _STDERR)
    with caplog.at_level(logging.WARNING, logger=base_mod.log.name):
        samples = backend().sample(_INDICES)
    assert samples == {}
    assert any("printed no rows" in r.message and _STDERR in r.message for r in caplog.records)


def test_a_repeated_failure_logs_once_and_a_clean_run_rearms_it(
    tools, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=base_mod.log.name):
        tools("ioreg", _IOREG_ROWS, 1)
        run_smi("ioreg", [])
        run_smi("ioreg", [])
        assert len(caplog.records) == 1
        assert "(none)" in caplog.records[0].message
        tools("ioreg", _IOREG_ROWS, 0)
        run_smi("ioreg", [])
        tools("ioreg", _IOREG_ROWS, 1)
        run_smi("ioreg", [])
    assert len(caplog.records) == 2


def test_a_tool_that_cannot_start_logs_nothing(tools, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=base_mod.log.name):
        assert run_smi("ioreg", []) == ""
    assert caplog.records == []
