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
    monkeypatch.setattr(base_mod, "_warned", {}, raising=False)


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
            f"sys.exit({code})\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

    return _install


@pytest.mark.parametrize("name", sorted(_VENDORS))
def test_rows_printed_before_a_non_zero_exit_are_kept(
    name: str, tools, caplog: pytest.LogCaptureFixture
) -> None:
    backend, rows = _VENDORS[name]
    tools(name, rows, 1, _STDERR)
    with caplog.at_level(logging.WARNING):
        samples = backend().sample(_INDICES)
    assert set(samples) == _INDICES
    assert any(name in r.message and _STDERR in r.message for r in caplog.records)


@pytest.mark.parametrize("name", sorted(_VENDORS))
def test_a_non_zero_exit_with_no_rows_gives_none_and_a_warning(
    name: str, tools, caplog: pytest.LogCaptureFixture
) -> None:
    backend, _ = _VENDORS[name]
    tools(name, "", 1, _STDERR)
    with caplog.at_level(logging.WARNING):
        samples = backend().sample(_INDICES)
    assert samples == {}
    assert any("printed no rows" in r.message and _STDERR in r.message for r in caplog.records)


def test_a_repeated_failure_logs_once_and_a_clean_run_rearms_it(
    tools, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
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


def test_one_failing_device_among_working_ones_logs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """xpu-smi runs once per device: a device that keeps failing is one warning, not one per poll."""
    monkeypatch.setenv("PATH", str(tmp_path))
    script = tmp_path / "xpu-smi"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nsys.exit(1 if sys.argv[3] == '1' else 0)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            run_smi("xpu-smi", ["stats", "-d", "0", "-j"])
            run_smi("xpu-smi", ["stats", "-d", "1", "-j"])
    assert len(caplog.records) == 1


def test_only_the_tail_of_a_flooding_stderr_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    script = tmp_path / "ioreg"
    script.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "sys.stderr.write('x' * 1000000 + 'THE-END')\nsys.exit(1)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    with caplog.at_level(logging.WARNING):
        run_smi("ioreg", [])
    (record,) = caplog.records
    assert record.message.endswith("THE-END")
    assert len(record.message) < 5000


def test_rocm_smi_missing_a_device_falls_back_to_amd_smi_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.providers.fleet.gpu_backends import amd as amd_mod

    amd_rows = (
        '[{"gpu": 0, "gfx_activity": 10, "temperature_c": 1},'
        ' {"gpu": 1, "gfx_activity": 72, "temperature_c": 61}]'
    )
    monkeypatch.setattr(
        amd_mod,
        "run_smi",
        lambda tool, *_a, **_k: _ROCM_ROWS if tool == amd_mod._TOOL_ROCM_SMI else amd_rows,
    )
    samples = AmdBackend().sample(frozenset({0, 1}))
    assert samples[0].utilization_pct != 10  # rocm-smi keeps the device it printed
    assert samples[1].utilization_pct == 72  # amd-smi fills the one it missed


def test_a_rocm_smi_that_covers_every_device_never_runs_amd_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.providers.fleet.gpu_backends import amd as amd_mod

    def _run(tool: str, *_a: object, **_k: object) -> str:
        assert tool == amd_mod._TOOL_ROCM_SMI
        return _ROCM_ROWS

    monkeypatch.setattr(amd_mod, "run_smi", _run)
    assert set(AmdBackend().sample(_INDICES)) == _INDICES


@pytest.mark.parametrize("raw", ["5", "null", '{"gpu_data": 5}', '{"gpu": "x"}'])
def test_amd_smi_scalar_json_gives_no_rows(raw: str) -> None:
    from lilbee.providers.fleet.gpu_backends.amd import _parse_amd_smi

    assert _parse_amd_smi(raw, _INDICES) == {}


def test_a_tool_that_cannot_start_logs_nothing(tools, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert run_smi("ioreg", []) == ""
    assert caplog.records == []
