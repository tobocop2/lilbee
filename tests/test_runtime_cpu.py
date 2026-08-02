"""Tests for ``lilbee.runtime.cpu`` sizing helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from lilbee.runtime.cpu import (
    _cgroup_cpu_quota,
    available_cpu_count,
    cpu_quota,
    engine_thread_count,
)


def test_default_is_half_of_available() -> None:
    with mock.patch("lilbee.runtime.cpu.available_cpu_count", return_value=8):
        assert cpu_quota() == 4


def test_default_floors_at_one() -> None:
    with mock.patch("lilbee.runtime.cpu.available_cpu_count", return_value=1):
        assert cpu_quota() == 1


def test_env_override_accepts_positive_int(monkeypatch) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "12")
    assert cpu_quota() == 12


def test_env_override_rejects_zero(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "0")
    with mock.patch("lilbee.runtime.cpu.available_cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)


def test_env_override_rejects_negative(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "-3")
    with mock.patch("lilbee.runtime.cpu.available_cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)


def test_env_override_rejects_non_integer(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "garbage")
    with mock.patch("lilbee.runtime.cpu.available_cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)


def _affinity(cpus: int):
    """Pin the affinity signal. Unpatched, the runner's own mask joins the min
    and a test that fixes cpu_count and the quota still reads the host."""
    return mock.patch(
        "lilbee.runtime.cpu.os.sched_getaffinity", return_value=set(range(cpus)), create=True
    )


def _write_cgroup(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def test_cgroup_v2_quota_keeps_the_fraction(tmp_path) -> None:
    # Unrounded: available_cpu_count rounds it up, engine_thread_count down.
    _write_cgroup(tmp_path, {"cpu.max": "250000 100000\n"})
    assert _cgroup_cpu_quota(tmp_path) == 2.5


def test_cgroup_v2_unlimited_is_none(tmp_path) -> None:
    _write_cgroup(tmp_path, {"cpu.max": "max 100000\n"})
    assert _cgroup_cpu_quota(tmp_path) is None


def test_cgroup_v2_malformed_is_none(tmp_path) -> None:
    # A single-field or non-integer cpu.max is unparseable, not a limit.
    _write_cgroup(tmp_path, {"cpu.max": "notanumber\n"})
    assert _cgroup_cpu_quota(tmp_path) is None


def test_cgroup_v1_quota(tmp_path) -> None:
    _write_cgroup(
        tmp_path,
        {"cpu/cpu.cfs_quota_us": "400000\n", "cpu/cpu.cfs_period_us": "100000\n"},
    )
    assert _cgroup_cpu_quota(tmp_path) == 4


def test_cgroup_v1_unlimited_is_none(tmp_path) -> None:
    _write_cgroup(
        tmp_path,
        {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"},
    )
    assert _cgroup_cpu_quota(tmp_path) is None


def test_cgroup_absent_is_none(tmp_path) -> None:
    assert _cgroup_cpu_quota(tmp_path) is None


def test_available_cpu_count_rounds_a_fractional_quota_up() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=96),
        _affinity(96),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=7.65),
    ):
        assert available_cpu_count() == 8


def test_available_cpu_count_takes_min_of_signals() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=128),
        _affinity(64),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=8),
    ):
        assert available_cpu_count() == 8


def test_available_cpu_count_ignores_absent_quota() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=16),
        _affinity(16),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
    ):
        assert available_cpu_count() == 16


def test_available_cpu_count_floors_at_one() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=None),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
    ):
        assert available_cpu_count() >= 1


def test_engine_thread_count_is_none_when_nothing_constrains_the_process() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=16),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
        _affinity(16),
    ):
        assert engine_thread_count() is None


def test_engine_thread_count_rounds_a_fractional_quota_down() -> None:
    # 7 threads out-generate 8 on a 7.65-core quota: the eighth gets the whole
    # lockstep group throttled (measured 63.5 vs 60.3 tok/s).
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=96),
        _affinity(96),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=7.65),
    ):
        assert engine_thread_count() == 7


def test_engine_thread_count_follows_an_affinity_mask() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=96),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
        _affinity(8),
    ):
        assert engine_thread_count() == 8


def test_engine_thread_count_floors_at_one() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=96),
        _affinity(96),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=0.4),
    ):
        assert engine_thread_count() == 1
