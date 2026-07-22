"""Tests for ``lilbee.runtime.cpu`` sizing helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from lilbee.runtime.cpu import _cgroup_cpu_quota, available_cpu_count, cpu_quota


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


def _write_cgroup(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def test_cgroup_v2_quota_ceils_to_cores(tmp_path) -> None:
    _write_cgroup(tmp_path, {"cpu.max": "250000 100000\n"})
    assert _cgroup_cpu_quota(tmp_path) == 3  # ceil(2.5)


def test_cgroup_v2_unlimited_is_none(tmp_path) -> None:
    _write_cgroup(tmp_path, {"cpu.max": "max 100000\n"})
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


def test_available_cpu_count_takes_min_of_signals() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=128),
        mock.patch("lilbee.runtime.cpu.os.sched_getaffinity", return_value=set(range(64)), create=True),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=8),
    ):
        assert available_cpu_count() == 8


def test_available_cpu_count_ignores_absent_quota() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=16),
        mock.patch("lilbee.runtime.cpu.os.sched_getaffinity", return_value=set(range(16)), create=True),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
    ):
        assert available_cpu_count() == 16


def test_available_cpu_count_floors_at_one() -> None:
    with (
        mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=None),
        mock.patch("lilbee.runtime.cpu._cgroup_cpu_quota", return_value=None),
    ):
        assert available_cpu_count() >= 1
