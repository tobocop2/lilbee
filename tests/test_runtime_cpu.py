"""Tests for ``lilbee.runtime.cpu.cpu_quota``."""

from __future__ import annotations

from unittest import mock

from lilbee.runtime.cpu import cpu_quota


def test_default_is_half_of_cpu_count() -> None:
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=8):
        assert cpu_quota() == 4


def test_default_floors_at_one() -> None:
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=1):
        assert cpu_quota() == 1


def test_default_handles_unknown_cpu_count() -> None:
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=None):
        assert cpu_quota() == 1


def test_env_override_accepts_positive_int(monkeypatch) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "12")
    assert cpu_quota() == 12


def test_env_override_rejects_zero(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "0")
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)


def test_env_override_rejects_negative(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "-3")
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)


def test_env_override_rejects_non_integer(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LILBEE_CPU_QUOTA", "garbage")
    with mock.patch("lilbee.runtime.cpu.os.cpu_count", return_value=8):
        assert cpu_quota() == 4
    assert any("LILBEE_CPU_QUOTA" in rec.message for rec in caplog.records)
