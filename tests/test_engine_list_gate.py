"""The engine list names only the missing engines that the host gate lets this host run."""

from __future__ import annotations

import pytest

from lilbee.app import engine_service
from lilbee.providers.fleet.host_caps import HostCaps, collect


def _use_host(monkeypatch: pytest.MonkeyPatch, system: str, machine: str) -> HostCaps:
    host = collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvml=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    return host


def test_an_engine_the_gate_refuses_is_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_host(monkeypatch, "Windows", "AMD64")
    listed = engine_service.not_installed()
    assert "tei" not in listed
    assert "mistral.rs" in listed


def test_the_list_follows_the_host_not_the_adapter_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_host(monkeypatch, "Linux", "aarch64")
    listed = engine_service.not_installed()
    assert "tei" not in listed
    assert "mistral.rs" not in listed


def test_an_engine_the_gate_passes_is_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_host(monkeypatch, "Linux", "x86_64")
    assert {"tei", "mistral.rs"} <= set(engine_service.not_installed())


def test_a_host_that_could_not_be_read_lists_every_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_host(monkeypatch, "", "")
    assert {"tei", "mistral.rs"} <= set(engine_service.not_installed())
