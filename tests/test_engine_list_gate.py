"""The engine list names only the missing engines that the host gate lets this host run."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.providers.fleet.host_caps import HostCaps, collect


def _use_host(monkeypatch: pytest.MonkeyPatch, system: str, machine: str) -> HostCaps:
    host = collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvml=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
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


def test_a_host_that_could_not_be_read_lists_no_engine_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_host(monkeypatch, "", "")
    assert engine_service.not_installed() == []


def _install_mlx_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put a complete mlx build in the store, as a moved data folder or a restore does."""
    from lilbee.core.config import cfg
    from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
    from lilbee.providers.fleet.engines.base import EngineName
    from lilbee.providers.fleet.engines.mlx import MLX
    from tests._published import publish
    from tests.providers.fleet.test_mlx_adapter import host as mac_host
    from tests.providers.fleet.test_mlx_adapter import make_pack

    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    mac = mac_host("Darwin", "arm64", "15.2")
    monkeypatch.setattr(engine_service, "session_host", lambda: mac)
    monkeypatch.setattr(runnable, "session_host", lambda: mac)
    sha = make_pack(tmp_path / "pack.whl")
    publish(monkeypatch, EngineName.MLX, sha)
    pin = MLX.pin_record()
    lock = PinLock(
        pin["engine"], pin["version"], pin["backend"], {}, ("MIT",), sha, "mlx-lm-server"
    )
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, tmp_path / "pack.whl")


def test_an_installed_build_the_gate_refuses_is_not_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.providers.fleet.engines.choice import installed_engines

    _install_mlx_build(monkeypatch, tmp_path)
    assert "mlx" in [b.engine for b in engine_service.engine_list().builds]
    _use_host(monkeypatch, "Linux", "x86_64")
    listed = engine_service.engine_list()
    assert "mlx" not in [b.engine for b in listed.builds]
    assert "mlx" not in listed.not_installed
    assert "mlx" not in installed_engines()
    assert "mlx" not in engine_service.engine_names("org/repo#mlx")


def test_an_installed_build_the_gate_passes_stays_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_mlx_build(monkeypatch, tmp_path)
    assert "mlx" in [b.engine for b in engine_service.engine_list().builds]


def test_tei_installed_on_a_windows_host_is_not_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    import stat
    import zipfile

    from lilbee.core.config import cfg
    from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
    from lilbee.providers.fleet.engines.choice import installed_engines
    from lilbee.providers.fleet.engines.tei import TEI

    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    wheel = tmp_path / "tei.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/text-embeddings-router")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: Apache-2.0\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    from lilbee.providers.fleet.engines.base import EngineName
    from tests._published import publish

    _use_host(monkeypatch, "Linux", "x86_64")
    publish(monkeypatch, EngineName.TEI, sha)
    pin = TEI.pin_record()
    lock = PinLock(
        pin["engine"], pin["version"], pin["backend"], {}, ("Apache-2.0",), sha, pin["tool"]
    )
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    assert "tei" in installed_engines()
    assert "tei" in [b.engine for b in engine_service.engine_list().builds]
    _use_host(monkeypatch, "Windows", "AMD64")
    assert "tei" not in installed_engines()
    assert "tei" not in [b.engine for b in engine_service.engine_list().builds]


def test_an_installed_engine_leaves_the_not_installed_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_mlx_build(monkeypatch, tmp_path)
    assert "mlx" not in engine_service.engine_list().not_installed
