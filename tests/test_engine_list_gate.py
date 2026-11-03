"""The engine list names only the missing engines that the host gate lets this host run."""

from __future__ import annotations

import re
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


def _installed_ik_llama(monkeypatch: pytest.MonkeyPatch, *, avx2: bool) -> None:
    """An installed ik_llama.cpp build on a Linux x86_64 CPU with or without AVX2."""
    from lilbee.providers.fleet.engines.base import EngineName
    from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA

    flags = ["avx2"] if avx2 else ["sse4_2"]
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": flags}, nvml=None)
    monkeypatch.setattr(engine_service, "session_host", lambda: host)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName.IK_LLAMA: IK_LLAMA.formats}
    )


def test_the_picker_names_an_installed_engine_the_cpu_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_ik_llama(monkeypatch, avx2=True)
    assert "ik_llama.cpp" in engine_service.engine_names("org/repo.gguf")


def test_the_picker_hides_an_installed_engine_the_cpu_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_ik_llama(monkeypatch, avx2=False)
    assert "ik_llama.cpp" not in engine_service.engine_names("org/repo.gguf")


def _refuse(choose: object, model: str, name: str) -> engine_service.EngineRefusedError:
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        if choose is None:
            engine_service.use(model, name)
        else:
            engine_service.prefer(name)
    return raised.value


@pytest.mark.parametrize("choose", [engine_service.prefer, None], ids=["prefer", "use"])
def test_an_engine_the_cpu_cannot_run_cannot_be_chosen(
    monkeypatch: pytest.MonkeyPatch, choose: object
) -> None:
    _installed_ik_llama(monkeypatch, avx2=False)
    message = str(_refuse(choose, "org/repo.gguf", "ik_llama.cpp"))
    assert "is not installed" not in message
    assert message.startswith("Engine ik_llama.cpp is installed, but this computer cannot run it.")
    assert "This CPU lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON." in message
    assert not re.search(r"\. (?!mlx|ik_llama)[a-z]", message)


@pytest.mark.parametrize("choose", [engine_service.prefer, None], ids=["prefer", "use"])
@pytest.mark.parametrize(
    ("engine", "system", "machine", "model", "reason"),
    [
        ("mlx", "Linux", "x86_64", "org/repo#mlx", "not offered on linux x86_64"),
        ("tei", "Windows", "AMD64", "org/repo#safetensors", "No build exists for Windows"),
    ],
    ids=["mlx-on-linux", "tei-on-windows"],
)
def test_an_installed_engine_the_host_refuses_cannot_be_chosen(
    monkeypatch: pytest.MonkeyPatch,
    choose: object,
    engine: str,
    system: str,
    machine: str,
    model: str,
    reason: str,
) -> None:
    from lilbee.providers.fleet.engines.base import EngineName

    _use_host(monkeypatch, system, machine)
    adapter = engine_service.choice.ADAPTERS[EngineName(engine)]
    monkeypatch.setattr(
        engine_service, "installed_engines", lambda: {EngineName(engine): adapter.formats}
    )
    assert engine not in engine_service.offered_engines()
    message = str(_refuse(choose, model, engine))
    assert message.startswith(f"Engine {engine} is installed, but this computer cannot run it.")
    assert reason in message
    assert "is not installed" not in message
    assert not re.search(r"\. (?!mlx|ik_llama)[a-z]", message)


def test_an_engine_with_no_build_that_the_host_passes_points_to_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_host(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(engine_service, "installed_engines", lambda: {})
    assert "tei" in engine_service.not_installed()
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.prefer("tei")
    assert str(raised.value) == (
        "Engine tei is not installed. Run lilbee engine list to see the installed engines."
    )


@pytest.mark.parametrize("choose", [engine_service.prefer, None], ids=["prefer", "use"])
def test_a_host_refused_engine_with_no_build_gives_the_reason_not_the_list(
    monkeypatch: pytest.MonkeyPatch, choose: object
) -> None:
    _use_host(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(engine_service, "installed_engines", lambda: {})
    assert "mlx" not in engine_service.not_installed()
    message = str(_refuse(choose, "org/repo#mlx", "mlx"))
    assert message.startswith("Engine mlx is not installed, and this computer cannot run it.")
    assert "not offered on linux x86_64" in message
    assert "engine list" not in message
    assert not re.search(r"\. (?!mlx|ik_llama)[a-z]", message)
