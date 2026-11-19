"""The ik_llama.cpp adapter: opt-in, gated by the CPU, composed from the llama.cpp adapter."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from typer.testing import CliRunner

from lilbee.app import engine_choice, engine_service
from lilbee.catalog import runnable
from lilbee.catalog.types import ModelFormat
from lilbee.cli import app as cli_app
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import adapters as fleet_adapters
from lilbee.providers.fleet import engine_manifest, host_caps
from lilbee.providers.fleet.engine_store import (
    NOTICE_RULES,
    EngineStore,
    PinLock,
    UnsafeWheelError,
    engines_root,
    licence_permitted,
)
from lilbee.providers.fleet.engines import choice, ik_llama
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import installed_engines
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.lemonade import LemonadeAdapter
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.host_caps import NO_NVIDIA_DRIVER, collect
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._pack_workflow import pack_workflow
from tests._published import publish
from tests._workflow_steps import commands, steps

COMMIT = "13cffc006b82e03e4d89cbb8c7327f0255ae3d18"
runner = CliRunner()


def _host(system: str, machine: str, flags: list[str] | None) -> Any:
    cpu_info: dict[str, Any] = {} if flags is None else {"flags": flags, "brand_raw": "Test CPU"}
    return host_caps.collect(
        system=system,
        machine=machine,
        cpu_info=cpu_info,
        nvidia=None,
    )


def _make_wheel(path: Path, *, libraries: bool = True) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo("lilbee_engine_ik/bin/llama-server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        if libraries:
            archive.writestr("lilbee_engine_ik/bin/libggml.so", "x")
            archive.writestr("lilbee_engine_ik/bin/libllama.so", "x")
        archive.writestr("lilbee_engine_ik-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "preferred_engine", "auto")
    monkeypatch.setattr(cfg, "engine_choices", None)
    linux = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
    monkeypatch.setattr(runnable, "detect", lambda: linux)


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fixture pack as the pinned build; the pin takes the digest of that pack."""
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    publish(monkeypatch, EngineName.IK_LLAMA, sha)
    lock = PinLock(
        EngineName.IK_LLAMA.value,
        ik_llama.VERSION,
        ik_llama.IkBackend.CPU,
        {},
        ("MIT",),
        sha,
        "llama-server",
    )
    return EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)


def _pin_licences_are_permitted(pin: dict[str, Any]) -> bool:
    """Whether the install screen accepts every licence expression of *pin*."""
    notices = frozenset(rule.file for rule in NOTICE_RULES.values())
    return bool(pin["licenses"]) and all(licence_permitted(e, notices) for e in pin["licenses"])


def test_the_pin_names_the_commit_by_its_first_twelve_digits() -> None:
    pin = IK_LLAMA.pin_record()
    assert ik_llama.IK_LLAMA_COMMIT == COMMIT
    assert pin["version"] == f"0+git{COMMIT[:12]}"
    assert pin["upstream_ref"] == COMMIT
    assert pin["memory"] == {"sizing": ["gguf-parser", "--model", "{model}"], "rule": None}
    assert (pin["engine"], pin["backend"]) == ("ik_llama.cpp", "cpu")
    assert pin["licenses"] == ["MIT AND Apache-2.0 AND MIT-0 AND Unlicense"]
    assert _pin_licences_are_permitted(pin)
    assert choice.pin_lock({**pin, "sha256": "a" * 64}).version == "0+git13cffc006b82"


def test_the_pin_expects_no_shared_library() -> None:
    assert "libraries" not in IK_LLAMA.pin_record()


def test_the_static_server_only_pack_installs_from_the_pin(tmp_path: Path) -> None:
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel, libraries=False)
    lock = choice.pin_lock({**IK_LLAMA.pin_record(), "sha256": sha})
    root = EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    assert lock.libraries == ()
    assert (root / "lilbee_engine_ik" / "bin" / "llama-server").is_file()


def test_the_llama_cpp_pin_refuses_the_one_ggml_library_layout(tmp_path: Path) -> None:
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    lock = choice.pin_lock({**LLAMA.pin_record(), "backend": "cpu", "sha256": sha})
    with pytest.raises(UnsafeWheelError, match="no library file for 'ggml-base'"):
        EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)


def test_the_engine_is_registered_and_does_not_ship_in_the_package() -> None:
    assert choice.ADAPTERS[EngineName.IK_LLAMA] is IK_LLAMA
    assert IK_LLAMA.bundled is False
    assert LLAMA.bundled is True


def test_nothing_is_installed_without_a_store_record() -> None:
    assert EngineName.IK_LLAMA not in installed_engines()
    assert EngineName.LLAMA in installed_engines()
    assert not engines_root(cfg.models_dir).exists()


def test_a_complete_record_makes_the_engine_installed(build: Path) -> None:
    assert installed_engines()[EngineName.IK_LLAMA] == frozenset({ModelFormat.GGUF})


def test_a_record_of_another_wheel_leaves_the_engine_not_installed(
    build: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    publish(monkeypatch, EngineName.IK_LLAMA, "1" * 64)
    with caplog.at_level("WARNING"):
        assert EngineName.IK_LLAMA not in installed_engines()
    assert "different wheel" in caplog.text


def test_binary_is_the_server_of_the_installed_build(build: Path) -> None:
    assert IK_LLAMA.binary() == build / "lilbee_engine_ik" / "bin" / "llama-server"


def test_binary_without_a_build_names_the_install_command() -> None:
    with pytest.raises(ProviderError, match=r"lilbee engine install ik_llama\.cpp"):
        IK_LLAMA.binary()


def test_binary_skips_a_build_of_another_version_and_one_with_no_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EngineStore(engines_root(cfg.models_dir))
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    other = PinLock("ik_llama.cpp", "0+gitaaaaaaaaaaaa", "cpu", {}, ("MIT",), sha, "llama-server")
    store.install_from_wheel(other, wheel)
    current = PinLock("ik_llama.cpp", ik_llama.VERSION, "cpu", {}, ("MIT",), sha, "llama-server")
    for server in store.install_from_wheel(current, wheel).glob("*/bin/llama-server*"):
        server.unlink()
    with pytest.raises(ProviderError, match="is not installed"):
        IK_LLAMA.binary()


@pytest.mark.parametrize(
    ("version", "backend"),
    [("0+gitaaaaaaaaaaaa", ik_llama.IkBackend.CPU), (ik_llama.VERSION, ik_llama.IkBackend.CUDA)],
)
def test_binary_skips_a_build_of_another_version_or_backend_that_has_a_server(
    version: str, backend: str, tmp_path: Path
) -> None:
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    lock = PinLock("ik_llama.cpp", version, backend, {}, ("MIT",), sha, "llama-server")
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    with pytest.raises(ProviderError, match="is not installed"):
        IK_LLAMA.binary()


@pytest.mark.parametrize(
    ("system", "machine", "flags"),
    [("Linux", "x86_64", ["avx2"]), ("Darwin", "arm64", None)],
)
def test_a_cpu_with_avx2_or_neon_passes_the_gate(
    system: str, machine: str, flags: list[str] | None
) -> None:
    gate = IK_LLAMA.host_gate(_host(system, machine, flags))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, True, True)


def test_a_cpu_without_avx2_is_refused_naming_avx2() -> None:
    gate = IK_LLAMA.host_gate(_host("Linux", "x86_64", ["avx", "sse4_2"]))
    assert gate.ok is False
    assert gate.reason == "Test CPU lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON."


def test_an_unreadable_cpu_stays_selectable_but_never_automatic() -> None:
    gate = IK_LLAMA.host_gate(_host("Linux", "x86_64", None))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)
    assert gate.reason == "The CPU flags of this CPU could not be read."


def test_an_unreadable_architecture_stays_selectable_but_never_automatic() -> None:
    gate = IK_LLAMA.host_gate(_host("Plan9", "mips", None))
    assert (gate.ok, gate.known) == (True, False)


def test_install_on_a_host_without_avx2_is_refused_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "detect", lambda: _host("Linux", "x86_64", ["avx"]))
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("ik_llama.cpp")
    assert "AVX2" in str(exc.value)
    assert str(exc.value).endswith(
        "ik_llama.cpp needs a CPU with AVX2 or NEON. llama.cpp stays in use."
    )
    assert not engines_root(cfg.models_dir).exists()


def test_install_with_no_published_pack_says_so_for_a_wheel_and_the_packs_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    wheel = tmp_path / "ik.whl"
    _make_wheel(wheel)
    packs = engines_root(cfg.models_dir) / "packs"
    packs.mkdir(parents=True)
    (packs / f"ik_llama.cpp-{ik_llama.VERSION}-cpu.whl").write_bytes(wheel.read_bytes())
    for given in (wheel, None):
        with pytest.raises(engine_service.EngineRefusedError) as exc:
            engine_service.install("ik_llama.cpp", given)
        assert str(exc.value) == "No pack of ik_llama.cpp is published yet. llama.cpp stays in use."
    assert EngineStore(engines_root(cfg.models_dir)).installed() == []


def test_install_from_a_pack_with_the_pinned_digest_lists_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
    wheel = tmp_path / "ik.whl"
    publish(monkeypatch, EngineName.IK_LLAMA, _make_wheel(wheel))
    built = engine_service.install("ik_llama.cpp", wheel)
    assert (built.engine, built.version, built.backend) == (
        "ik_llama.cpp",
        f"0+git{COMMIT[:12]}",
        "cpu",
    )
    assert EngineName.IK_LLAMA in installed_engines()


def test_install_of_a_wheel_other_than_the_published_pack_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
    wheel = tmp_path / "ik.whl"
    _make_wheel(wheel)
    publish(monkeypatch, EngineName.IK_LLAMA, "0" * 64)
    with pytest.raises(engine_service.EngineRefusedError, match="does not match its pin"):
        engine_service.install("ik_llama.cpp", wheel)
    assert EngineStore(engines_root(cfg.models_dir)).installed() == []


def test_install_on_a_system_below_the_published_floor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
    wheel = tmp_path / "ik.whl"
    publish(monkeypatch, EngineName.IK_LLAMA, _make_wheel(wheel), min_os="2.40", host_os="2.31")
    with pytest.raises(engine_service.EngineRefusedError, match=r"needs glibc 2\.40 or later"):
        engine_service.install("ik_llama.cpp", wheel)
    assert EngineStore(engines_root(cfg.models_dir)).installed() == []


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_auto_never_picks_the_installed_engine(system: str, build: Path, monkeypatch) -> None:
    machine = "arm64" if system == "darwin" else "x86_64"
    installed = installed_engines()
    assert EngineName.IK_LLAMA in installed
    monkeypatch.setattr(
        runnable,
        "session_host",
        lambda: collect(
            system=system.capitalize(), machine=machine, cpu_info={"flags": ["avx2"]}, nvidia=None
        ),
    )
    monkeypatch.setattr(engine_choice, "installed_engines", lambda: installed)
    got = engine_choice.resolve_engines(cfg).choices[WorkerRole.CHAT]
    assert got.name is not EngineName.IK_LLAMA
    assert got.name is not None
    assert got.notice_kind is None


def test_the_installed_engine_is_chosen_when_the_user_names_it(build: Path, monkeypatch) -> None:
    monkeypatch.setattr(cfg, "preferred_engine", "ik_llama.cpp")
    got = engine_choice.resolve_engines(cfg).choices[WorkerRole.CHAT]
    assert got.name is EngineName.IK_LLAMA


def test_a_named_engine_that_is_not_installed_falls_back_with_the_notice(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "preferred_engine", "ik_llama.cpp")
    got = engine_choice.resolve_engines(cfg).choices[WorkerRole.CHAT]
    assert got.name is EngineName.LLAMA
    assert got.notice is not None
    assert "Engine ik_llama.cpp for chat is not installed. Using llama.cpp." in got.notice


def test_engine_list_shows_the_engine_as_not_installed_and_writes_nothing() -> None:
    result = runner.invoke(cli_app, ["engine", "list"])
    assert result.exit_code == 0
    assert "ik_llama.cpp not installed" in result.output
    assert not engines_root(cfg.models_dir).exists()


def test_engine_list_json_names_the_engines_that_are_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setattr(LemonadeAdapter, "user_binary", lambda self: None)
    monkeypatch.setattr("lilbee.app.engine_service.llama_server_ready", lambda: False)
    result = runner.invoke(cli_app, ["--json", "engine", "list"])
    body = json.loads(result.output)
    assert body["builds"] == []
    assert body["not_installed"] == [
        "llama.cpp",
        "ik_llama.cpp",
        "mistral.rs",
        "tei",
        "openvino",
        "lemonade",
    ]


def test_engine_list_shows_an_installed_build_and_no_not_installed_line(build: Path) -> None:
    result = runner.invoke(cli_app, ["engine", "list"])
    assert f"ik_llama.cpp 0+git{COMMIT[:12]} (cpu)" in result.output
    assert "ik_llama.cpp not installed" not in result.output
    assert "tei not installed" in result.output


def test_engine_roles_shows_llama_cpp_with_the_engine_off() -> None:
    result = runner.invoke(cli_app, ["engine", "roles"])
    assert result.exit_code == 0
    assert "llama.cpp" in result.output
    assert "ik_llama.cpp" not in result.output


def _request() -> LaunchRequest:
    return LaunchRequest(
        binary=Path("/b/server"),
        model_path=Path("/m/x.gguf"),
        devices=(0, 1),
        device_names=("CUDA0", "CUDA1"),
        tensor_split=(1, 1),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=4096,
        flash_attn="on",
    )


_HELP_NONE = "usage: server [options]\n  -m, --model FNAME     model path\n"
_HELP_BOTH = (
    _HELP_NONE
    + "  -fa, --flash-attn [on|off|auto]  attention\n"
    + "  -dev, --device <dev1,dev2,..>  devices\n"
    + "  -ts, --tensor-split N0,N1,N2,...  split\n"
)


@pytest.mark.parametrize("adapter", [LLAMA, IK_LLAMA], ids=["llama.cpp", "ik_llama.cpp"])
def test_placement_flags_are_dropped_when_the_help_omits_them(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = adapter.role_spec(WorkerRole.CHAT, None, None)
    monkeypatch.setattr(fleet_adapters, "engine_help", lambda _b: _HELP_NONE)
    argv = adapter.launch_argv(spec, _request())
    assert "--model" in argv
    assert "--device" not in argv
    assert "--tensor-split" not in argv
    monkeypatch.setattr(fleet_adapters, "engine_help", lambda _b: _HELP_BOTH)
    full = adapter.launch_argv(spec, _request())
    assert full[full.index("--device") + 1] == "CUDA0,CUDA1"
    assert full[full.index("--tensor-split") + 1] == "1,1"


def test_the_argv_equals_the_llama_cpp_argv_for_every_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fleet_adapters, "engine_help", lambda _b: _HELP_BOTH)
    for role in IK_LLAMA.served_roles:
        spec = IK_LLAMA.role_spec(role, None, None)
        assert IK_LLAMA.launch_argv(spec, _request()) == LLAMA.launch_argv(
            LLAMA.role_spec(role, None, None), _request()
        )


def test_the_request_profile_differs_from_llama_cpp_in_its_wording_only() -> None:
    mine = IK_LLAMA.request_profile(WorkerRole.CHAT)
    theirs = LLAMA.request_profile(WorkerRole.CHAT)
    assert mine.label == "ik_llama.cpp server"
    differing = {
        field
        for field in mine.__dataclass_fields__
        if getattr(mine, field) != getattr(theirs, field)
    }
    assert differing == {"label", "unreadable_embedding_hint"}


def test_the_engine_serves_chat_and_embed_and_loads_gguf() -> None:
    assert IK_LLAMA.served_roles == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert IK_LLAMA.formats == frozenset({ModelFormat.GGUF})
    assert IK_LLAMA.supports_architecture("llama") == LLAMA.supports_architecture("llama")


def test_the_adapter_delegates_what_the_engine_shares_with_llama_cpp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def spy(name: str, result: Any = None):
        def inner(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(name)
            return result

        return inner

    for method, result in [
        ("identity", "id"),
        ("discover_devices", None),
        ("model_meta", None),
        ("assert_loadable", None),
        ("runtime_env", {}),
        ("target_env", {}),
        ("estimate_memory", None),
        ("check_readback", False),
        ("supports_fit_target", False),
        ("supports_shared_kv", False),
        ("swap_extras", {}),
    ]:
        monkeypatch.setattr(type(LLAMA), method, spy(method, result))
    path = Path("/b/server")
    IK_LLAMA.identity(path)
    IK_LLAMA.discover_devices(path)
    IK_LLAMA.model_meta(path)
    IK_LLAMA.assert_loadable("repo", "file", None)
    IK_LLAMA.runtime_env()
    IK_LLAMA.target_env(RoleTarget.CPU, 0)
    IK_LLAMA.estimate_memory(
        MemoryRequest(
            model_path=path,
            ctx=16,
            slots=1,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=cfg.kv_cache_type,
            on_cpu=True,
        )
    )
    IK_LLAMA.check_readback(path)
    IK_LLAMA.supports_fit_target(path)
    IK_LLAMA.supports_shared_kv(path)
    IK_LLAMA.swap_extras(None)  # type: ignore[arg-type]
    assert calls == [
        "identity",
        "discover_devices",
        "model_meta",
        "assert_loadable",
        "runtime_env",
        "target_env",
        "estimate_memory",
        "check_readback",
        "supports_fit_target",
        "supports_shared_kv",
        "swap_extras",
    ]


def _nvidia_host(
    system: str = "Linux",
    machine: str = "x86_64",
    *,
    cards: tuple[tuple[int, int], ...] = ((8, 9),),
    cuda: int | None = 12040,
) -> Any:
    facts = {
        "driver": "550.135",
        "cuda_driver": cuda,
        "gpus": [{"name": f"card{i}", "cc": list(cc)} for i, cc in enumerate(cards)],
    }
    return collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvidia=facts)


@pytest.mark.parametrize(
    "cards",
    [((7, 5),), ((8, 9),), ((12, 0),), ((6, 1), (8, 6))],
    ids=["turing", "ada", "blackwell", "one-card-served"],
)
def test_a_linux_host_with_a_turing_or_newer_card_and_the_driver_gets_the_cuda_build(
    cards: tuple[tuple[int, int], ...],
) -> None:
    pick = IK_LLAMA.pick_build(_nvidia_host(cards=cards))
    assert pick.backend == "cuda"
    assert pick.reason == ""


@pytest.mark.parametrize(
    ("host", "reason"),
    [
        pytest.param(
            _nvidia_host(cards=((7, 0),)),
            "The NVIDIA GPU card0 has compute capability 7.0, "
            "and the CUDA build needs 7.5 or newer.",
            id="volta",
        ),
        pytest.param(
            _nvidia_host(cards=((6, 1),)),
            "The NVIDIA GPU card0 has compute capability 6.1, "
            "and the CUDA build needs 7.5 or newer.",
            id="pascal",
        ),
        pytest.param(
            _nvidia_host(cards=((6, 1), (7, 0))),
            "The NVIDIA GPU card1 has compute capability 7.0, "
            "and the CUDA build needs 7.5 or newer.",
            id="best-of-two-old-cards",
        ),
        pytest.param(
            _nvidia_host(cuda=12030),
            "The NVIDIA driver offers CUDA 12.3, and the CUDA build needs CUDA 12.4. "
            "Update the driver.",
            id="driver",
        ),
        pytest.param(
            _nvidia_host(cards=((7, 5),), cuda=12030),
            "The NVIDIA driver offers CUDA 12.3, and the CUDA build needs CUDA 12.4. "
            "Update the driver.",
            id="driver-with-the-oldest-served-card",
        ),
        pytest.param(
            _nvidia_host(cuda=None),
            "The NVIDIA driver offers CUDA 0.0, and the CUDA build needs CUDA 12.4. "
            "Update the driver.",
            id="driver-unreadable",
        ),
        pytest.param(_nvidia_host(cards=()), "No NVIDIA GPU found.", id="no-card"),
        pytest.param(
            collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None),
            NO_NVIDIA_DRIVER,
            id="no-driver",
        ),
        pytest.param(
            _nvidia_host("Linux", "aarch64"),
            "No CUDA build exists for this host.",
            id="linux-arm64",
        ),
        pytest.param(
            _nvidia_host("Windows", "AMD64"), "No CUDA build exists for this host.", id="windows"
        ),
        pytest.param(
            _nvidia_host("Darwin", "arm64"), "No CUDA build exists for this host.", id="macos"
        ),
    ],
)
def test_a_host_the_cuda_build_cannot_serve_gets_the_cpu_build_with_the_reason(
    host: Any, reason: str
) -> None:
    pick = IK_LLAMA.pick_build(host)
    assert pick.backend == "cpu"
    assert pick.reason == reason


def test_the_pin_of_a_cuda_host_names_the_cuda_cell_and_its_licences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _nvidia_host())
    cpu = engine_manifest.MANIFESTS["ik_llama.cpp"][ik_llama.VERSION]["linux-x86_64-cpu"]
    table = {"ik_llama.cpp": {ik_llama.VERSION: {"linux-x86_64-cuda": {**cpu, "sha256": "c" * 64}}}}
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)
    pin = IK_LLAMA.pin_record()
    assert pin["backend"] == "cuda"
    assert pin["sha256"] == "c" * 64
    assert list(pin["platforms"]) == ["linux-x86_64-cuda"]
    assert pin["licenses"] == [
        "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND MIT-0 AND Unlicense"
    ]
    assert _pin_licences_are_permitted(pin)


def test_binary_on_a_cuda_host_is_the_cuda_build_and_not_the_cpu_one(
    build: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _nvidia_host())
    with pytest.raises(ProviderError, match="is not installed"):
        IK_LLAMA.binary()
    wheel = tmp_path / "ik-cuda.whl"
    sha = _make_wheel(wheel)
    lock = PinLock("ik_llama.cpp", ik_llama.VERSION, "cuda", {}, ("MIT",), sha, "llama-server")
    cuda = EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    assert IK_LLAMA.binary() == cuda / "lilbee_engine_ik" / "bin" / "llama-server"


_BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "tools/wheel-build/ik_llama_manylinux.sh"


def test_the_driver_floor_is_the_cuda_version_the_build_script_installs() -> None:
    text = _BUILD_SCRIPT.read_text(encoding="utf-8")
    dash = re.search(r"^cuda_dash=(\d+)-(\d+)$", text, re.MULTILINE)
    assert dash is not None
    assert (int(dash[1]), int(dash[2])) == ik_llama._CUDA_DRIVER_FLOOR
    assert f"/usr/local/cuda-{dash[1]}.{dash[2]}" in text
    assert re.search(rf"^cuda_full={dash[1]}\.{dash[2]}\.\d+$", text, re.MULTILINE)


def test_the_vendored_apache_text_is_the_canonical_licence() -> None:
    """The file is the text at apache.org/licenses/LICENSE-2.0.txt, which section 4(a) requires."""
    text = (_BUILD_SCRIPT.parent / "LICENSE-Apache-2.0").read_bytes()
    assert hashlib.sha256(text).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def _active_script() -> str:
    """The build script without its comment lines."""
    lines = _BUILD_SCRIPT.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _cuda_blocks() -> list[str]:
    """The active lines of each multi-line `if` block that runs for the CUDA backend only."""
    pattern = r'^if \[ "\$\{BACKEND\}" = cuda \]; then\n(.*?)^fi$'
    return re.findall(pattern, _active_script(), re.MULTILINE | re.DOTALL)


def _cuda_arch_set() -> list[str]:
    """The architecture set in the `else` branch of the CPU test, which is the CUDA build."""
    pattern = r'^if \[ "\$\{BACKEND\}" = cpu \]; then\n.*?^else\n(.*?)^fi$'
    found = re.search(pattern, _active_script(), re.MULTILINE | re.DOTALL)
    assert found is not None, "the build script has no active CUDA branch"
    listed = re.findall(r'"-DCMAKE_CUDA_ARCHITECTURES=([\d;a-z-]+)"', found[1])
    assert len(listed) == 1
    return listed[0].split(";")


def test_the_build_script_writes_the_nvidia_notice_from_the_libraries_it_copies() -> None:
    blocks = [block for block in _cuda_blocks() if "write_cuda_notice.py" in block]
    assert len(blocks) == 1, "the notice call is not inside an active CUDA block"
    call = re.search(r"^\s+python .*/write_cuda_notice\.py\" (.*)$", blocks[0], re.MULTILINE)
    assert call is not None
    assert call[1] == '"${NOTICE_FILE}" "${cuda_full}" "${SHA_FILE}"'
    assert (_BUILD_SCRIPT.parent / "write_cuda_notice.py").is_file()


def test_the_compute_floor_is_the_lowest_architecture_the_build_script_compiles() -> None:
    text = _active_script()
    lowest = min(int(arch.split("-")[0]) for arch in _cuda_arch_set())
    assert (lowest // 10, lowest % 10) == ik_llama._CUDA_MIN_COMPUTE
    assert "-DGGML_NATIVE=OFF" in text
    assert "-DGGML_NCCL=OFF" in text


def test_the_build_script_holds_a_virtual_architecture_at_or_above_hopper() -> None:
    """A card above the real architectures runs the PTX of the highest virtual one."""
    archs = _cuda_arch_set()
    virtual = [int(a.removesuffix("-virtual")) for a in archs if a.endswith("-virtual")]
    real = [int(a.removesuffix("-real")) for a in archs if a.endswith("-real")]
    assert real
    assert max(virtual) >= 90
    assert max(virtual) >= max(real)


# The ik job of the packs repository workflow builds the wheels that the shipped table names.

_IK_IMAGE = re.compile(r"manylinux_2_34_x86_64@sha256:[0-9a-f]{64}")
_CHECK_STEP = "Check the wheel and start the server from it"
_THIRD = "licenses/THIRD-PARTY/ik_llama.cpp/"
_GRANT = "Permission is hereby granted, free of charge, to any person obtaining a copy"
_DIST = "lilbee_engine_ik_llama_cpp-0+git13cffc006b82.dist-info/"


def _workflow_job(name: str) -> dict[str, Any]:
    parsed = yaml.safe_load(pack_workflow().read_text(encoding="utf-8"))
    return cast(dict[str, Any], parsed["jobs"][name])


def _ik_step(name: str) -> str:
    """The shell of the one step of the ik job with this name."""
    (run,) = (s["run"] for s in steps(pack_workflow(), "ik") if s.get("name") == name)
    return cast(str, run)


def _ik_cells() -> dict[str, dict[str, Any]]:
    matrix = _workflow_job("ik")["strategy"]["matrix"]["include"]
    return {cell["cell"]: cell for cell in matrix}


def test_the_ik_job_builds_the_commit_the_adapter_pins() -> None:
    env = _workflow_job("ik")["env"]
    assert env["UPSTREAM_REPO"] == "ikawrakow/ik_llama.cpp"
    assert env["UPSTREAM_COMMIT"] == ik_llama.IK_LLAMA_COMMIT
    assert "0+git${UPSTREAM_COMMIT:0:12}" in commands(_ik_step("Build the wheel"))
    assert f"0+git{env['UPSTREAM_COMMIT'][:12]}" == ik_llama.VERSION


def test_the_ik_job_has_one_cell_for_each_build_of_the_pack() -> None:
    cells = _ik_cells()
    assert set(cells) == {"linux-x86_64-cpu", "linux-x86_64-cuda"}
    for name, cell in cells.items():
        assert cell["backend"] == name.rsplit("-", 1)[1]
        assert cell["backend"] in {backend.value for backend in ik_llama.IkBackend}
        licences = ik_llama._CUDA_LICENCES if cell["backend"] == "cuda" else ik_llama._CPU_LICENCES
        pinned = licences[0].split(" AND ")
        stated = cell["license"].split(" AND ")
        assert pinned[: len(stated)] == stated


def test_a_cell_never_builds_a_lower_number_than_its_shipped_wheel() -> None:
    """A rebuild with new content takes a new build number, so one file name holds one digest."""
    shipped = engine_manifest.MANIFESTS["ik_llama.cpp"][ik_llama.VERSION]
    for name, cell in _ik_cells().items():
        found = re.search(r"-(\d+)\.(\w+)-py3", shipped[name]["url"])
        assert found is not None
        assert found[2] == cell["backend"]
        assert cell["build_number"] >= int(found[1])


def test_the_ik_job_runs_for_its_own_engine_only_and_its_wheels_are_published() -> None:
    workflow = yaml.safe_load(pack_workflow().read_text(encoding="utf-8"))
    assert "ik" in workflow[True]["workflow_dispatch"]["inputs"]["engine"]["options"]
    assert _workflow_job("ik")["if"] == "needs.plan.outputs.engine == 'ik'"
    assert "ik" in _workflow_job("publish")["needs"]
    (download,) = (
        s for s in steps(pack_workflow(), "publish") if s.get("uses", "").startswith("actions/down")
    )
    (upload,) = (
        s for s in steps(pack_workflow(), "ik") if s.get("uses", "").startswith("actions/upload")
    )
    assert fnmatch.fnmatch(upload["with"]["name"].replace("${{ matrix.cell }}", "x"), "*-pack-*")
    assert download["with"]["pattern"] == "*-pack-*"


def test_the_ik_job_calls_the_build_script_in_the_image_of_the_llama_cell() -> None:
    run = commands(_ik_step("Build ik_llama.cpp in the manylinux_2_34 image"))
    assert "bash /work/tools/wheel-build/ik_llama_manylinux.sh" in run
    llama = commands(
        next(
            s["run"]
            for s in steps(pack_workflow(), "cell")
            if s.get("name", "").startswith("Build llama.cpp with CUDA")
        )
    )
    assert set(_IK_IMAGE.findall(run)) == set(_IK_IMAGE.findall(llama)) != set()
    required = set(re.findall(r"\$\{(\w+):\?\}", _active_script()))
    assert {"UPSTREAM_COMMIT", "SRC_DIR", "SHA_FILE", "NOTICE_FILE"} <= required
    for name in required:
        assert re.search(rf"-e {name}(?:=|\s|$)", run), name


def test_the_ik_licence_step_ships_both_pages_and_the_sqlite_headers() -> None:
    run = commands(_ik_step("Collect the licences of the server and the code it carries"))
    ships = re.findall(r"--ship (\S+)", run)
    assert ships == [
        "examples/server/public/index.html.gz",
        "examples/server/public_llamacpp/index_llamacpp.html.gz",
        "examples/server/sqlite_modern_cpp",
    ]
    assert "--skip" not in run
    assert " ik-src " in run[run.rindex("--ship") :]


def test_the_ik_job_reads_the_source_epoch_before_it_builds_the_wheel() -> None:
    runs = [commands(step.get("run", "")) for step in steps(pack_workflow(), "ik")]
    (epoch,) = (i for i, run in enumerate(runs) if "source_date_epoch.sh" in run)
    (build,) = (i for i, run in enumerate(runs) if "python -m build --wheel" in run)
    assert epoch < build
    assert "bash tools/wheel-build/source_date_epoch.sh ik-src >> $GITHUB_ENV" in runs[epoch]


def test_the_ik_check_runs_the_server_and_the_cuda_byte_check_from_the_installed_wheel() -> None:
    run = commands(_ik_step(_CHECK_STEP))
    assert "pip install -q out/*.whl" in run
    adapter = Path(ik_llama.__file__).relative_to(_BUILD_SCRIPT.parents[2]).as_posix()
    assert f"python - out/*.whl ${{LICENSE_EXPRESSION}} {adapter}" in run
    assert "sha256sum -c" in run and "cuda-libs.sha256" in run
    assert "check_glibc_floor.sh" in run
    assert run.rstrip().endswith("cd / ${exe} --version")


def _wheel_check() -> str:
    """The program that the check step runs against the wheel."""
    found = re.search(r"<<'PY'\n(.*?)\nPY\n", _ik_step(_CHECK_STEP), re.DOTALL)
    assert found is not None
    return found[1]


def _notice_entries(backend: str) -> dict[str, str]:
    """A wheel that holds every notice the check wants, each with the text it wants."""
    apache = (_BUILD_SCRIPT.parent / "LICENSE-Apache-2.0").read_text(encoding="utf-8")
    entries = {
        "licenses/LICENSE-ik_llama.cpp": f"MIT License\n\n{_GRANT}",
        _THIRD + "LICENSE-Apache-2.0": apache,
        _THIRD + "examples/server/public/index.html.gz.NOTICE": f"@license React\n{_GRANT}",
        _THIRD + "examples/server/public_llamacpp/index_llamacpp.html.gz.NOTICE": (
            "Copyright 2024 Mozilla Foundation\nApache License, Version 2.0"
        ),
        _THIRD + "vendor/nlohmann/json.hpp.NOTICE": f"Niels Lohmann\n{_GRANT}",
        _THIRD + "vendor/cpp-httplib/httplib.h.NOTICE": f"Yuji Hirose\n{_GRANT}",
        _THIRD + "vendor/stb/stb_image_resize2.h.NOTICE": (
            f"Sean Barrett\nALTERNATIVE A - MIT License\n{_GRANT}"
        ),
    }
    if backend == "cuda":
        eula = "Attachment A: libcudart libcublas libcublasLt\n" + "EULA text. " * 3000
        entries["licenses/NOTICE-NVIDIA-CUDA"] = eula
    return entries


def _run_wheel_check(
    tmp_path: Path,
    backend: str,
    *,
    drop: str | None = None,
    text: dict[str, str] | None = None,
    built: str | None = None,
    metadata: str | None = None,
    pin_commit: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the check on a fake wheel: *built* is the job's expression, *metadata* the wheel's."""
    pinned = (ik_llama._CUDA_LICENCES if backend == "cuda" else ik_llama._CPU_LICENCES)[0]
    entries = _notice_entries(backend)
    entries.update(text or {})
    wheel = tmp_path / "ik.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, body in entries.items():
            if drop is None or not name.endswith(drop):
                archive.writestr(_DIST + name, body)
        archive.writestr(_DIST + "METADATA", f"License-Expression: {metadata or pinned}\n")
    adapter = Path(ik_llama.__file__)
    if pin_commit is not None:
        adapter = tmp_path / "ik_llama.py"
        source = Path(ik_llama.__file__).read_text(encoding="utf-8")
        assert ik_llama.IK_LLAMA_COMMIT in source
        adapter.write_text(source.replace(ik_llama.IK_LLAMA_COMMIT, pin_commit), encoding="utf-8")
    env = {**os.environ, "BACKEND": backend, "UPSTREAM_COMMIT": ik_llama.IK_LLAMA_COMMIT}
    argv = [str(wheel), built or pinned, str(adapter)]
    return subprocess.run(
        [sys.executable, "-I", "-c", _wheel_check(), *argv],
        capture_output=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_the_wheel_check_accepts_a_wheel_with_every_notice(tmp_path: Path, backend: str) -> None:
    done = _run_wheel_check(tmp_path, backend)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


_REQUIRED_NOTICES = [
    "licenses/LICENSE-ik_llama.cpp",
    "index.html.gz.NOTICE",
    "index_llamacpp.html.gz.NOTICE",
    "vendor/nlohmann/json.hpp.NOTICE",
    "vendor/cpp-httplib/httplib.h.NOTICE",
    "vendor/stb/stb_image_resize2.h.NOTICE",
]


@pytest.mark.parametrize("notice", _REQUIRED_NOTICES)
def test_the_wheel_check_refuses_a_wheel_without_a_notice(tmp_path: Path, notice: str) -> None:
    done = _run_wheel_check(tmp_path, "cpu", drop=notice)
    assert done.returncode != 0
    assert "the wheel holds 0 entries that end with " in done.stderr
    assert notice in done.stderr


@pytest.mark.parametrize("notice", [n for n in _REQUIRED_NOTICES if n.endswith("NOTICE")])
def test_the_wheel_check_refuses_a_notice_that_holds_no_licence_text(
    tmp_path: Path, notice: str
) -> None:
    name = next(n for n in _notice_entries("cpu") if n.endswith(notice))
    done = _run_wheel_check(tmp_path, "cpu", text={name: "see the upstream project\n"})
    assert done.returncode != 0
    assert f"{notice} lacks the text" in done.stderr


def test_the_wheel_check_refuses_a_cuda_notice_without_the_eula(tmp_path: Path) -> None:
    done = _run_wheel_check(tmp_path, "cuda", text={"licenses/NOTICE-NVIDIA-CUDA": "NVIDIA\n"})
    assert done.returncode != 0
    assert "NOTICE-NVIDIA-CUDA lacks the text: Attachment A" in done.stderr
    assert "NOTICE-NVIDIA-CUDA is shorter than the EULA" in done.stderr
    assert _run_wheel_check(tmp_path, "cuda", drop="NOTICE-NVIDIA-CUDA").returncode != 0


def test_the_wheel_check_wants_the_apache_text_once(tmp_path: Path) -> None:
    missing = _run_wheel_check(tmp_path, "cpu", drop="LICENSE-Apache-2.0")
    assert missing.returncode != 0
    assert "the Apache-2.0 text is in 0 licence files" in missing.stderr
    copy = _notice_entries("cpu")[_THIRD + "LICENSE-Apache-2.0"]
    twice = _run_wheel_check(tmp_path, "cpu", text={"licenses/COPY-Apache": copy})
    assert twice.returncode != 0
    assert "the Apache-2.0 text is in 2 licence files" in twice.stderr


def test_the_wheel_check_refuses_an_expression_that_the_pin_does_not_name(tmp_path: Path) -> None:
    stated = "MIT AND Apache-2.0"
    done = _run_wheel_check(tmp_path, "cpu", built=stated, metadata=stated)
    assert done.returncode != 0
    refusal = f"the lilbee pin names {ik_llama._CPU_LICENCES}, the wheel states {stated}"
    assert refusal in done.stderr
    assert "METADATA lacks" not in done.stderr


def test_the_wheel_check_refuses_a_wheel_whose_metadata_differs_from_the_build(
    tmp_path: Path,
) -> None:
    done = _run_wheel_check(tmp_path, "cpu", metadata="MIT")
    assert done.returncode != 0
    assert f"METADATA lacks License-Expression: {ik_llama._CPU_LICENCES[0]}" in done.stderr


def test_the_wheel_check_refuses_a_pin_for_another_commit(tmp_path: Path) -> None:
    done = _run_wheel_check(tmp_path, "cpu", pin_commit="0" * 40)
    assert done.returncode != 0
    assert "another ik_llama.cpp commit" in done.stderr
