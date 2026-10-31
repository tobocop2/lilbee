"""The ik_llama.cpp adapter: opt-in, gated by the CPU, composed from the llama.cpp adapter."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lilbee.app import engine_choice, engine_service
from lilbee.catalog.types import ModelFormat
from lilbee.cli import app as cli_app
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import adapters as fleet_adapters
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines import choice, ik_llama
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import HostFacts, installed_engines
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.roles import WorkerRole

COMMIT = "13cffc006b82e03e4d89cbb8c7327f0255ae3d18"
runner = CliRunner()


def _host(system: str, machine: str, flags: list[str] | None, mac_ver: str = "") -> Any:
    cpu_info: dict[str, Any] = {} if flags is None else {"flags": flags, "brand_raw": "Test CPU"}
    return host_caps.collect(
        system=system,
        machine=machine,
        mac_ver=mac_ver,
        cpu_info=cpu_info,
        nvml=None,
        npu_devices=[],
    )


def _make_wheel(path: Path) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo("lilbee_engine_ik/bin/llama-server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("lilbee_engine_ik-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "preferred_engine", "auto")
    monkeypatch.setattr(cfg, "engine_choices", None)
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system="linux", machine="x86_64")
    )


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fixture pack as the pinned build; the pin takes the digest of that pack."""
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    monkeypatch.setattr(ik_llama, "PACK_SHA256", sha)
    lock = PinLock(EngineName.IK_LLAMA.value, ik_llama.VERSION, ik_llama.BACKEND, {}, ("MIT",), sha)
    return EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)


def test_the_pin_names_the_commit_by_its_first_twelve_digits() -> None:
    pin = IK_LLAMA.pin_record()
    assert ik_llama.IK_LLAMA_COMMIT == COMMIT
    assert pin["version"] == f"git-{COMMIT[:12]}"
    assert (pin["engine"], pin["backend"], pin["licenses"]) == ("ik_llama.cpp", "cpu", ["MIT"])
    assert choice.pin_lock({**pin, "sha256": "a" * 64}).version == "git-13cffc006b82"


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
    monkeypatch.setattr(ik_llama, "PACK_SHA256", "1" * 64)
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
    toolless = tmp_path / "toolless.whl"
    with zipfile.ZipFile(toolless, "w") as archive:
        archive.writestr("lilbee_engine_ik-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(toolless.read_bytes()).hexdigest()
    store.install_from_wheel(
        PinLock("ik_llama.cpp", "git-aaaaaaaaaaaa", "cpu", {}, ("MIT",), sha), toolless
    )
    store.install_from_wheel(
        PinLock("ik_llama.cpp", ik_llama.VERSION, "cpu", {}, ("MIT",), sha), toolless
    )
    with pytest.raises(ProviderError, match="is not installed"):
        IK_LLAMA.binary()


@pytest.mark.parametrize(
    ("version", "backend"),
    [("git-aaaaaaaaaaaa", ik_llama.BACKEND), (ik_llama.VERSION, "cuda")],
)
def test_binary_skips_a_build_of_another_version_or_backend_that_has_a_server(
    version: str, backend: str, tmp_path: Path
) -> None:
    wheel = tmp_path / "ik.whl"
    sha = _make_wheel(wheel)
    lock = PinLock("ik_llama.cpp", version, backend, {}, ("MIT",), sha)
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    with pytest.raises(ProviderError, match="is not installed"):
        IK_LLAMA.binary()


@pytest.mark.parametrize(
    ("system", "machine", "flags", "mac_ver"),
    [("Linux", "x86_64", ["avx2"], ""), ("Darwin", "arm64", None, "15.0")],
)
def test_a_cpu_with_avx2_or_neon_passes_the_gate(
    system: str, machine: str, flags: list[str] | None, mac_ver: str
) -> None:
    gate = IK_LLAMA.host_gate(_host(system, machine, flags, mac_ver))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, True, True)


def test_a_cpu_without_avx2_is_refused_naming_avx2_and_llama_cpp() -> None:
    gate = IK_LLAMA.host_gate(_host("Linux", "x86_64", ["avx", "sse4_2"]))
    assert gate.ok is False
    assert gate.reason == (
        "Test CPU lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON. llama.cpp stays in use."
    )


def test_an_unreadable_cpu_stays_selectable_but_never_automatic() -> None:
    gate = IK_LLAMA.host_gate(_host("Linux", "x86_64", None))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)
    assert gate.reason == "the CPU flags of this CPU could not be read"


def test_an_unreadable_architecture_stays_selectable_but_never_automatic() -> None:
    gate = IK_LLAMA.host_gate(_host("Plan9", "mips", None))
    assert (gate.ok, gate.known) == (True, False)


def test_install_on_a_host_without_avx2_is_refused_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_caps, "detect", lambda: _host("Linux", "x86_64", ["avx"]))
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("ik_llama.cpp")
    assert "AVX2" in str(exc.value)
    assert "llama.cpp stays in use" in str(exc.value)
    assert not engines_root(cfg.models_dir).exists()


def test_the_adapter_records_no_pack_digest_until_a_pack_is_published() -> None:
    assert ik_llama.PACK_SHA256 is None
    assert IK_LLAMA.pin_record()["sha256"] is None


def test_install_with_no_published_pack_says_so_for_a_wheel_and_the_packs_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_caps, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
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
    monkeypatch.setattr(host_caps, "detect", lambda: _host("Linux", "x86_64", ["avx2"]))
    wheel = tmp_path / "ik.whl"
    monkeypatch.setattr(ik_llama, "PACK_SHA256", _make_wheel(wheel))
    built = engine_service.install("ik_llama.cpp", wheel)
    assert (built.engine, built.version, built.backend) == (
        "ik_llama.cpp",
        f"git-{COMMIT[:12]}",
        "cpu",
    )
    assert EngineName.IK_LLAMA in installed_engines()


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_auto_never_picks_the_installed_engine(system: str, build: Path, monkeypatch) -> None:
    machine = "arm64" if system == "darwin" else "x86_64"
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system=system, machine=machine)
    )
    assert EngineName.IK_LLAMA in installed_engines()
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


def test_engine_list_json_names_the_engines_that_are_not_installed() -> None:
    import json

    result = runner.invoke(cli_app, ["--json", "engine", "list"])
    body = json.loads(result.output)
    assert body["builds"] == []
    assert body["not_installed"] == ["ik_llama.cpp", "tei", "mistral.rs"]


def test_engine_list_shows_an_installed_build_and_no_not_installed_line(build: Path) -> None:
    result = runner.invoke(cli_app, ["engine", "list"])
    assert f"ik_llama.cpp git-{COMMIT[:12]} (cpu)" in result.output
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
        ("estimate_memory", None),
        ("check_readback", False),
        ("supports_fit_target", False),
        ("supports_shared_kv", False),
        ("swap_extras", {}),
    ]:
        monkeypatch.setattr(LLAMA, method, spy(method, result))
    path = Path("/b/server")
    IK_LLAMA.identity(path)
    IK_LLAMA.discover_devices(path)
    IK_LLAMA.model_meta(path)
    IK_LLAMA.assert_loadable("repo", "file", None)
    IK_LLAMA.runtime_env()
    IK_LLAMA.estimate_memory(None)  # type: ignore[arg-type]
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
        "estimate_memory",
        "check_readback",
        "supports_fit_target",
        "supports_shared_kv",
        "swap_extras",
    ]
