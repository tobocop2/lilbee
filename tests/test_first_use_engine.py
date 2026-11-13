"""The first use of an engine installs the llama.cpp pack once, then the launch goes on."""

from __future__ import annotations

import hashlib
import stat
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import Config, cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import engine_manifest, first_use
from lilbee.providers.fleet.binary import (
    EngineTool,
    resolve_engine_tool,
    resolve_gguf_parser,
    resolve_llama_server,
)
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import _WHEEL_VERSION
from lilbee.providers.fleet.host_caps import collect
from tests._published import render_table, wheel_name

PACKAGE = "lilbee_engine_llama_cpp"
CELL = "linux-x86_64-cpu"
NAMED = "The llama.cpp pack 1.12032.1 for linux-x86_64-cpu is not published yet."


def make_pack(path: Path) -> str:
    """A llama.cpp wheel with the three tools and the base library; returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        for tool in ("llama-server", "llama-swap", "gguf-parser"):
            info = zipfile.ZipInfo(f"{PACKAGE}/bin/{tool}")
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, "#!/bin/sh\n")
        archive.writestr(f"{PACKAGE}/libggml-base.so", "lib")
        archive.writestr(f"{PACKAGE}-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.llama_server_path = ""
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvml=None)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    monkeypatch.setattr(binary_mod, "_bundled_tool", lambda _tool: None)
    monkeypatch.setattr(binary_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def publish(monkeypatch: pytest.MonkeyPatch, sha: str) -> None:
    file = wheel_name(EngineName.LLAMA, _WHEEL_VERSION, CELL)
    table = render_table([(file, sha, 1)])
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)


@pytest.fixture
def fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A published pack and a fake download that records each fetch."""
    wheel = tmp_path / "pack" / "llama.whl"
    wheel.parent.mkdir()
    publish(monkeypatch, make_pack(wheel))
    calls: list[str] = []

    def fake(lock: Any, cell: str, *, dest: Path) -> Path:
        calls.append(cell)
        dest.mkdir(parents=True, exist_ok=True)
        copy = dest / wheel.name
        copy.write_bytes(wheel.read_bytes())
        return copy

    monkeypatch.setattr(first_use, "fetch_frozen", fake)
    return calls


def test_a_missing_engine_installs_the_pack_once_and_the_launch_gets_the_server(fetched) -> None:
    first = resolve_llama_server()
    second = resolve_llama_server()
    assert first == second
    assert first.name == "llama-server"
    assert first.is_file()
    assert fetched == [CELL]


def test_the_sibling_tools_come_from_the_installed_pack(fetched) -> None:
    resolve_llama_server()
    assert resolve_gguf_parser().parent == resolve_llama_server().parent
    assert fetched == [CELL]


def test_an_unpublished_pack_gives_the_named_error_with_the_install_command() -> None:
    with pytest.raises(ProviderError) as excinfo:
        resolve_llama_server()
    assert NAMED in str(excinfo.value)
    assert "lilbee engine install" in str(excinfo.value)
    assert excinfo.value.kind is ProviderErrorKind.NOT_FOUND


def test_an_unpublished_pack_leaves_aux_tools_unknown_not_not_found() -> None:
    with pytest.raises(ProviderError) as excinfo:
        resolve_engine_tool(EngineTool.GGUF_PARSER)
    assert excinfo.value.kind is ProviderErrorKind.UNKNOWN


def test_a_pack_copied_into_the_cache_by_hand_is_used_without_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EngineStore(engines_root(cfg.models_dir))
    lock, _cell = _lock_for(monkeypatch, tmp_path)
    pack = store.pack_path(lock)
    pack.parent.mkdir(parents=True)
    pack.write_bytes((tmp_path / "hand.whl").read_bytes())

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("the pack in the cache needs no download")

    monkeypatch.setattr(first_use, "fetch_frozen", refuse)
    assert resolve_llama_server().is_file()


def _lock_for(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    wheel = tmp_path / "hand.whl"
    publish(monkeypatch, make_pack(wheel))
    return first_use.pack_lock()


def test_two_concurrent_first_uses_download_once(fetched) -> None:
    results: list[Path] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(resolve_llama_server())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(set(results)) == 1
    assert fetched == [CELL]


def test_a_failed_download_names_the_pack_to_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(monkeypatch, make_pack(tmp_path / "w.whl"))

    def offline(*_args: object, **_kwargs: object) -> Path:
        raise OSError("no network")

    monkeypatch.setattr(first_use, "fetch_frozen", offline)
    with pytest.raises(ProviderError, match=r"Copy llama\.cpp-1\.12032\.1-cpu\.whl"):
        resolve_llama_server()


def test_an_unreadable_engine_store_gives_the_engine_error_not_a_raw_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(monkeypatch, make_pack(tmp_path / "w.whl"))

    def unreadable(_self: EngineStore) -> list[Any]:
        raise PermissionError("the engines folder cannot be listed")

    monkeypatch.setattr(EngineStore, "installed", unreadable)
    with pytest.raises(ProviderError, match="llama-server binary not found") as excinfo:
        resolve_llama_server()
    assert excinfo.value.kind is ProviderErrorKind.NOT_FOUND
    assert "the engines folder cannot be listed" in str(excinfo.value)
    assert "lilbee engine install" in str(excinfo.value)


def test_an_engine_on_the_path_wins_over_the_pack(fetched, monkeypatch) -> None:
    monkeypatch.setattr(binary_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert resolve_llama_server() == Path("/usr/bin/llama-server")
    assert fetched == []


def test_the_install_hint_does_not_name_the_engine_extra() -> None:
    from lilbee.providers.fleet.engine_install import InstallMethod, engine_install_hint

    for method in InstallMethod:
        assert "[engine]" not in engine_install_hint(method)
    assert "lilbee engine install" in engine_install_hint(InstallMethod.PIP)


def test_the_engine_pin_follows_the_installed_pack(fetched, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)
    assert binary_mod.engine_build_id() == "unpinned"
    server = resolve_llama_server()
    assert binary_mod.engine_build_id().startswith(f"pack:{server}")


def test_the_install_command_fetches_the_pack_when_no_engine_is_present(fetched) -> None:
    from typer.testing import CliRunner

    from lilbee.cli.app import app as cli_app

    result = CliRunner().invoke(cli_app, ["engine", "install"])
    assert result.exit_code == 0, result.output
    assert "Installed engine llama.cpp" in result.output
    assert fetched == [CELL]
    assert resolve_llama_server().is_file()
    assert fetched == [CELL]


def test_the_install_command_names_the_missing_pack_when_none_is_published() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="is not published yet"):
        engine_service.install(EngineName.LLAMA.value)


def test_a_host_older_than_the_floor_of_the_pack_gets_the_floor_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = wheel_name(EngineName.LLAMA, _WHEEL_VERSION, CELL, "2.28")
    monkeypatch.setattr(
        engine_manifest, "MANIFESTS", render_table([(file, make_pack(tmp_path / "w.whl"), 1)])
    )
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "2.17")
    with pytest.raises(ProviderError, match=r"needs glibc 2\.28 or later"):
        resolve_llama_server()


def test_the_install_command_reports_a_lock_held_too_long(
    fetched, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.core.security import FileLockUnavailableError

    def busy(*_args: object, **_kwargs: object) -> Path:
        raise FileLockUnavailableError("another lilbee process held the lock")

    monkeypatch.setattr(first_use, "install_pack", busy)
    with pytest.raises(engine_service.EngineRefusedError, match="held the lock"):
        engine_service.install(EngineName.LLAMA.value)


@pytest.mark.parametrize(
    ("engine", "version", "backend"),
    [
        ("fx", _WHEEL_VERSION, "cpu"),
        ("llama.cpp", "0.0.1", "cpu"),
        ("llama.cpp", _WHEEL_VERSION, "cuda"),
    ],
    ids=["other-engine", "old-version", "other-backend"],
)
def test_a_runnable_build_that_does_not_match_the_pin_is_not_the_llama_server(
    engine: str, version: str, backend: str
) -> None:
    import json

    build = engines_root(cfg.models_dir) / engine / f"{version}-{backend}"
    (build / "bin").mkdir(parents=True)
    server = build / "bin" / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    record = {
        "engine": engine,
        "version": version,
        "backend": backend,
        "sha256": "0" * 64,
        "licences": ["MIT"],
        "installed_at": "2026-01-01T00:00:00+00:00",
        "binary": "bin/llama-server",
    }
    (build / "ENGINE-RECORD.json").write_text(json.dumps(record), encoding="utf-8")
    assert any(b.binary == server for b in EngineStore(engines_root(cfg.models_dir)).installed())
    assert first_use.installed_tool("llama-server") is None


def test_a_wheel_file_installs_the_pack_with_no_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "given.whl"
    publish(monkeypatch, make_pack(wheel))

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("a given wheel needs no download")

    monkeypatch.setattr(first_use, "fetch_frozen", refuse)
    first_use.install_pack(wheel)
    assert EngineStore(engines_root(cfg.models_dir)).installed()[0].engine == "llama.cpp"
    assert first_use.installed_tool("gguf-parser") is not None


def test_an_unreadable_store_leaves_the_engine_pin_unpinned(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)

    def unreadable(_tool: str) -> Path:
        raise OSError("store unreadable")

    monkeypatch.setattr(first_use, "installed_tool", unreadable)
    assert binary_mod.engine_build_id() == "unpinned"


@pytest.mark.parametrize(("vendor_id", "vendor"), [(0x1002, "AMD"), (0x8086, "Intel")])
def test_installing_the_cpu_pack_on_an_unserved_gpu_host_names_the_vendor(
    vendor_id: int,
    vendor: str,
    fetched: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: frozenset({vendor_id}))
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.install_pack()
    assert fetched == [CELL]
    [record] = caplog.records
    message = record.getMessage()
    assert f"has an {vendor} GPU" in message
    assert "installs the CPU build" in message
    assert "LILBEE_LLAMA_SERVER_PATH" in message


@pytest.mark.parametrize(
    "vendors",
    [frozenset(), frozenset({0x10DE}), frozenset({0x1AF4})],
    ids=["none", "nvidia", "other"],
)
def test_a_host_with_no_amd_or_intel_gpu_installs_the_cpu_pack_silently(
    vendors: frozenset[int],
    fetched: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: vendors)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.install_pack()
    assert fetched == [CELL]
    assert caplog.records == []


def test_a_host_whose_pack_uses_the_gpu_is_not_warned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mac = collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None)
    monkeypatch.setattr(first_use, "session_host", lambda: mac)
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: frozenset({0x1002}))
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.records == []


def test_a_wheel_that_is_not_a_zip_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken.whl"
    broken.write_bytes(b"not a zip")
    publish(monkeypatch, hashlib.sha256(broken.read_bytes()).hexdigest())
    with pytest.raises(engine_service.EngineRefusedError, match="was not installed"):
        engine_service.install(EngineName.LLAMA.value, broken)


def _own_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, on_path: bool) -> None:
    server = tmp_path / "my" / "llama-server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    if on_path:
        monkeypatch.setattr(
            binary_mod.shutil, "which", lambda name: str(server) if name == "llama-server" else None
        )
    else:
        cfg.llama_server_path = str(server)


@pytest.mark.parametrize("on_path", [False, True], ids=["configured-path", "on-path"])
@pytest.mark.parametrize(
    ("tool", "command"),
    [
        (
            EngineTool.GGUF_PARSER,
            "go install github.com/gpustack/gguf-parser-go/cmd/gguf-parser@latest",
        ),
        (EngineTool.LLAMA_SWAP, "go install github.com/mostlygeek/llama-swap@latest"),
    ],
    ids=["gguf-parser", "llama-swap"],
)
def test_your_own_server_never_installs_a_pack_for_a_missing_helper(
    tool: EngineTool,
    command: str,
    on_path: bool,
    fetched,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _own_server(monkeypatch, tmp_path, on_path=on_path)
    with pytest.raises(ProviderError) as excinfo:
        resolve_engine_tool(tool)
    assert fetched == []
    assert excinfo.value.kind is ProviderErrorKind.UNKNOWN
    assert f"Install {tool.value} with '{command}'" in str(excinfo.value)
    assert "engine install" not in str(excinfo.value)
