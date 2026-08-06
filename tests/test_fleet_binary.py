"""Tests for bundled engine binary resolution (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet.binary import (
    EngineTool,
    _engine_build_id,
    engine_pin,
    llama_server_runtime_env,
    resolve_engine_tool,
    resolve_gguf_parser,
    resolve_llama_server,
    resolve_llama_swap,
)

_WHICH = "lilbee.providers.fleet.binary.shutil.which"

_ALL_TOOLS = [EngineTool.LLAMA_SERVER, EngineTool.LLAMA_SWAP, EngineTool.GGUF_PARSER]


def _fake_engine(tmp_path: Path, *, make_files: bool) -> SimpleNamespace:
    """A stand-in ``lilbee_engine`` module exposing the three path accessors."""
    paths = {}
    for tool in _ALL_TOOLS:
        p = tmp_path / tool.value
        if make_files:
            p.write_text("#!/bin/sh\n")
        paths[tool] = p
    return SimpleNamespace(
        get_llama_server_path=lambda: paths[EngineTool.LLAMA_SERVER],
        get_llama_swap_path=lambda: paths[EngineTool.LLAMA_SWAP],
        get_gguf_parser_path=lambda: paths[EngineTool.GGUF_PARSER],
    )


@pytest.mark.parametrize("tool", _ALL_TOOLS)
def test_resolves_bundled_tool_when_present(
    tool: EngineTool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure the configured llama_server_path override does not mask the bundled binary.
    cfg.llama_server_path = ""
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    assert resolve_engine_tool(tool) == tmp_path / tool.value


def test_thin_wrappers_resolve_their_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the configured llama_server_path override does not mask the bundled binary.
    cfg.llama_server_path = ""
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    assert resolve_llama_server() == tmp_path / "llama-server"
    assert resolve_llama_swap() == tmp_path / "llama-swap"
    assert resolve_gguf_parser() == tmp_path / "gguf-parser"


def test_bundled_tool_none_when_package_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing a module set to None raises ImportError -> resolution falls through.
    monkeypatch.setitem(sys.modules, "lilbee_engine", None)
    assert binary_mod._bundled_tool(EngineTool.LLAMA_SWAP) is None


def test_bundled_tool_none_when_binary_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Installed wheel whose bin/ has no binary yet (CI fills it at build time).
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=False))
    assert binary_mod._bundled_tool(EngineTool.GGUF_PARSER) is None


def test_llama_server_uses_configured_path_when_file_exists(tmp_path: Path) -> None:
    binpath = tmp_path / "llama-server"
    binpath.write_text("#!/bin/sh\n")
    cfg.llama_server_path = str(binpath)
    assert resolve_llama_server() == binpath


def test_llama_server_raises_when_configured_path_missing(tmp_path: Path) -> None:
    cfg.llama_server_path = str(tmp_path / "absent")
    with pytest.raises(ProviderError, match="is not a file"):
        resolve_llama_server()


def test_llama_server_configured_path_wins_over_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    custom = tmp_path / "custom-llama-server"
    custom.write_text("#!/bin/sh\n")
    cfg.llama_server_path = str(custom)
    assert resolve_llama_server() == custom


def test_llama_server_unset_falls_back_to_bundled_then_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.llama_server_path = ""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    empty_wheel_dir = tmp_path / "empty-wheel"
    empty_wheel_dir.mkdir()
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(wheel_dir, make_files=True))
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server() == wheel_dir / "llama-server"
    monkeypatch.setitem(
        sys.modules, "lilbee_engine", _fake_engine(empty_wheel_dir, make_files=False)
    )
    assert resolve_llama_server() == Path("/usr/local/bin/llama-server")


def test_llama_server_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server() == Path("/usr/local/bin/llama-server")


def test_llama_server_raises_with_install_hint_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError, match="not found"):
        resolve_llama_server()


def test_missing_llama_server_is_not_found_but_aux_tools_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the llama-server resolution carries NOT_FOUND (the quiet engine-less
    host path). A missing gguf-parser must not, or the sizing fallback would
    misreport it as a model that isn't installed."""
    from lilbee.providers.base import ProviderErrorKind

    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError) as server_exc:
        resolve_llama_server()
    assert server_exc.value.kind is ProviderErrorKind.NOT_FOUND
    with pytest.raises(ProviderError) as parser_exc:
        resolve_gguf_parser()
    assert parser_exc.value.kind is ProviderErrorKind.UNKNOWN


def test_aux_tools_ignore_llama_server_path_and_use_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # llama-swap / gguf-parser do not honor LILBEE_LLAMA_SERVER_PATH; only PATH.
    cfg.llama_server_path = str(tmp_path / "llama-server")  # would mislead a shared path
    monkeypatch.setattr(_WHICH, lambda name: f"/usr/local/bin/{name}")
    assert resolve_llama_swap() == Path("/usr/local/bin/llama-swap")
    assert resolve_gguf_parser() == Path("/usr/local/bin/gguf-parser")


def test_aux_tool_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError, match="gguf-parser binary not found"):
        resolve_gguf_parser()


def test_runtime_env_delegates_to_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # The self-contained wheel and a BYO binary carry their own libs; the only
    # per-spawn env is the CUDA-runtime wheel path that driver-only images need.
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.cuda_runtime_env",
        lambda: {"LD_LIBRARY_PATH": "/wheel/lib"},
    )
    assert llama_server_runtime_env() == {"LD_LIBRARY_PATH": "/wheel/lib"}


class TestEnginePin:
    """The pin identifies the engine BUILD a lilbee would spawn; sharing keys on it."""

    def test_custom_llama_server_path_is_its_own_identity(self, tmp_path: Path) -> None:
        exe = tmp_path / "llama-server"
        exe.write_text("#!/bin/sh\n")
        original = cfg.llama_server_path
        cfg.llama_server_path = str(exe)
        try:
            assert _engine_build_id().startswith(f"custom:{exe}@")  # path + build fingerprint
        finally:
            cfg.llama_server_path = original

    def test_binary_signature_degrades_on_an_unstatable_path(self) -> None:
        # engine_pin runs on every state write and must not raise; an unstatable
        # binary path degrades to a fixed marker.
        assert binary_mod._binary_signature(Path("/does/not/exist/llama-server")) == "unstatable"

    def test_custom_pin_tracks_an_in_place_binary_replacement(self, tmp_path: Path) -> None:
        # Replacing the binary at the same path (a brew upgrade) must change the pin,
        # so a new process never binds to an engine spawned from the old build.
        import os
        import time

        exe = tmp_path / "llama-server"
        exe.write_text("#!/bin/sh\n# build A\n")
        original = cfg.llama_server_path
        cfg.llama_server_path = str(exe)
        try:
            pin_a = _engine_build_id()
            time.sleep(0.01)
            exe.write_text("#!/bin/sh\n# build B is larger than A\n")
            os.utime(exe, ns=(time.time_ns(), time.time_ns()))  # a real replace bumps mtime
            assert _engine_build_id() != pin_a
        finally:
            cfg.llama_server_path = original

    def test_placement_is_part_of_the_load_signature(self) -> None:
        # A process with a manual GPU placement must not adopt an engine placed
        # differently, so placement flows into the pin's load signature.
        original = cfg.placement
        try:
            cfg.placement = None
            base = binary_mod._load_config_signature()
            cfg.placement = '{"chat": {"devices": [1]}}'
            assert binary_mod._load_config_signature() != base
        finally:
            cfg.placement = original

    def test_ctx_sizing_values_are_not_part_of_the_load_signature(self) -> None:
        # Ctx sizing shares by window coverage (contract.chat_ctx_covers), not
        # value equality: a co-tenant whose target fits inside the running
        # window must compute the same pin and adopt the engine.
        originals = {k: getattr(cfg, k) for k in ("num_ctx", "num_ctx_max", "chat_n_ctx_target")}
        try:
            cfg.num_ctx = None
            cfg.num_ctx_max = None
            cfg.chat_n_ctx_target = 12288
            base = binary_mod._load_config_signature()
            cfg.chat_n_ctx_target = 49152
            assert binary_mod._load_config_signature() == base
            cfg.num_ctx = 4096
            assert binary_mod._load_config_signature() == base
            cfg.num_ctx_max = 16384
            assert binary_mod._load_config_signature() == base
        finally:
            for key, value in originals.items():
                setattr(cfg, key, value)

    def test_gpu_devices_is_part_of_the_load_signature(self) -> None:
        original = cfg.gpu_devices
        try:
            cfg.gpu_devices = None
            base = binary_mod._load_config_signature()
            cfg.gpu_devices = "1"
            assert binary_mod._load_config_signature() != base
        finally:
            cfg.gpu_devices = original

    def test_bundled_wheel_pin_wins_when_no_custom_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_engine(tmp_path, make_files=True)
        fake.get_engine_pin = lambda: "llama-cpp-9.9.9+swap-v999+gguf-v9.9.9"
        monkeypatch.setitem(sys.modules, "lilbee_engine", fake)
        assert _engine_build_id() == "llama-cpp-9.9.9+swap-v999+gguf-v9.9.9"

    def test_path_fallback_identity_when_wheel_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "lilbee_engine", None)
        monkeypatch.setattr(_WHICH, lambda name: f"/opt/homebrew/bin/{name}")
        # path + build fingerprint; the fake path is unstatable so it degrades safely.
        assert _engine_build_id().startswith("path:/opt/homebrew/bin/llama-server@")

    def test_unpinned_when_nothing_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lilbee_engine", None)
        monkeypatch.setattr(_WHICH, lambda name: None)
        assert _engine_build_id() == "unpinned"

    def test_wheel_without_pin_accessor_reports_its_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_engine(tmp_path, make_files=True)  # no get_engine_pin attribute
        monkeypatch.setitem(sys.modules, "lilbee_engine", fake)
        monkeypatch.setattr("lilbee.providers.fleet.binary._pkg_version", lambda name: "0.6.91")
        assert _engine_build_id() == "wheel:0.6.91"

    def test_pin_folds_in_load_affecting_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same build, different expert-offload config: the pins must differ so two
        # processes with conflicting load flags never share one engine.
        monkeypatch.setattr(binary_mod, "_engine_build_id", lambda: "build-x")
        monkeypatch.setattr(cfg, "cpu_moe", False)
        pin_off = engine_pin()
        monkeypatch.setattr(cfg, "cpu_moe", True)
        pin_on = engine_pin()
        assert pin_off != pin_on
        assert pin_on.startswith("build-x|")  # build identity still leads the pin

    def test_pin_is_stable_for_identical_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The load signature is deterministic: identical config yields the same pin
        # (a jittering pin would make same-setup peers overflow instead of share).
        monkeypatch.setattr(binary_mod, "_engine_build_id", lambda: "build-x")
        assert engine_pin() == engine_pin()

    def test_checked_in_pins_match_engine_versions_env(self) -> None:
        import lilbee_engine

        env = {}
        env_path = Path(__file__).parent.parent / "engine-versions.env"
        for line in env_path.read_text().splitlines():
            if line.startswith("ENGINE_") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
        pin = lilbee_engine.get_engine_pin()
        assert env["ENGINE_LLAMA_CPP_VERSION"] in pin
        assert env["ENGINE_LLAMA_SWAP_VERSION"] in pin
        assert env["ENGINE_GGUF_PARSER_REF"] in pin


def _plant_host_binary(directory: Path, tool: EngineTool) -> Path:
    """An executable engine binary as a developer machine would carry on PATH."""
    exe = directory / (tool.value + (".exe" if os.name == "nt" else ""))
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return exe


class TestEngineResolutionSeal:
    """The suite-wide seal: engine binaries a test did not plant never resolve."""

    @pytest.mark.parametrize("tool", _ALL_TOOLS)
    def test_host_path_binary_does_not_resolve(
        self, tool: EngineTool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "llama_server_path", "")
        _plant_host_binary(tmp_path, tool)
        monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
        with pytest.raises(ProviderError, match="not found"):
            resolve_engine_tool(tool)

    def test_engine_build_id_ignores_host_path_binaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "llama_server_path", "")
        monkeypatch.setitem(sys.modules, "lilbee_engine", None)
        _plant_host_binary(tmp_path, EngineTool.LLAMA_SERVER)
        monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
        assert _engine_build_id() == "unpinned"

    @pytest.mark.real_engine_resolution
    def test_marker_restores_host_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "llama_server_path", "")
        planted = _plant_host_binary(tmp_path, EngineTool.LLAMA_SERVER)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_llama_server() == planted


def test_engine_pin_survives_an_engine_wheel_without_metadata(monkeypatch) -> None:
    """engine_pin runs on every state write, so it must not raise here.

    lilbee_engine can be importable with nothing to look up: an extracted wheel
    on sys.path, a vendored copy, or a dist whose name does not normalize to
    lilbee-engine. A PackageNotFoundError escaping here aborts the state write.
    """
    import sys
    import types
    from importlib.metadata import PackageNotFoundError

    from lilbee.providers.fleet import binary as binary_mod

    stub = types.ModuleType("lilbee_engine")  # no get_engine_pin -> pre-pin path
    monkeypatch.setitem(sys.modules, "lilbee_engine", stub)
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "llama_server_path", "", raising=False)

    def _no_metadata(_name: str) -> str:
        raise PackageNotFoundError("lilbee-engine")

    monkeypatch.setattr(binary_mod, "_pkg_version", _no_metadata)
    assert binary_mod._engine_build_id() == "wheel:unknown"
    assert binary_mod.engine_pin()  # total: a pin is still produced


def test_install_hint_indexes_are_documented() -> None:
    """Every index the error message names must also appear in the install docs.

    The bug this guards was the two disagreeing: the docs shipped install commands
    that produced no engine while the message that fired blamed the model. A reader
    who follows the error and then checks the README has to find the same set.
    """
    import re

    from lilbee.providers.fleet.binary import _INSTALL_HINT

    hint_indexes = set(re.findall(r"https://lilbee\.sh/(\w+)/", _INSTALL_HINT))
    assert hint_indexes, "the hint must name at least one index"

    root = Path(__file__).resolve().parents[1]
    for doc in ("README.md", "docs/usage.md"):
        documented = set(re.findall(r"https://lilbee\.sh/(\w+)/", (root / doc).read_text()))
        assert hint_indexes <= documented, f"{doc} is missing {hint_indexes - documented}"
