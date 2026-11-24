"""Tests for bundled engine binary resolution (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import swap_config as swap_config_mod
from lilbee.providers.fleet.binary import (
    EngineTool,
    engine_build_id,
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
    """A stand-in ``lilbee_engine_llama_cpp`` module exposing the three path accessors."""
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
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(tmp_path, make_files=True)
    )
    assert resolve_engine_tool(tool) == tmp_path / tool.value


def test_thin_wrappers_resolve_their_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the configured llama_server_path override does not mask the bundled binary.
    cfg.llama_server_path = ""
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(tmp_path, make_files=True)
    )
    assert resolve_llama_server() == tmp_path / "llama-server"
    assert resolve_llama_swap() == tmp_path / "llama-swap"
    assert resolve_gguf_parser() == tmp_path / "gguf-parser"


def test_bundled_tool_none_when_package_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing a module set to None raises ImportError -> resolution falls through.
    monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)
    assert binary_mod._bundled_tool(EngineTool.LLAMA_SWAP) is None


def test_bundled_tool_none_when_the_wheel_predates_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An engine wheel old enough to ship no llama-swap has no accessor for it.
    # Resolution must fall through to PATH rather than raise out of every caller.
    monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", SimpleNamespace())
    assert binary_mod._bundled_tool(EngineTool.LLAMA_SWAP) is None


def test_the_binary_identity_survives_a_path_that_is_not_there(tmp_path: Path) -> None:
    # Read on every plan, so an engine that vanished answers with a marker.
    assert binary_mod.engine_binary_identity(tmp_path / "gone") == f"{tmp_path / 'gone'}@unreadable"


def test_a_rewritten_binary_is_a_new_identity_with_every_stat_field_unmoved(
    tmp_path: Path,
) -> None:
    # An upgrade rewrites the engine in place. The rewrite keeps the inode, a
    # build of the same size written inside one mtime tick moves no timestamp,
    # and then no stat field says anything changed. Keying on metadata would
    # serve the old binary's device probe for the new engine, which is the whole
    # defect. Only the bytes tell them apart.
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"the stub engine")
    before = binary_mod.engine_binary_identity(binary)
    stat = binary.stat()
    with binary.open("r+b") as handle:
        handle.write(b"the real engine")
    os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    rewritten = binary.stat()
    assert (rewritten.st_dev, rewritten.st_ino, rewritten.st_size, rewritten.st_mtime_ns) == (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    assert binary_mod.engine_binary_identity(binary) != before


def test_bundled_tool_none_when_binary_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Installed wheel whose bin/ has no binary yet (CI fills it at build time).
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(tmp_path, make_files=False)
    )
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
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(tmp_path, make_files=True)
    )
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
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(wheel_dir, make_files=True)
    )
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server() == wheel_dir / "llama-server"
    monkeypatch.setitem(
        sys.modules, "lilbee_engine_llama_cpp", _fake_engine(empty_wheel_dir, make_files=False)
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


class TestRouterRelease:
    """The router reports its release; an old one is still used."""

    @staticmethod
    def _router_says(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
        calls: list[list[str]] = []

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_WHICH, lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(binary_mod.subprocess, "run", run)
        return calls

    def test_a_router_older_than_the_floor_is_found_like_any_other(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._router_says(monkeypatch, "version: 223 (8c5e40e), built at x\n")
        assert resolve_llama_swap() == Path("/usr/bin/llama-swap")
        assert calls == []

    @pytest.mark.parametrize(
        ("stdout", "release"),
        [
            ("version: 223 (8c5e40e), built at x\n", 223),
            ("version: v258 (abc1234), built at x\n", 258),
            ("version: v259 (abc1234), built at x\n", 259),
            ("version: v272 (802f1c4), built at x\n", 272),
            ("version: 0 (unknown), built at unknown\n", None),
            ("llama-swap\n", None),
            ("", None),
        ],
        ids=["no-v", "last-before-floor", "floor", "pin", "source-build", "no-line", "silent"],
    )
    def test_the_release_is_read_from_the_version_line(
        self, stdout: str, release: int | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._router_says(monkeypatch, stdout)
        router = Path("/usr/bin/llama-swap")
        assert binary_mod.router_release(router) == release
        assert calls == [[str(router), "--version"]]  # the path in the form of this host

    @pytest.mark.parametrize(
        "error",
        [OSError("exec format error"), subprocess.TimeoutExpired(["llama-swap"], 15.0)],
        ids=["cannot-run", "hangs"],
    )
    def test_a_router_that_does_not_answer_has_no_release(
        self, error: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            raise error

        monkeypatch.setattr(binary_mod.subprocess, "run", run)
        assert binary_mod.router_release(Path("/usr/bin/llama-swap")) is None

    def test_the_release_of_one_router_file_is_read_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        router = tmp_path / "llama-swap"
        router.write_bytes(b"one")
        calls = self._router_says(monkeypatch, "version: v272 (802f1c4), built at x\n")
        assert [binary_mod.router_release(router) for _ in range(3)] == [272, 272, 272]
        assert len(calls) == 1

    def test_a_replaced_router_file_is_read_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        router = tmp_path / "llama-swap"
        router.write_bytes(b"one")
        says = ["version: 223 (8c5e40e), built at x\n"]

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout=says[-1], stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", run)
        assert binary_mod.router_release(router) == 223
        router.write_bytes(b"a longer file")
        says.append("version: v272 (802f1c4), built at x\n")
        assert binary_mod.router_release(router) == 272

    def test_a_router_file_of_the_same_size_with_a_new_mtime_is_read_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        router = tmp_path / "llama-swap"
        router.write_bytes(b"old")
        os.utime(router, ns=(1_000_000_000, 1_000_000_000))
        says = ["version: 223 (8c5e40e), built at x\n"]

        def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout=says[-1], stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", run)
        assert binary_mod.router_release(router) == 223
        router.write_bytes(b"new")
        os.utime(router, ns=(2_000_000_000, 2_000_000_000))
        says.append("version: v272 (802f1c4), built at x\n")
        assert binary_mod.router_release(router) == 272

    def test_the_release_of_a_router_that_cannot_be_statted_is_not_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._router_says(monkeypatch, "version: v272 (802f1c4), built at x\n")
        missing = tmp_path / "missing"
        assert [binary_mod.router_release(missing) for _ in range(2)] == [272, 272]
        assert len(calls) == 2


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
            assert engine_build_id().startswith(f"custom:{exe}@")  # path + build fingerprint
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
            pin_a = engine_build_id()
            time.sleep(0.01)
            exe.write_text("#!/bin/sh\n# build B is larger than A\n")
            os.utime(exe, ns=(time.time_ns(), time.time_ns()))  # a real replace bumps mtime
            assert engine_build_id() != pin_a
        finally:
            cfg.llama_server_path = original

    def test_bundled_wheel_pin_wins_when_no_custom_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_engine(tmp_path, make_files=True)
        fake.get_engine_pin = lambda: "llama-cpp-9.9.9+swap-v999+gguf-v9.9.9"
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", fake)
        assert engine_build_id() == "llama-cpp-9.9.9+swap-v999+gguf-v9.9.9"

    def test_path_fallback_identity_when_wheel_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)
        monkeypatch.setattr(_WHICH, lambda name: f"/opt/homebrew/bin/{name}")
        # path + build fingerprint; the fake path is unstatable so it degrades safely.
        assert engine_build_id().startswith("path:/opt/homebrew/bin/llama-server@")

    def test_unpinned_when_nothing_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)
        monkeypatch.setattr(_WHICH, lambda name: None)
        assert engine_build_id() == "unpinned"

    def test_wheel_without_a_server_file_yields_to_the_path_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stub wheel carries the pin accessor but no llama-server file: the server
        # that launches is the PATH one, so the pin names that one.
        fake = _fake_engine(tmp_path, make_files=False)
        fake.get_engine_pin = lambda: "llama-cpp-0.0.0-stub"
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", fake)
        monkeypatch.setattr(cfg, "llama_server_path", "")
        (tmp_path / "bin").mkdir()
        server = _plant_host_binary(tmp_path / "bin", EngineTool.LLAMA_SERVER)
        monkeypatch.setattr(_WHICH, lambda name: str(server))
        assert resolve_engine_tool(EngineTool.LLAMA_SERVER) == server
        assert engine_build_id().startswith(f"path:{server}@")

    def test_wheel_without_a_server_file_and_no_server_is_unpinned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_engine(tmp_path, make_files=False)
        fake.get_engine_pin = lambda: "llama-cpp-0.0.0-stub"
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", fake)
        monkeypatch.setattr(cfg, "llama_server_path", "")
        monkeypatch.setattr(_WHICH, lambda name: None)
        assert engine_build_id() == "unpinned"

    def test_wheel_without_pin_accessor_reports_its_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _fake_engine(tmp_path, make_files=True)  # no get_engine_pin attribute
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", fake)
        monkeypatch.setattr("lilbee.providers.fleet.binary._pkg_version", lambda name: "0.6.91")
        assert engine_build_id() == "wheel:0.6.91"

    def test_pin_folds_in_the_swap_config_revision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A peer whose llama-swap config is another revision never shares its engine.
        monkeypatch.setattr(binary_mod, "engine_build_id", lambda: "build-x")
        current = engine_pin()
        revision = swap_config_mod.SWAP_CONFIG_REVISION
        monkeypatch.setattr(swap_config_mod, "SWAP_CONFIG_REVISION", revision + 1)
        assert engine_pin() != current
        assert current.endswith(f"|swap-config={revision}")

    def test_a_setting_never_moves_the_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Settings are matched per launch, so no write can make a kept group foreign.
        monkeypatch.setattr(binary_mod, "engine_build_id", lambda: "build-x")
        before = engine_pin()
        monkeypatch.setattr(cfg, "cpu_moe", True)
        monkeypatch.setattr(cfg, "chunk_size", 126)
        monkeypatch.setattr(cfg, "gpu_devices", "1")
        assert engine_pin() == before
        assert before.startswith("build-x|")

    def test_pin_is_stable_for_identical_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The load signature is deterministic: identical config yields the same pin
        # (a jittering pin would make same-setup peers overflow instead of share).
        monkeypatch.setattr(binary_mod, "engine_build_id", lambda: "build-x")
        assert engine_pin() == engine_pin()

    def test_checked_in_pins_match_engine_versions_env(self) -> None:
        import lilbee_engine_llama_cpp

        env = {}
        env_path = Path(__file__).parent.parent / "engine-versions.env"
        for line in env_path.read_text().splitlines():
            if line.startswith("ENGINE_") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
        pin = lilbee_engine_llama_cpp.get_engine_pin()
        assert env["ENGINE_LLAMA_CPP_REF"] in pin
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
        monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", None)
        _plant_host_binary(tmp_path, EngineTool.LLAMA_SERVER)
        monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
        assert engine_build_id() == "unpinned"

    @pytest.mark.real_engine_resolution
    def test_marker_restores_host_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "llama_server_path", "")
        planted = _plant_host_binary(tmp_path, EngineTool.LLAMA_SERVER)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_llama_server() == planted


def test_engine_pin_survives_an_engine_wheel_without_metadata(monkeypatch, tmp_path) -> None:
    """engine_pin runs on every state write, so it must not raise here.

    lilbee_engine_llama_cpp can be importable with nothing to look up: an extracted wheel
    on sys.path, a vendored copy, or a dist whose name does not normalize to
    lilbee-engine-llama-cpp. A PackageNotFoundError escaping here aborts the state write.
    """
    import sys
    from importlib.metadata import PackageNotFoundError

    from lilbee.providers.fleet import binary as binary_mod

    stub = _fake_engine(tmp_path, make_files=True)  # no get_engine_pin -> pre-pin path
    monkeypatch.setitem(sys.modules, "lilbee_engine_llama_cpp", stub)
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "llama_server_path", "", raising=False)

    def _no_metadata(_name: str) -> str:
        raise PackageNotFoundError("lilbee-engine-llama-cpp")

    monkeypatch.setattr(binary_mod, "_pkg_version", _no_metadata)
    assert binary_mod.engine_build_id() == "wheel:unknown"
    assert binary_mod.engine_pin()  # total: a pin is still produced


class TestEngineHelp:
    def test_joins_stdout_and_stderr(self, monkeypatch) -> None:
        monkeypatch.setattr(
            binary_mod.subprocess,
            "run",
            lambda *_a, **_k: SimpleNamespace(stdout="--port PORT", stderr="--no-mmap"),
        )
        assert binary_mod.engine_help(Path("/bin/llama-server")) == "--port PORT\n--no-mmap"

    def test_a_binary_that_hangs_reads_as_empty(self, monkeypatch) -> None:
        def _hang(*_a, **_k):
            raise binary_mod.subprocess.TimeoutExpired("llama-server", 15.0)

        monkeypatch.setattr(binary_mod.subprocess, "run", _hang)
        assert binary_mod.engine_help(Path("/bin/llama-server")) == ""

    def test_a_failed_read_is_retried_on_the_next_call(self, monkeypatch) -> None:
        calls: list[int] = []

        def _flaky(*_a, **_k):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient")
            return SimpleNamespace(stdout="--port PORT", stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", _flaky)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == ""
        assert binary_mod.engine_help(binary) == "--port PORT\n"
        assert len(calls) == 2

    def test_a_blank_read_is_retried_on_the_next_call(self, monkeypatch) -> None:
        outputs = iter(["", "--port PORT"])
        calls: list[int] = []

        def _run(*_a, **_k):
            calls.append(1)
            return SimpleNamespace(stdout=next(outputs), stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", _run)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == ""
        assert binary_mod.engine_help(binary) == "--port PORT\n"
        assert len(calls) == 2

    @staticmethod
    def _clocked_timeouts(monkeypatch) -> list[float]:
        """Run the timeout memory on a fake clock; the list's one item is now."""
        now = [0.0]
        monkeypatch.setattr(binary_mod._help_timeouts, "clock", lambda: now[0])
        return now

    @staticmethod
    def _hang_then_answer(monkeypatch, *, hangs: int) -> list[int]:
        """Make the probe time out *hangs* times, then print ``--port PORT``; the call log."""
        calls: list[int] = []

        def _run(*_a, **_k):
            calls.append(1)
            if len(calls) <= hangs:
                raise binary_mod.subprocess.TimeoutExpired("llama-server", 15.0)
            return SimpleNamespace(stdout="--port PORT", stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", _run)
        return calls

    def test_a_timed_out_probe_is_not_run_again_before_its_expiry(self, monkeypatch) -> None:
        now = self._clocked_timeouts(monkeypatch)
        calls = self._hang_then_answer(monkeypatch, hangs=1)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == ""
        now[0] = binary_mod._HELP_TIMEOUT_MEMORY_S - 1
        assert binary_mod.engine_help(binary) == ""
        assert len(calls) == 1

    def test_a_timed_out_probe_runs_again_after_its_expiry(self, monkeypatch) -> None:
        now = self._clocked_timeouts(monkeypatch)
        calls = self._hang_then_answer(monkeypatch, hangs=1)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == ""
        now[0] = binary_mod._HELP_TIMEOUT_MEMORY_S + 1
        assert binary_mod.engine_help(binary) == "--port PORT\n"
        assert len(calls) == 2

    def test_a_probe_that_times_out_again_is_remembered_for_a_new_period(self, monkeypatch) -> None:
        now = self._clocked_timeouts(monkeypatch)
        calls = self._hang_then_answer(monkeypatch, hangs=2)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == ""
        now[0] = binary_mod._HELP_TIMEOUT_MEMORY_S + 1
        assert binary_mod.engine_help(binary) == ""
        now[0] += binary_mod._HELP_TIMEOUT_MEMORY_S - 1
        assert binary_mod.engine_help(binary) == ""
        assert len(calls) == 2

    def test_a_timeout_on_one_path_does_not_hold_back_another(self, monkeypatch) -> None:
        self._clocked_timeouts(monkeypatch)
        calls = self._hang_then_answer(monkeypatch, hangs=1)
        assert binary_mod.engine_help(Path("/bin/llama-server")) == ""
        assert binary_mod.engine_help(Path("/opt/other/llama-server")) == "--port PORT\n"
        assert len(calls) == 2

    def test_a_good_read_stays_cached_long_after_the_expiry(self, monkeypatch) -> None:
        now = self._clocked_timeouts(monkeypatch)
        calls = self._hang_then_answer(monkeypatch, hangs=0)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == "--port PORT\n"
        now[0] = 100 * binary_mod._HELP_TIMEOUT_MEMORY_S
        assert binary_mod.engine_help(binary) == "--port PORT\n"
        assert len(calls) == 1

    def test_a_good_read_runs_the_binary_once_per_path(self, monkeypatch) -> None:
        calls: list[int] = []

        def _run(*_a, **_k):
            calls.append(1)
            return SimpleNamespace(stdout="--port PORT", stderr="")

        monkeypatch.setattr(binary_mod.subprocess, "run", _run)
        binary = Path("/bin/llama-server")
        assert binary_mod.engine_help(binary) == binary_mod.engine_help(binary)
        assert len(calls) == 1


@pytest.mark.parametrize(
    ("help_text", "flag", "listed"),
    [
        ("-lm,   --load-mode MODE   loading mode", "--load-mode", True),
        ("--mmap, --no-mmap   whether to memory-map", "--no-mmap", True),
        ("-ngl,  --n-gpu-layers N   see --load-mode", "--load-mode", False),
        ("        (with --load-mode none the load is slower)", "--load-mode", False),
        ("--load-mode-extra   a longer flag", "--load-mode", False),
        ("", "--load-mode", False),
        ("  -t,    --threads N        threads", "--threads", True),
        ("         --jinja            use the template", "--jinja", True),
        ("                                  --jinja is quoted here", "--jinja", False),
        ("         --embedding(s)     embeddings only", "--embeddings", True),
        ("         --embedding(s)     embeddings only", "--embedding", True),
    ],
    ids=[
        "short-and-long",
        "pair",
        "in-description",
        "continuation",
        "longer-flag",
        "empty",
        "indented-short-and-long",
        "indented-long-only",
        "indented-description",
        "plural-spelling-plural",
        "plural-spelling-singular",
    ],
)
def test_help_lists_option_reads_option_names_only(help_text: str, flag: str, listed: bool) -> None:
    assert binary_mod.help_lists_option(help_text, flag) is listed
