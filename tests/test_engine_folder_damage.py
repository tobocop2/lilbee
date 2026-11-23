"""One damaged folder of the engine store hides no other engine, and the list names it."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.core.config import cfg
from lilbee.providers.fleet import engine_manifest, engine_store
from lilbee.providers.fleet.engine_store import (
    RECORD_NAME,
    EngineFolderError,
    EngineStore,
    PinLock,
    engines_root,
    prune_engines,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.engines.tei import TeiAdapter
from lilbee.providers.fleet.host_caps import collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests._unreadable import POSIX_DENIES_READS, unreadable

GOOD = ["ik_llama.cpp", "mistral.rs"]
ENGINE = "tei"
BUILD = f"{ENGINE}/1-cpu"
DENIED_IN_BUILD = [BUILD, f"{BUILD}/bin", f"{BUILD}/{RECORD_NAME}"]
DENIED = [(ENGINE, ENGINE), *((name, BUILD) for name in DENIED_IN_BUILD)]


def _error(folder: Path, install: str = " and run 'lilbee engine install tei'") -> str:
    return (
        f"The engine folder {folder} holds no build that lilbee can read. If its "
        f"permissions deny a read, correct them. lilbee cannot remove it until then. "
        f"If they do not, remove it{install}."
    )


def _write_build(
    root: Path, engine: str, version: str = "1", backend: str = "cpu", tool: str = "server"
) -> Path:
    """A complete build of *engine* under *root*; returns its folder."""
    build = root / engine / f"{version}-{backend}"
    (build / "bin").mkdir(parents=True)
    server = build / "bin" / tool
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    record = {
        "engine": engine,
        "version": version,
        "backend": backend,
        "sha256": "0" * 64,
        "licences": ["MIT"],
        "installed_at": "2026-01-01T00:00:00+00:00",
        "binary": f"bin/{tool}",
    }
    (build / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    return build


def _file_in_place(folder: Path) -> None:
    shutil.rmtree(folder)
    folder.write_text("not a folder", encoding="utf-8")


def _folder_for_record(build: Path) -> None:
    (build / RECORD_NAME).unlink()
    (build / RECORD_NAME).mkdir()


def _folder_for_binary(build: Path) -> None:
    (build / "bin" / "server").unlink()
    (build / "bin" / "server").mkdir()


def _loop_for_binary(build: Path) -> None:
    (build / "bin" / "server").unlink()
    (build / "bin" / "server").symlink_to("other")
    (build / "bin" / "other").symlink_to("server")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A store with two good engines and tei, the engine to damage."""
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    monkeypatch.setenv("LILBEE_MCP_ENGINES_ENABLED", "true")
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "mcp_engines_enabled", True)
    # No shipped pin names these builds; the startup prune would remove them.
    monkeypatch.setattr(engine_store, "prune_engines", lambda _models: [])
    monkeypatch.setattr(choice, "host_answer", lambda *_a: PASS)
    flags = {"flags": ["avx2", "avx512f"]}
    host = collect(system="Linux", machine="x86_64", cpu_info=flags, nvidia=None)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    # The host above is Linux on every computer, so its glibc is set here too.
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    store = engines_root(cfg.models_dir)
    for engine in (*GOOD, ENGINE):
        _write_build(store, engine)
    return store


def _cli_json(*argv: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return result.exit_code, json.loads((result.output.strip() or "{}").splitlines()[-1])


def _rest_list() -> tuple[int, dict[str, Any]]:
    with create_test_client([routes.engine_builds_route], debug=False) as client:
        response = client.get("/api/engines")
    return response.status_code, response.json()


def _assert_every_surface_names(folder: Path) -> None:
    """Every form of the list shows the good engines, and names *folder* with its fix."""
    error = _error(folder)
    code, cli = _cli_json("list")
    assert code == 0
    status, rest = _rest_list()
    assert status == 200
    for body in (cli, rest, mcp_server.engine_list_tool()):
        assert [build["engine"] for build in body["builds"]] == GOOD
        assert body.get("damaged") == [{"folder": str(folder), "error": error}]
    text = CliRunner().invoke(cli_app, ["engine", "list"])
    assert (text.exit_code, text.output.splitlines().count(error)) == (0, 1)
    chat = run_engine_command("list").text.splitlines()
    assert [line.split()[0] for line in chat[: len(GOOD)]] == GOOD
    assert chat[len(GOOD)] == error
    rows = engine_service.engine_rows()
    assert [row.engine for row in rows if row.version] == GOOD


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda b: (b / RECORD_NAME).unlink(), id="no-manifest"),
        pytest.param(
            lambda b: (b / RECORD_NAME).write_text("{not json", encoding="utf-8"), id="not-json"
        ),
        pytest.param(
            lambda b: (b / RECORD_NAME).write_text("{}", encoding="utf-8"), id="missing-key"
        ),
        pytest.param(_folder_for_record, id="folder-for-record"),
        pytest.param(_file_in_place, id="file-in-place"),
        pytest.param(_folder_for_binary, id="folder-for-binary"),
        pytest.param(
            _loop_for_binary,
            id="loop-for-binary",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="symlinks need privileges on Windows"
            ),
        ),
    ],
)
def test_the_list_names_a_damaged_build(root: Path, damage: Callable[[Path], object]) -> None:
    damage(root / BUILD)
    _assert_every_surface_names(root / BUILD)


def test_the_list_names_a_file_in_place_of_an_engine_folder(root: Path) -> None:
    _file_in_place(root / ENGINE)
    _assert_every_surface_names(root / ENGINE)


@POSIX_DENIES_READS
@pytest.mark.parametrize("name", DENIED_IN_BUILD)
def test_the_list_names_a_build_that_denies_a_read(root: Path, name: str) -> None:
    with unreadable(root / name):
        _assert_every_surface_names(root / BUILD)


@POSIX_DENIES_READS
@pytest.mark.parametrize("mode", [0o000, 0o444])
def test_the_list_names_an_engine_folder_that_denies_a_read(root: Path, mode: int) -> None:
    with unreadable(root / ENGINE):
        (root / ENGINE).chmod(mode)
        _assert_every_surface_names(root / ENGINE)


def test_the_list_names_an_engine_folder_that_cannot_be_listed_on_any_platform(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.listdir

    def listdir(path: Any = ".") -> list[str]:
        if not isinstance(path, int) and Path(path) == root / ENGINE:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(os, "listdir", listdir)
    _assert_every_surface_names(root / ENGINE)


def _fail_listing(monkeypatch: pytest.MonkeyPatch, folder: Path, code: int) -> None:
    real = os.listdir

    def listdir(path: Any = ".") -> list[str]:
        if not isinstance(path, int) and Path(path) == folder:
            raise OSError(code, os.strerror(code), str(path))
        return real(path)

    monkeypatch.setattr(os, "listdir", listdir)


def _fail_record_read(monkeypatch: pytest.MonkeyPatch, build: Path, code: int) -> None:
    real = Path.read_text

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == build / RECORD_NAME:
            raise OSError(code, os.strerror(code), str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


@pytest.mark.parametrize(
    ("fail", "named", "code"),
    [
        pytest.param(_fail_listing, ENGINE, errno.EMFILE, id="listing-emfile"),
        pytest.param(_fail_record_read, BUILD, errno.EIO, id="record-eio"),
        pytest.param(_fail_record_read, BUILD, errno.EMFILE, id="record-emfile"),
    ],
)
def test_a_read_that_fails_for_no_permission_keeps_the_build_and_names_the_fault(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: Callable[[pytest.MonkeyPatch, Path, int], None],
    named: str,
    code: int,
) -> None:
    fail(monkeypatch, root / named, code)
    error = (
        f"A read of the engine folder {root / named} failed: {os.strerror(code)}. "
        f"lilbee leaves the folder as it is. Run the command again."
    )
    listed, body = _cli_json("list")
    assert (listed, body["damaged"]) == (0, [{"folder": str(root / named), "error": error}])
    removed, answer = _cli_json("remove", ENGINE, "1", "cpu")
    assert (removed, answer.get("error")) == (1, error)
    assert [entry.name for entry in os.scandir(root / ENGINE)] == ["1-cpu"]


def test_the_folders_that_hold_no_engine_are_not_named(root: Path) -> None:
    (root / "router" / "v1").mkdir(parents=True)
    (root / "lemonade-data" / "models").mkdir(parents=True)
    (root / "desktop.ini").write_text("x", encoding="utf-8")
    (root / ENGINE / ".stage-abc").mkdir()
    for stray in ("Thumbs.db", ".DS_Store", "tei-1-py3-none-any.whl"):
        (root / ENGINE / stray).write_text("x", encoding="utf-8")
    (root / "packs").mkdir()
    (root / "packs" / "tei-1-cpu.whl").write_text("pack", encoding="utf-8")
    (root / "mlx").mkdir()
    code, body = _cli_json("list")
    assert (code, body["damaged"]) == (0, [])
    assert [build["engine"] for build in body["builds"]] == [*GOOD, ENGINE]


@POSIX_DENIES_READS
def test_another_engine_is_removed_beside_a_denied_folder(root: Path) -> None:
    with unreadable(root / ENGINE):
        code, body = _cli_json("remove", "mistral.rs", "1", "cpu")
    assert (code, body["removed"]) == (0, "mistral.rs 1 (cpu)")
    assert not (root / "mistral.rs" / "1-cpu").exists()


@POSIX_DENIES_READS
@pytest.mark.parametrize(("denied", "named"), DENIED)
def test_remove_of_a_build_that_denies_a_read_names_the_folder(
    root: Path, denied: str, named: str
) -> None:
    (root / denied).chmod(0)
    try:
        code, body = _cli_json("remove", ENGINE, "1", "cpu")
    finally:
        # A removal that does not refuse moves the build, so the path is read again.
        (root / ENGINE).chmod(0o700)
        for build in (root / ENGINE).iterdir():
            build.chmod(0o700)
            (build / "bin").chmod(0o700)
    assert [entry.name for entry in (root / ENGINE).iterdir()] == ["1-cpu"]
    assert (code, body.get("error")) == (1, _error(root / named))


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda b: _file_in_place(b / "bin"), id="file-for-bin"),
        pytest.param(_folder_for_record, id="folder-for-record"),
    ],
)
def test_remove_clears_a_damaged_build_that_denies_no_read(
    root: Path, damage: Callable[[Path], object]
) -> None:
    damage(root / BUILD)
    code, body = _cli_json("remove", ENGINE, "1", "cpu")
    assert (code, body.get("removed")) == (0, "tei 1 (cpu)")
    assert list((root / ENGINE).iterdir()) == []


def _no_fetch(_lock: PinLock, _scratch: Path) -> Path:
    raise AssertionError("the pack was downloaded")


def _install(root: Path, named: str) -> None:
    lock = PinLock(ENGINE, "1", "cpu", {}, ("MIT",), "1" * 64, "server")
    with pytest.raises(OSError, match=re.escape(_error(root / named))):
        EngineStore(root).ensure(lock, _no_fetch)


@POSIX_DENIES_READS
@pytest.mark.parametrize(("denied", "named"), DENIED)
def test_an_install_into_a_denied_folder_names_it_before_the_download(
    root: Path, denied: str, named: str
) -> None:
    with unreadable(root / denied):
        _install(root, named)


def test_an_install_into_a_file_names_it_before_the_download(root: Path) -> None:
    _file_in_place(root / ENGINE)
    _install(root, ENGINE)


@POSIX_DENIES_READS
def test_the_list_prints_the_fix_of_a_denied_bin_at_the_pin_once(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pin = TeiAdapter().pin_record()
    build = _write_build(root, ENGINE, pin["version"], pin["backend"])
    monkeypatch.setattr(engine_store, "prune_engines", prune_engines)
    with unreadable(build / "bin"), caplog.at_level("WARNING"):
        output = CliRunner().invoke(cli_app, ["engine", "list"]).output
    assert (output + caplog.text).count(_error(build)) == 1
    assert f"Engine tei counts as not installed: cannot read {build}" in caplog.messages


@pytest.mark.parametrize(
    ("engine", "ok"), [("mlx", True), ("lemonade", True), ("router", True), (ENGINE, False)]
)
def test_the_fix_names_no_install_that_lilbee_refuses(
    root: Path, monkeypatch: pytest.MonkeyPatch, engine: str, ok: bool
) -> None:
    monkeypatch.setattr(choice, "install_answer", lambda *_a: HostGate(ok))
    assert str(EngineFolderError(root / engine, engine)) == _error(root / engine, install="")


def test_the_rows_come_from_the_list_that_the_tab_read(root: Path) -> None:
    listed = engine_service.EngineListView([], [], [], root)
    assert [row.engine for row in engine_service.engine_rows(listed=listed) if row.version] == []


@POSIX_DENIES_READS
@pytest.mark.parametrize(("denied", "named"), DENIED)
def test_a_command_warns_once_of_an_engine_folder_that_the_prune_skips(
    root: Path, caplog: pytest.LogCaptureFixture, denied: str, named: str
) -> None:
    (root / "desktop.ini").write_text("x", encoding="utf-8")
    (root / "router").mkdir()
    with unreadable(root / "router"), unreadable(root / denied), caplog.at_level("WARNING"):
        prune_engines(cfg.models_dir)
    folder = root / named
    assert caplog.messages == [
        f"The engine prune cannot read {folder}. Run 'lilbee engine list' for the fix."
    ]


@pytest.mark.parametrize(
    ("fail", "named"),
    [
        pytest.param(_fail_listing, ENGINE, id="engine-folder"),
        pytest.param(_fail_record_read, BUILD, id="build-record"),
    ],
)
def test_the_prune_warns_of_a_folder_that_cannot_be_read_on_any_platform(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fail: Callable[[pytest.MonkeyPatch, Path, int], None],
    named: str,
) -> None:
    fail(monkeypatch, root / named, errno.EACCES)
    with caplog.at_level("WARNING"):
        prune_engines(cfg.models_dir)
    assert caplog.messages == [
        f"The engine prune cannot read {root / named}. Run 'lilbee engine list' for the fix."
    ]
    assert (root / BUILD).is_dir()


@POSIX_DENIES_READS
@pytest.mark.parametrize("denied", [ENGINE, *DENIED_IN_BUILD])
def test_prune_clears_the_other_engines_beside_a_denied_folder(root: Path, denied: str) -> None:
    with unreadable(root / denied):
        gone = EngineStore(root).prune({("mistral.rs", "1", "cpu")}, in_use=lambda *_: False)
    assert gone == [root / "ik_llama.cpp" / "1-cpu"]
    assert (root / BUILD).is_dir()


def test_the_prune_reads_no_pack_no_hidden_folder_and_no_file(root: Path) -> None:
    builds = [path for *_, path in EngineStore(root).build_dirs()]
    for folder in ("packs", ".locks"):
        (root / folder / "1-cpu").mkdir(parents=True)
    (root / ENGINE / "2-cpu").write_text("x", encoding="utf-8")
    assert [path for *_, path in EngineStore(root).build_dirs()] == builds


@POSIX_DENIES_READS
def test_a_launch_reads_past_the_denied_folder_of_another_engine(root: Path) -> None:
    pin = TeiAdapter().pin_record()
    build = _write_build(root, ENGINE, pin["version"], pin["backend"], tool="llama-server")
    with unreadable(root / "mistral.rs"):
        assert TeiAdapter().binary() == build / "bin" / "llama-server"
