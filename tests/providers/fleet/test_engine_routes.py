"""Tests for the ways a build reaches the engine store: pack, download, package argv."""

from __future__ import annotations

import hashlib
import stat
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest
from huggingface_hub import constants as hub_constants

from lilbee.catalog import download as dl
from lilbee.providers.fleet.engine_store import (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineStore,
    InstallTool,
    LicenceRefusedError,
    MissingPackError,
    PinLock,
    PlatformAsset,
    UnsafeWheelError,
    UnsupportedPlatformError,
    fetch_frozen,
    package_name,
    pip_install_argv,
    pip_requirement,
)
from tests._release_server import release_server
from tests.providers.fleet.test_engine_store import make_wheel

RELEASE = "https://github.com/tobocop2/lilbee-engines/releases/download/fx-9-1/fx.whl"
CELL = "linux-x86_64-cpu"


@pytest.fixture
def wheel(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "fx.whl"
    return path, make_wheel(path)


@pytest.fixture
def store(tmp_path: Path) -> EngineStore:
    return EngineStore(tmp_path / "engines")


@pytest.fixture(autouse=True)
def hub_online(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite switches the hub's network off; the download tests use a loopback server."""
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)


def lock_of(sha: str, size: int = 1, url: str = RELEASE) -> PinLock:
    asset = PlatformAsset(url, sha, size, None)
    return PinLock("fx", "9", "cpu", {CELL: asset}, ("MIT",), sha, "server")


def no_network(_lock: PinLock, _dest: Path) -> Path:
    raise ConnectionError("down")


def never_called(_lock: PinLock, _dest: Path) -> Path:
    raise AssertionError("the build is present, nothing is fetched")


def test_missing_pack_error_names_engine_pin_and_pack_directory(store: EngineStore) -> None:
    with pytest.raises(MissingPackError) as caught:
        store.ensure(lock_of("a" * 64), no_network)
    text = str(caught.value)
    assert "Engine fx 9 (cpu) is not installed and cannot be downloaded." in text
    assert f"Copy fx-9-cpu.whl into {store.root / 'packs'}" in text
    assert [p.name for p in store.root.iterdir()] == [".locks"]
    assert [p for p in store.root.rglob("*") if p.is_file()] == []


def test_pack_copied_by_hand_installs_without_the_network(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.pack_path(lock_of(sha)).parent.mkdir(parents=True)
    store.pack_path(lock_of(sha)).write_bytes(path.read_bytes())
    built = store.ensure(lock_of(sha), no_network)
    assert built == store.root / "fx" / "9-cpu"
    assert [found.path for found in store.installed()] == [built]


def test_installed_build_is_returned_before_pack_or_download(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    built = store.install_from_wheel(lock_of(sha), path)
    assert store.ensure(lock_of(sha), never_called) == built


def test_download_goes_through_the_wheel_checks(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    assert store.ensure(lock_of(sha), lambda _lock, _dest: path) == store.root / "fx" / "9-cpu"


def _remove_the_tool(binary: Path) -> None:
    binary.unlink()


def _clear_the_exec_bit(binary: Path) -> None:
    binary.chmod(0o644)


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_remove_the_tool, id="tool-removed"),
        pytest.param(
            _clear_the_exec_bit,
            id="exec-bit-cleared",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="Windows has no exec bit, and the store skips that check there",
            ),
        ),
    ],
)
def test_damaged_build_is_fetched_and_replaced(
    store: EngineStore, wheel: tuple[Path, str], damage: Callable[[Path], None]
) -> None:
    path, sha = wheel
    built = store.install_from_wheel(lock_of(sha), path)
    damage(built / "lilbee_engine_fx" / "bin" / "server")
    assert store.installed() == []
    fetched: list[PinLock] = []

    def fetch(lock: PinLock, _dest: Path) -> Path:
        fetched.append(lock)
        return path

    assert store.ensure(lock_of(sha), fetch) == built
    assert len(fetched) == 1
    assert len(store.installed()) == 1


def test_installed_build_of_another_wheel_conflicts_before_any_fetch(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.install_from_wheel(lock_of(sha), path)
    with pytest.raises(EngineConflictError, match="different wheel"):
        store.ensure(lock_of("b" * 64), never_called)


def test_licence_outside_the_list_is_refused_before_any_download(store: EngineStore) -> None:
    lock = PinLock("fx", "9", "cpu", {}, ("GPL-3.0-only",), "a" * 64, "server")
    with pytest.raises(LicenceRefusedError, match=r"GPL-3\.0-only"):
        store.ensure(lock, never_called)
    assert store.installed() == []


def test_pack_with_the_wrong_digest_is_refused_and_leaves_nothing(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, _sha = wheel
    lock = lock_of("0" * 64)
    store.pack_path(lock).parent.mkdir(parents=True)
    store.pack_path(lock).write_bytes(path.read_bytes())
    with pytest.raises(ChecksumError):
        store.ensure(lock, no_network)
    assert not (store.root / "fx").exists()


def test_downloaded_wheel_with_an_escaping_entry_is_refused(
    store: EngineStore, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as archive:
        info = zipfile.ZipInfo("../victim")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "x")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(UnsafeWheelError, match="outside the engine directory"):
        store.ensure(lock_of(sha), lambda _lock, _dest: bad)
    assert store.installed() == []
    assert not (tmp_path / "victim").exists()


def test_fetch_frozen_downloads_the_url_of_the_cell(tmp_path: Path) -> None:
    with release_server() as server:
        server.files["/fx.whl"] = b"wheel bytes"
        lock = lock_of("a" * 64, size=11, url=f"{server.url}/fx.whl")
        got = fetch_frozen(lock, CELL, dest=tmp_path / "dl")
    assert got == tmp_path / "dl" / "fx.whl"
    assert got.read_bytes() == b"wheel bytes"


def test_fetch_frozen_for_a_cell_without_a_build_is_refused_before_any_request(
    tmp_path: Path,
) -> None:
    with release_server() as server:
        lock = lock_of("a" * 64, url=f"{server.url}/fx.whl")
        with pytest.raises(UnsupportedPlatformError, match="no build for macos-arm64-metal"):
            fetch_frozen(lock, "macos-arm64-metal", dest=tmp_path)
    assert server.requests == []


def test_a_pin_with_no_cells_has_no_download_and_ensure_asks_for_a_pack(
    store: EngineStore,
) -> None:
    lock = PinLock("fx", "9", "cpu", {}, ("MIT",), "a" * 64, "server")
    with pytest.raises(MissingPackError, match=r"Copy fx-9-cpu\.whl"):
        store.ensure(lock, lambda pin, dest: fetch_frozen(pin, CELL, dest=dest))


def test_fetch_frozen_checks_the_disk_before_the_download(tmp_path: Path) -> None:
    with release_server() as server:
        lock = lock_of("a" * 64, size=7, url=f"{server.url}/fx.whl")
        with (
            mock.patch(
                "lilbee.providers.fleet.engine_store.disk_shortfall",
                return_value="Not enough disk space",
            ) as shortfall,
            pytest.raises(EngineDiskError, match="Not enough disk space"),
        ):
            fetch_frozen(lock, CELL, dest=tmp_path)
    shortfall.assert_called_once_with(tmp_path, "fx.whl", 7)
    assert server.requests == []


def test_a_server_that_stays_down_is_a_missing_pack_through_ensure(store: EngineStore) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep"):
        server.fail_next = 10
        lock = lock_of("a" * 64, url=f"{server.url}/fx.whl")
        with pytest.raises(MissingPackError, match=r"Copy fx-9-cpu\.whl"):
            store.ensure(lock, lambda pin, dest: fetch_frozen(pin, CELL, dest=dest))
    assert len(server.requests) == 3


def test_a_missing_release_file_is_a_missing_pack_after_one_request(store: EngineStore) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep") as slept:
        lock = lock_of("a" * 64, url=f"{server.url}/gone.whl")
        with pytest.raises(MissingPackError, match=r"Copy fx-9-cpu\.whl"):
            store.ensure(lock, lambda pin, dest: fetch_frozen(pin, CELL, dest=dest))
    assert len(server.requests) == 1
    slept.assert_not_called()


def test_uv_argv_installs_pinned_hashes_into_the_stage_from_the_engines_index(
    tmp_path: Path,
) -> None:
    lock = PinLock("llama.cpp", "1", "cu125", {}, ("MIT",), "A" * 64, "server")
    argv = pip_install_argv(lock, tmp_path / "stage")
    assert argv[:3] == ["uv", "pip", "install"]
    assert argv[argv.index("--target") + 1] == str(tmp_path / "stage")
    assert argv[argv.index("--index-url") + 1] == "https://lilbee.sh/engines/simple/"
    assert {"--no-deps", "--require-hashes", "--only-binary=:all:"} <= set(argv)
    assert argv[-2:] == ["-r", "-"]
    assert pip_requirement(lock) == f"lilbee-engine-llama-cpp==1 --hash=sha256:{'a' * 64}\n"


def test_pip_argv_runs_pip_through_the_current_interpreter(tmp_path: Path) -> None:
    argv = pip_install_argv(lock_of("a" * 64), tmp_path, tool=InstallTool.PIP)
    assert argv[1:4] == ["-m", "pip", "install"]


@pytest.mark.parametrize(
    ("engine", "name"),
    [("llama.cpp", "lilbee-engine-llama-cpp"), ("mistralrs", "lilbee-engine-mistralrs")],
)
def test_package_name_is_the_index_name_of_the_engine(engine: str, name: str) -> None:
    assert package_name(PinLock(engine, "1", "cpu", {}, ("MIT",), "a" * 64, "server")) == name


def scratch_dirs(store: EngineStore) -> list[str]:
    return sorted(p.name for p in store.root.iterdir() if p.name.startswith(".download-"))


def test_download_lands_under_the_engines_root_and_nothing_stays(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    seen: dict[str, Path] = {}

    def fetch(_lock: PinLock, dest: Path) -> Path:
        seen["dest"] = dest
        dest.mkdir(parents=True)
        held = dest / "fx.whl"
        held.write_bytes(path.read_bytes())
        return held

    built = store.ensure(lock_of(sha), fetch)
    assert seen["dest"].parent == store.root
    assert not seen["dest"].exists()
    assert scratch_dirs(store) == []
    assert built == store.root / "fx" / "9-cpu"


def test_refused_download_leaves_no_scratch_folder(store: EngineStore, tmp_path: Path) -> None:
    def fetch(_lock: PinLock, dest: Path) -> Path:
        dest.mkdir(parents=True)
        held = dest / "fx.whl"
        held.write_bytes(wheel_bytes)
        return held

    other = tmp_path / "o.whl"
    make_wheel(other)
    wheel_bytes = other.read_bytes()
    with pytest.raises(ChecksumError):
        store.ensure(lock_of("0" * 64), fetch)
    assert scratch_dirs(store) == []


def test_failed_download_leaves_no_scratch_folder(store: EngineStore) -> None:
    def fetch(_lock: PinLock, dest: Path) -> Path:
        dest.mkdir(parents=True)
        (dest / "partial").write_bytes(b"x")
        raise ConnectionError("down")

    with pytest.raises(MissingPackError):
        store.ensure(lock_of("a" * 64), fetch)
    assert scratch_dirs(store) == []


def test_download_from_a_release_url_installs_through_the_store(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    seen: list[Path] = []

    def fetch(pin: PinLock, dest: Path) -> Path:
        seen.append(dest)
        return fetch_frozen(pin, CELL, dest=dest)

    with release_server() as server:
        server.files["/fx.whl"] = path.read_bytes()
        lock = lock_of(sha, size=path.stat().st_size, url=f"{server.url}/fx.whl")
        built = store.ensure(lock, fetch)
    assert built == store.root / "fx" / "9-cpu"
    assert seen[0].parent == store.root
    assert scratch_dirs(store) == []


def test_download_of_a_wheel_with_a_bad_digest_is_refused_and_leaves_no_build(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, _sha = wheel
    with release_server() as server:
        server.files["/fx.whl"] = path.read_bytes()
        lock = lock_of("0" * 64, size=path.stat().st_size, url=f"{server.url}/fx.whl")
        with pytest.raises(ChecksumError, match="does not match its pin"):
            store.ensure(lock, lambda pin, dest: fetch_frozen(pin, CELL, dest=dest))
    assert not (store.root / "fx").exists()
    assert scratch_dirs(store) == []


def two_cells(second_sha: str, url: str = RELEASE, size: int = 1) -> PinLock:
    """A pin whose top digest is the first cell's; the second cell holds *second_sha*."""
    first = PlatformAsset(RELEASE, "f" * 64, 1, None)
    second = PlatformAsset(url, second_sha, size, None)
    return PinLock(
        "fx",
        "9",
        "cpu",
        {CELL: first, "macos-arm64-cpu": second},
        ("MIT",),
        first.sha256,
        "server",
    )


def test_the_second_cell_downloads_its_own_wheel_and_records_its_own_digest(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    with release_server() as server:
        server.files["/fx.whl"] = path.read_bytes()
        lock = two_cells(sha, f"{server.url}/fx.whl", path.stat().st_size)
        built = store.ensure(
            lock.for_cell("macos-arm64-cpu"),
            lambda pin, scratch: fetch_frozen(lock, "macos-arm64-cpu", dest=scratch),
        )
    assert [found.sha256 for found in store.installed()] == [sha]
    assert store.complete_build(lock.for_cell("macos-arm64-cpu")) == built


def test_the_pack_of_the_second_cell_installs_with_the_digest_of_that_cell(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    lock = two_cells(sha).for_cell("macos-arm64-cpu")
    store.pack_path(lock).parent.mkdir(parents=True)
    store.pack_path(lock).write_bytes(path.read_bytes())
    built = store.ensure(lock, no_network)
    assert [found.sha256 for found in store.installed()] == [sha]
    assert built == store.root / "fx" / "9-cpu"


def test_the_pack_of_the_second_cell_is_refused_under_the_top_digest(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    lock = two_cells(sha)
    store.pack_path(lock).parent.mkdir(parents=True)
    store.pack_path(lock).write_bytes(path.read_bytes())
    with pytest.raises(ChecksumError):
        store.ensure(lock, no_network)
    assert store.installed() == []
