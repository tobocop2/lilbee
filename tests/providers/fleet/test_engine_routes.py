"""Tests for the ways a build reaches the engine store: pack, download, package argv."""

from __future__ import annotations

import hashlib
import stat
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import httpx
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

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
from tests.providers.fleet.test_engine_store import make_wheel

REVISION = "c" * 40


@pytest.fixture
def wheel(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "fx.whl"
    return path, make_wheel(path)


@pytest.fixture
def store(tmp_path: Path) -> EngineStore:
    return EngineStore(tmp_path / "engines")


def lock_of(sha: str, size: int = 1) -> PinLock:
    asset = PlatformAsset("owner/engines", REVISION, "fx.whl", sha, size, None)
    return PinLock("fx", "9", "cpu", {"linux-x86_64": asset}, ("MIT",), sha)


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
    lock = PinLock("fx", "9", "cpu", {}, ("GPL-3.0-only",), "a" * 64)
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


def test_fetch_frozen_downloads_at_the_pinned_commit(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def downloader(**kwargs: str) -> str:
        seen.update(kwargs)
        return str(tmp_path / "fx.whl")

    got = fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path, downloader=downloader)
    assert got == tmp_path / "fx.whl"
    assert seen == {
        "repo_id": "owner/engines",
        "revision": REVISION,
        "filename": "fx.whl",
        "local_dir": str(tmp_path),
    }


def test_fetch_frozen_for_a_platform_without_a_build_is_refused() -> None:
    with pytest.raises(UnsupportedPlatformError, match="no build for macos-arm64"):
        fetch_frozen(lock_of("a" * 64), "macos-arm64", dest=Path("d"), downloader=mock.Mock())


def test_fetch_frozen_checks_the_disk_before_the_download(tmp_path: Path) -> None:
    downloader = mock.Mock()
    with (
        mock.patch(
            "lilbee.providers.fleet.engine_store.disk_shortfall",
            return_value="Not enough disk space",
        ) as shortfall,
        pytest.raises(EngineDiskError, match="Not enough disk space"),
    ):
        fetch_frozen(
            lock_of("a" * 64, size=7), "linux-x86_64", dest=tmp_path, downloader=downloader
        )
    shortfall.assert_called_once_with(tmp_path, "owner/engines", 7)
    downloader.assert_not_called()


def test_fetch_frozen_downloads_when_the_disk_has_room(tmp_path: Path) -> None:
    downloader = mock.Mock(return_value=str(tmp_path / "fx.whl"))
    got = fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path, downloader=downloader)
    assert got == tmp_path / "fx.whl"
    downloader.assert_called_once()


def test_fetch_frozen_retries_a_dropped_connection(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(**_kwargs: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        return str(tmp_path / "fx.whl")

    with mock.patch.object(dl.time, "sleep"):
        got = fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path, downloader=flaky)
    assert (got, calls["n"]) == (tmp_path / "fx.whl", 2)


def test_fetch_frozen_keeps_failing_as_missing_pack_through_ensure(store: EngineStore) -> None:
    def always_down(**_kwargs: str) -> str:
        raise OSError("disk fault")

    lock = lock_of("a" * 64)
    with (
        mock.patch.object(dl.time, "sleep"),
        pytest.raises(MissingPackError, match=r"Copy fx-9-cpu\.whl"),
    ):
        store.ensure(
            lock,
            lambda lk, dest: fetch_frozen(lk, "linux-x86_64", dest=dest, downloader=always_down),
        )


def test_fetch_frozen_does_not_retry_a_defect(tmp_path: Path) -> None:
    boom = mock.Mock(side_effect=KeyError("bug"))
    with pytest.raises(KeyError):
        fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path, downloader=boom)
    boom.assert_called_once()


def test_default_downloader_is_the_hub_call(tmp_path: Path) -> None:
    with mock.patch("huggingface_hub.hf_hub_download", return_value=str(tmp_path / "w.whl")) as hub:
        got = fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path / "dl")
    hub.assert_called_once_with(
        repo_id="owner/engines",
        revision=REVISION,
        filename="fx.whl",
        local_dir=str(tmp_path / "dl"),
    )
    assert got == tmp_path / "w.whl"


def test_uv_argv_installs_pinned_hashes_into_the_stage_from_the_backend_index(
    tmp_path: Path,
) -> None:
    lock = PinLock("llama.cpp", "1", "cu125", {}, ("MIT",), "A" * 64)
    argv = pip_install_argv(lock, tmp_path / "stage")
    assert argv[:3] == ["uv", "pip", "install"]
    assert argv[argv.index("--target") + 1] == str(tmp_path / "stage")
    assert argv[argv.index("--index-url") + 1] == "https://lilbee.sh/cu125/"
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
    assert package_name(PinLock(engine, "1", "cpu", {}, ("MIT",), "a" * 64)) == name


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


def test_default_download_writes_into_the_scratch_folder_not_the_hub_cache(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    seen: dict[str, str] = {}

    def hub(**kwargs: str) -> str:
        seen.update(kwargs)
        return str(path)

    store.ensure(
        lock_of(sha), lambda lk, dest: fetch_frozen(lk, "linux-x86_64", dest=dest, downloader=hub)
    )
    assert Path(seen["local_dir"]).parent == store.root


def permanent(kind: type[Exception]) -> Exception:
    if kind is EntryNotFoundError:
        return kind("gone")
    response = httpx.Response(404, request=httpx.Request("GET", "https://hub.example/x"))
    return kind("gone", response=response)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kind", [EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError]
)
def test_a_missing_file_repo_or_revision_is_a_missing_pack_after_one_call(
    store: EngineStore, kind: type[Exception]
) -> None:
    calls = {"n": 0}

    def gone(**_kwargs: str) -> str:
        calls["n"] += 1
        raise permanent(kind)

    with (
        mock.patch.object(dl.time, "sleep") as slept,
        pytest.raises(MissingPackError, match=r"Copy fx-9-cpu\.whl"),
    ):
        store.ensure(
            lock_of("a" * 64),
            lambda lk, dest: fetch_frozen(lk, "linux-x86_64", dest=dest, downloader=gone),
        )
    assert calls["n"] == 1
    slept.assert_not_called()


def test_fetch_frozen_downloads_into_the_destination(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def downloader(**kwargs: str) -> str:
        seen.update(kwargs)
        return str(tmp_path / "fx.whl")

    fetch_frozen(lock_of("a" * 64), "linux-x86_64", dest=tmp_path, downloader=downloader)
    assert seen["local_dir"] == str(tmp_path)
