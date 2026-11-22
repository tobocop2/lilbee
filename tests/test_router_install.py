"""The llama-swap router installs from its pinned release for any engine, not only llama.cpp."""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.core.config import Config, cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import first_use, router_install
from lilbee.providers.fleet.binary import EngineTool, resolve_llama_swap
from lilbee.providers.fleet.engine_store import (
    PACK_DIR,
    STAGE_PREFIX,
    ChecksumError,
    EngineDiskError,
    EngineStore,
    engines_root,
)
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.fleet.router_install import (
    LICENCE_FILE,
    ROUTER_ASSETS,
    ROUTER_REFUSALS,
    ROUTER_VERSION,
    ArchiveKind,
    RouterArchiveError,
    RouterAsset,
    RouterDownloadError,
    RouterNotPublishedError,
    install_router,
    installed_router,
    router_asset,
    router_on_first_use,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MIT_TEXT = "MIT License\n\nCopyright (c) 2024 Benson Wong\n"
BINARY_BODY = b"#!/bin/sh\nexit 0\n"


def _linux() -> HostCaps:
    return collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)


def _tar(path: Path, members: dict[str, bytes | None]) -> None:
    """A tar.gz; a value of None makes a directory entry."""
    with tarfile.open(path, "w:gz") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            if body is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
                continue
            info.size = len(body)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))


def _zip(path: Path, members: dict[str, bytes | None]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name if body is not None else f"{name}/", body or b"")


def _asset(path: Path, kind: ArchiveKind, binary: str) -> RouterAsset:
    data = path.read_bytes()
    return RouterAsset(
        filename=path.name,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        archive=kind,
        binary=binary,
    )


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.llama_server_path = ""
    host = _linux()
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(binary_mod, "_bundled_tool", lambda _tool: None)
    monkeypatch.setattr(binary_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(router_install, "_refusals", first_use.RefusedInstalls())
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A published linux archive and a download that copies it and records each fetch."""
    archive = tmp_path / "release" / "llama-swap_test_linux_amd64.tar.gz"
    archive.parent.mkdir()
    _tar(archive, {"llama-swap": BINARY_BODY, LICENCE_FILE: MIT_TEXT.encode(), "README.md": b"x"})
    monkeypatch.setattr(
        router_install,
        "ROUTER_ASSETS",
        {"linux-x86_64": _asset(archive, ArchiveKind.TAR_GZ, "llama-swap")},
    )
    calls: list[str] = []

    def fake(url: str, dest: Path, size: int, **_signals: Any) -> Path:
        calls.append(url)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / archive.name
        target.write_bytes(archive.read_bytes())
        return target

    monkeypatch.setattr(router_install, "download_url", fake)
    return calls


def _packs_copy(source_name: str, body: bytes) -> Path:
    pack = engines_root(cfg.models_dir) / PACK_DIR / source_name
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(body)
    return pack


class TestPinTable:
    def test_every_cell_is_a_release_asset_of_the_pinned_version(self) -> None:
        assert set(ROUTER_ASSETS) == {
            "linux-x86_64",
            "linux-arm64",
            "macos-arm64",
            "macos-x86_64",
            "windows-x86_64",
        }
        for asset in ROUTER_ASSETS.values():
            assert re.fullmatch(r"[0-9a-f]{64}", asset.sha256)
            assert asset.filename.startswith(f"llama-swap_{ROUTER_VERSION.removeprefix('v')}_")
            assert asset.size > 0

    def test_the_pin_is_the_version_the_engine_builds_use(self) -> None:
        text = (REPO_ROOT / "engine-versions.env").read_text(encoding="utf-8")
        assert f"ENGINE_LLAMA_SWAP_VERSION={ROUTER_VERSION}\n" in text

    @pytest.mark.parametrize(
        ("system", "machine", "cell", "filename", "binary"),
        [
            ("Linux", "x86_64", "linux-x86_64", "llama-swap_272_linux_amd64.tar.gz", "llama-swap"),
            ("Linux", "aarch64", "linux-arm64", "llama-swap_272_linux_arm64.tar.gz", "llama-swap"),
            ("Darwin", "arm64", "macos-arm64", "llama-swap_272_darwin_arm64.tar.gz", "llama-swap"),
            (
                "Darwin",
                "x86_64",
                "macos-x86_64",
                "llama-swap_272_darwin_amd64.tar.gz",
                "llama-swap",
            ),
            (
                "Windows",
                "AMD64",
                "windows-x86_64",
                "llama-swap_272_windows_amd64.zip",
                "llama-swap.exe",
            ),
        ],
    )
    def test_a_computer_with_an_engine_build_has_a_router_asset(
        self, system: str, machine: str, cell: str, filename: str, binary: str
    ) -> None:
        host = collect(system=system, machine=machine, cpu_info={}, nvidia=None)
        found_cell, asset = router_asset(host)
        assert found_cell == cell
        assert asset.filename == filename
        assert asset.binary == binary

    @pytest.mark.parametrize(("system", "machine"), [("Windows", "ARM64"), ("Plan9", "x86_64")])
    def test_a_computer_with_no_asset_is_refused_in_plain_words(
        self, system: str, machine: str
    ) -> None:
        host = collect(system=system, machine=machine, cpu_info={}, nvidia=None)
        with pytest.raises(
            RouterNotPublishedError,
            match=r"has no build for this computer\. Put a llama-swap binary on PATH\.",
        ):
            router_asset(host)


class TestInstall:
    def test_the_router_installs_beside_the_packs_with_its_licence_notice(
        self, release: list[str]
    ) -> None:
        binary = install_router()
        folder = engines_root(cfg.models_dir) / "router" / ROUTER_VERSION
        assert binary == folder / "llama-swap"
        assert binary.read_bytes() == BINARY_BODY
        # Windows has no execute bit: there, every file that exists passes.
        assert os.access(binary, os.X_OK)
        assert "MIT License" in (folder / LICENCE_FILE).read_text(encoding="utf-8")
        assert sorted(p.name for p in folder.iterdir()) == [LICENCE_FILE, "llama-swap"]
        assert release == [
            f"https://github.com/mostlygeek/llama-swap/releases/download/{ROUTER_VERSION}/"
            "llama-swap_test_linux_amd64.tar.gz"
        ]

    def test_a_second_install_downloads_nothing(self, release: list[str]) -> None:
        first = install_router()
        second = install_router()
        assert first == second
        assert len(release) == 1

    def test_a_failed_install_leaves_no_stage_folder(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = router_install.ROUTER_ASSETS["linux-x86_64"]
        bad = RouterAsset(good.filename, "0" * 64, good.size, good.archive, good.binary)
        monkeypatch.setattr(router_install, "ROUTER_ASSETS", {"linux-x86_64": bad})
        with pytest.raises(ChecksumError):
            install_router()
        parent = engines_root(cfg.models_dir) / "router"
        assert list(parent.iterdir()) == []

    def test_a_folder_left_by_an_earlier_install_is_replaced(self, release: list[str]) -> None:
        stale = engines_root(cfg.models_dir) / "router" / ROUTER_VERSION
        stale.mkdir(parents=True)
        (stale / "stale.txt").write_text("old", encoding="utf-8")
        install_router()
        assert sorted(p.name for p in stale.iterdir()) == [LICENCE_FILE, "llama-swap"]

    def test_the_install_holds_the_router_lock_for_the_whole_install(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        held: list[tuple[Path, float]] = []
        real_lock = router_install.file_lock

        def record(path: Path, wait_s: float) -> Any:
            held.append((path, wait_s))
            return real_lock(path, wait_s)

        monkeypatch.setattr(router_install, "file_lock", record)
        install_router()
        assert held == [(engines_root(cfg.models_dir) / ".locks" / "router", 1800.0)]

    def test_the_download_gets_the_size_the_name_and_the_caller_signals(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asset = router_install.ROUTER_ASSETS["linux-x86_64"]
        sizing: list[tuple[Path, str, int]] = []
        fetching: list[tuple[int, dict[str, Any]]] = []
        fixture_download = router_install.download_url

        def shortfall(stage: Path, name: str, size: int) -> None:
            sizing.append((stage, name, size))

        def download(url: str, dest: Path, size: int, **signals: Any) -> Path:
            fetching.append((size, signals))
            return fixture_download(url, dest, size)

        monkeypatch.setattr(router_install, "disk_shortfall", shortfall)
        monkeypatch.setattr(router_install, "download_url", download)

        def progress(_done: int, _total: int) -> None:
            return None

        def retry(_attempt: int, _limit: int, _reason: str) -> None:
            return None

        cancel = threading.Event()
        install_router(on_progress=progress, on_retry=retry, cancel=cancel)
        assert [(name, size) for _stage, name, size in sizing] == [(asset.filename, asset.size)]
        assert fetching == [
            (asset.size, {"on_progress": progress, "on_retry": retry, "cancel": cancel})
        ]

    def test_an_archive_with_the_wrong_digest_is_refused_and_installs_nothing(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = router_install.ROUTER_ASSETS["linux-x86_64"]
        bad = RouterAsset(good.filename, "0" * 64, good.size, good.archive, good.binary)
        monkeypatch.setattr(router_install, "ROUTER_ASSETS", {"linux-x86_64": bad})
        with pytest.raises(
            ChecksumError,
            match=r"does not match its pin, so lilbee did not install it\. "
            r"Run the install again\. If this happens again, report it\.",
        ):
            install_router()
        assert installed_router() is None
        assert not (engines_root(cfg.models_dir) / "router" / ROUTER_VERSION).exists()

    @pytest.mark.parametrize(
        ("members", "message"),
        [
            ({LICENCE_FILE: MIT_TEXT.encode()}, "holds no file 'llama-swap'"),
            ({"llama-swap": BINARY_BODY}, f"holds no file {LICENCE_FILE!r}"),
            ({"llama-swap": None, LICENCE_FILE: MIT_TEXT.encode()}, "is not a file"),
            ({"llama-swap": BINARY_BODY, LICENCE_FILE: b"GPL"}, "holds no MIT License notice"),
            ({"llama-swap": BINARY_BODY, LICENCE_FILE: b"\xff\xfe"}, "holds no MIT License notice"),
        ],
        ids=["no-binary", "no-licence", "directory-binary", "wrong-licence", "unreadable-licence"],
    )
    def test_a_tar_that_lacks_the_router_or_its_notice_is_refused(
        self,
        tmp_path: Path,
        release: list[str],
        monkeypatch: pytest.MonkeyPatch,
        members: dict[str, bytes | None],
        message: str,
    ) -> None:
        archive = tmp_path / "release" / "odd.tar.gz"
        _tar(archive, members)
        monkeypatch.setattr(
            router_install,
            "ROUTER_ASSETS",
            {"linux-x86_64": _asset(archive, ArchiveKind.TAR_GZ, "llama-swap")},
        )
        monkeypatch.setattr(
            router_install,
            "download_url",
            lambda url, dest, size, **_s: (
                (dest / "odd.tar.gz").write_bytes(archive.read_bytes()) and dest / "odd.tar.gz"
            ),
        )
        with pytest.raises(RouterArchiveError, match=message):
            install_router()
        assert installed_router() is None

    def test_a_windows_zip_installs_the_exe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = tmp_path / "release" / "llama-swap_test_windows_amd64.zip"
        archive.parent.mkdir()
        _zip(archive, {"llama-swap.exe": BINARY_BODY, LICENCE_FILE: MIT_TEXT.encode()})
        host = collect(system="Windows", machine="AMD64", cpu_info={}, nvidia=None)
        monkeypatch.setattr(runnable, "session_host", lambda: host)
        monkeypatch.setattr(
            router_install,
            "ROUTER_ASSETS",
            {"windows-x86_64": _asset(archive, ArchiveKind.ZIP, "llama-swap.exe")},
        )
        _packs_copy(archive.name, archive.read_bytes())
        assert install_router().name == "llama-swap.exe"
        assert installed_router() == install_router()

    @pytest.mark.parametrize(
        ("members", "message"),
        [
            ({LICENCE_FILE: MIT_TEXT.encode()}, "holds no file 'llama-swap.exe'"),
        ],
        ids=["no-binary"],
    )
    def test_a_zip_that_lacks_the_router_is_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        members: dict[str, bytes | None],
        message: str,
    ) -> None:
        archive = tmp_path / "release" / "odd.zip"
        archive.parent.mkdir()
        _zip(archive, members)
        monkeypatch.setattr(
            router_install,
            "ROUTER_ASSETS",
            {"linux-x86_64": _asset(archive, ArchiveKind.ZIP, "llama-swap.exe")},
        )
        _packs_copy(archive.name, archive.read_bytes())
        with pytest.raises(RouterArchiveError, match=message):
            install_router()

    def test_an_archive_copied_into_the_packs_folder_needs_no_download(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asset = router_install.ROUTER_ASSETS["linux-x86_64"]
        source = next((Path(cfg.data_root) / "release").iterdir())
        _packs_copy(asset.filename, source.read_bytes())
        assert install_router().is_file()
        assert release == []
        assert (engines_root(cfg.models_dir) / "packs" / asset.filename).is_file()

    def test_an_archive_rewritten_after_the_check_installs_the_checked_bytes(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asset = router_install.ROUTER_ASSETS["linux-x86_64"]
        source = next((Path(cfg.data_root) / "release").iterdir())
        pack = _packs_copy(asset.filename, source.read_bytes())
        real_check = router_install._check_digest

        def check_then_rewrite(checked: RouterAsset, digest: str) -> None:
            real_check(checked, digest)
            _tar(pack, {"llama-swap": b"#!/bin/sh\nexit 9\n", LICENCE_FILE: MIT_TEXT.encode()})

        monkeypatch.setattr(router_install, "_check_digest", check_then_rewrite)
        assert install_router().read_bytes() == BINARY_BODY

    def test_every_archive_open_reads_the_copy_in_the_stage_folder(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asset = router_install.ROUTER_ASSETS["linux-x86_64"]
        source = next((Path(cfg.data_root) / "release").iterdir())
        pack = _packs_copy(asset.filename, source.read_bytes())
        opened: list[Path] = []
        real_open = tarfile.open

        def spy(name: Path, *args: Any, **kwargs: Any) -> tarfile.TarFile:
            opened.append(Path(name))
            return real_open(name, *args, **kwargs)

        monkeypatch.setattr(router_install.tarfile, "open", spy)
        install_router()
        assert len(opened) == 1
        assert opened[0] != pack
        assert opened[0].parent.parent == engines_root(cfg.models_dir) / "router"
        assert opened[0].parent.name.startswith(STAGE_PREFIX)

    def test_the_copy_is_gone_after_a_refused_packs_archive(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = router_install.ROUTER_ASSETS["linux-x86_64"]
        bad = RouterAsset(good.filename, "0" * 64, good.size, good.archive, good.binary)
        monkeypatch.setattr(router_install, "ROUTER_ASSETS", {"linux-x86_64": bad})
        source = next((Path(cfg.data_root) / "release").iterdir())
        pack = _packs_copy(good.filename, source.read_bytes())
        with pytest.raises(ChecksumError):
            install_router()
        assert list((engines_root(cfg.models_dir) / "router").iterdir()) == []
        assert pack.is_file()

    def test_a_download_that_fails_names_the_file_to_copy(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_a: object, **_k: object) -> Path:
            raise RuntimeError("HTTP 404")

        monkeypatch.setattr(router_install, "download_url", refuse)
        with pytest.raises(
            RouterDownloadError, match=r"Copy llama-swap_test_linux_amd64\.tar\.gz into"
        ):
            install_router()

    def test_a_disk_too_small_is_refused_before_the_download(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(router_install, "disk_shortfall", lambda *_a: "Not enough disk space.")
        with pytest.raises(EngineDiskError, match="Not enough disk space"):
            install_router()
        assert release == []

    def test_a_computer_with_no_asset_installs_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = collect(system="Windows", machine="ARM64", cpu_info={}, nvidia=None)
        monkeypatch.setattr(runnable, "session_host", lambda: host)
        with pytest.raises(RouterNotPublishedError):
            install_router()


class TestFirstUse:
    def test_a_refused_install_is_remembered_and_fetches_nothing_again(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(url: str, *_a: object, **_k: object) -> Path:
            release.append(url)
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", refuse)
        with pytest.raises(RouterDownloadError) as first:
            router_on_first_use()
        with pytest.raises(RouterDownloadError) as second:
            router_on_first_use()
        assert second.value is first.value
        assert len(release) == 1

    def test_an_explicit_install_after_a_refusal_works_and_clears_it(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = router_install.download_url

        def offline(*_a: object, **_k: object) -> Path:
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", offline)
        with pytest.raises(RouterDownloadError):
            router_on_first_use()
        monkeypatch.setattr(router_install, "download_url", real)
        installed = install_router()
        assert router_on_first_use() == installed

    def test_a_refusal_in_one_models_folder_does_not_stop_an_install_in_another(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = router_install.download_url

        def offline(*_a: object, **_k: object) -> Path:
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", offline)
        with pytest.raises(RouterDownloadError):
            router_on_first_use()
        monkeypatch.setattr(router_install, "download_url", real)
        cfg.models_dir = Path(cfg.data_root) / "elsewhere" / "models"
        assert router_on_first_use().is_file()

    def test_an_explicit_install_clears_the_refusal_for_a_later_missing_router(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = router_install.download_url

        def offline(*_a: object, **_k: object) -> Path:
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", offline)
        with pytest.raises(RouterDownloadError):
            router_on_first_use()
        monkeypatch.setattr(router_install, "download_url", real)
        install_router()
        shutil.rmtree(router_install.router_dir())
        assert router_on_first_use().is_file()
        assert len(release) == 2

    def test_a_router_placed_after_a_refusal_is_used(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def offline(*_a: object, **_k: object) -> Path:
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", offline)
        with pytest.raises(RouterDownloadError):
            router_on_first_use()
        placed = router_install.router_dir() / "llama-swap"
        placed.parent.mkdir(parents=True)
        placed.write_bytes(BINARY_BODY)
        assert router_on_first_use() == placed

    def test_the_refusals_are_the_errors_a_caller_can_act_on(self) -> None:
        assert RouterDownloadError in ROUTER_REFUSALS
        assert ChecksumError in ROUTER_REFUSALS


class TestResolve:
    """A TEI-only, mistral.rs-only or MLX-only store has no llama.cpp pack."""

    def test_an_install_with_no_llama_pack_gets_the_router_not_the_pack(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_pack(*_a: object, **_k: object) -> Path:
            raise AssertionError("the router needs no llama.cpp pack")

        monkeypatch.setattr(first_use, "fetch_frozen", no_pack)
        found = resolve_llama_swap()
        assert found == engines_root(cfg.models_dir) / "router" / ROUTER_VERSION / "llama-swap"
        assert found.is_file()
        assert len(release) == 1

    def test_a_second_lookup_reads_the_installed_router(self, release: list[str]) -> None:
        assert resolve_llama_swap() == resolve_llama_swap()
        assert len(release) == 1

    def test_the_bundled_router_wins_over_the_installed_one(
        self, release: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_router()
        bundled = tmp_path / "bundled" / "llama-swap"
        monkeypatch.setattr(
            binary_mod,
            "_bundled_tool",
            lambda tool: bundled if tool is EngineTool.LLAMA_SWAP else None,
        )
        assert resolve_llama_swap() == bundled

    def test_the_installed_router_wins_over_path(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed = install_router()
        monkeypatch.setattr(binary_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert resolve_llama_swap() == installed

    def test_the_installed_router_wins_over_a_copy_in_the_llama_pack(
        self, release: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed = install_router()
        in_pack = tmp_path / "pack" / "llama-swap"
        monkeypatch.setattr(binary_mod, "_installed_pack_tool", lambda _tool: in_pack)
        assert resolve_llama_swap() == installed

    def test_a_copy_in_the_llama_pack_is_used_when_no_router_is_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        in_pack = tmp_path / "pack" / "llama-swap"
        monkeypatch.setattr(binary_mod, "_installed_pack_tool", lambda _tool: in_pack)
        assert resolve_llama_swap() == in_pack

    def test_path_is_used_when_no_router_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(binary_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert resolve_llama_swap() == Path("/usr/bin/llama-swap")

    def test_an_unreadable_router_folder_reads_as_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken() -> Path:
            raise OSError("denied")

        monkeypatch.setattr(router_install, "installed_router", broken)
        assert binary_mod._installed_router() is None

    def test_a_refused_install_raises_the_error_with_the_fix(
        self, release: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def offline(*_a: object, **_k: object) -> Path:
            raise RuntimeError("offline")

        monkeypatch.setattr(router_install, "download_url", offline)
        with pytest.raises(
            ProviderError, match=r"llama-swap router .* cannot be downloaded"
        ) as err:
            resolve_llama_swap()
        assert err.value.kind is ProviderErrorKind.UNKNOWN

    def test_your_own_llama_server_still_gets_the_path_hint_for_a_missing_router(
        self, release: list[str], tmp_path: Path
    ) -> None:
        server = tmp_path / "my-llama-server"
        server.write_text("#!/bin/sh\n", encoding="utf-8")
        cfg.llama_server_path = str(server)
        with pytest.raises(ProviderError, match="installs no engine pack"):
            resolve_llama_swap()
        assert release == []


class TestStore:
    def test_the_engine_list_and_prune_leave_the_router_alone(self, release: list[str]) -> None:
        installed = install_router()
        store = EngineStore(engines_root(cfg.models_dir))
        assert store.installed() == []
        assert store.build_dirs() == []
        assert store.sweep() == []
        assert installed.is_file()

    def test_a_sweep_removes_an_abandoned_stage_folder(self, release: list[str]) -> None:
        install_router()
        leftover = engines_root(cfg.models_dir) / "router" / ".stage-abandoned"
        leftover.mkdir()
        store = EngineStore(engines_root(cfg.models_dir))
        assert store.sweep() == [leftover]
        assert installed_router() is not None
