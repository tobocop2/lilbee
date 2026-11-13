"""The digest table lookups, the cell choice for a host, and the OS floor."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_manifest import (
    UnsupportedOsError,
    host_cell,
    host_os_version,
)
from lilbee.providers.fleet.engine_store import (
    PinLock,
    PlatformAsset,
    UnsupportedPlatformError,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import HostOS, collect
from tests._published import render_table, wheel_name

BASE = "https://github.com/tobocop2/lilbee-engines/releases/download/llama.cpp-b12032-1/"
SHA = "a" * 64
OWN_ENGINE = "set LILBEE_LLAMA_SERVER_PATH to an engine you built."


def row(os_name: str, arch: str, backend: str, min_os: str | None, version: str = "b12032") -> Any:
    return {
        "os": os_name,
        "arch": arch,
        "backend": backend,
        "version": version,
        "url": f"{BASE}lilbee_engine_llama_cpp-1-{backend}-{os_name}.whl",
        "sha256": SHA,
        "size": 1000,
        "min_os": min_os,
    }


CELLS = {
    "macos-arm64-metal": PlatformAsset(BASE + "m.whl", SHA, 1000, "13.3"),
    "linux-x86_64-vulkan": PlatformAsset(BASE + "v.whl", SHA, 1000, "2.34"),
    "linux-x86_64-cu121": PlatformAsset(BASE + "c.whl", SHA, 1000, "2.35"),
    "windows-x86_64-cpu": PlatformAsset(BASE + "w.whl", SHA, 1000, None),
}


def select(
    os_name: str, arch: str, backend: str, os_version: str, version: str = "b12032"
) -> PlatformAsset:
    """The cell for a host and the floor check, as a pin with these cells runs them."""
    asset = engine_manifest.find_cell("llama.cpp", CELLS, os_name, arch, backend)
    engine_manifest.require_floor("llama.cpp", version, backend, asset, HostOS(os_name), os_version)
    return asset


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The table the generator writes for one wheel per cell of CELLS."""
    wheels = [
        (wheel_name(EngineName.LLAMA, "b12032", key, asset.min_os), SHA, 1000)
        for key, asset in CELLS.items()
    ]
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table(wheels))


def test_a_pinned_digest_is_read_from_the_cell_of_the_engine() -> None:
    assert engine_manifest.pack_digest("llama.cpp", "b12032", linux(), "vulkan") == SHA


@pytest.mark.parametrize(
    ("engine", "version", "backend"),
    [("tei", "b12032", "vulkan"), ("llama.cpp", "b12032", "rocm"), ("llama.cpp", "b9", "vulkan")],
)
def test_an_engine_version_or_cell_the_table_does_not_list_has_no_digest(
    engine: str, version: str, backend: str
) -> None:
    assert engine_manifest.pack_digest(engine, version, linux(), backend) is None


MAC_OLD = (
    "Engine llama.cpp b12032 (metal) needs macOS 13.3 or later. This computer has macOS {}. "
    "Update macOS, or " + OWN_ENGINE
)
GLIBC_OLD = (
    "Engine llama.cpp b12032 (vulkan) needs glibc 2.34 or later. This computer has {}. "
    "Update the system, or " + OWN_ENGINE
)


@pytest.mark.parametrize(
    ("os_name", "arch", "backend", "version"),
    [
        ("macos", "arm64", "metal", "13.3"),
        ("macos", "arm64", "metal", "13.10"),
        ("macos", "arm64", "metal", "14.0"),
        ("linux", "x86_64", "vulkan", "2.34"),
        ("linux", "x86_64", "vulkan", "2.39"),
        ("linux", "x86_64", "cu121", "2.35"),
        ("windows", "x86_64", "cpu", "10.0.26100"),
        ("windows", "x86_64", "cpu", ""),
    ],
)
def test_a_host_at_or_above_the_floor_gets_the_cell_wheel(
    os_name: str, arch: str, backend: str, version: str
) -> None:
    asset = select(os_name, arch, backend, version)
    assert asset is CELLS[f"{os_name}-{arch}-{backend}"]


@pytest.mark.parametrize(
    ("os_name", "backend", "version", "want"),
    [
        ("macos", "metal", "13.2", MAC_OLD.format("13.2")),
        ("macos", "metal", "12.6", MAC_OLD.format("12.6")),
        (
            "macos",
            "metal",
            "",
            MAC_OLD.format("").replace("macOS .", "an unreadable macOS version."),
        ),
        ("linux", "vulkan", "2.31", GLIBC_OLD.format("glibc 2.31")),
        ("linux", "vulkan", "", GLIBC_OLD.format("no glibc")),
    ],
)
def test_a_host_below_the_floor_is_refused_with_the_fix(
    os_name: str, backend: str, version: str, want: str
) -> None:
    with pytest.raises(UnsupportedOsError) as caught:
        select(os_name, "arm64" if os_name == "macos" else "x86_64", backend, version)
    assert str(caught.value) == want


def test_a_floor_names_the_version_of_its_own_cell() -> None:
    with pytest.raises(
        UnsupportedOsError, match=r"Engine llama\.cpp b9665 \(cu121\) needs glibc 2\.35"
    ):
        select("linux", "x86_64", "cu121", "2.34", "b9665")


def test_a_floor_on_windows_names_windows() -> None:
    asset = PlatformAsset(BASE + "x.whl", SHA, 1, "10.0.19045")
    with pytest.raises(
        UnsupportedOsError, match=r"needs Windows 10\.0\.19045 or later.*Update Windows"
    ):
        engine_manifest.require_floor("e", "1", "cpu", asset, HostOS.WINDOWS, "10.0.17763")


def test_a_missing_backend_lists_the_builds_for_this_computer() -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        select("windows", "x86_64", "rocm", "10.0.26100")
    assert str(caught.value) == (
        "Engine llama.cpp has no rocm build for windows-x86_64. Builds for this computer: cpu."
    )


def test_a_host_with_no_build_at_all_says_none() -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        select("linux", "aarch64", "vulkan", "2.39")
    assert str(caught.value) == (
        "Engine llama.cpp has no vulkan build for linux-aarch64. Builds for this computer: none."
    )


def test_builds_for_this_computer_are_sorted_and_leave_out_other_hosts() -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        select("linux", "x86_64", "rocm", "2.39")
    assert str(caught.value).endswith("Builds for this computer: cu121, vulkan.")


@pytest.mark.parametrize(
    ("host_os", "patches", "want"),
    [
        (HostOS.MACOS, {"mac_ver": ("14.5", ("", "", ""), "arm64")}, "14.5"),
        (HostOS.LINUX, {"libc_ver": ("glibc", "2.39")}, "2.39"),
        (HostOS.LINUX, {"libc_ver": ("", "")}, ""),
        (HostOS.WINDOWS, {"version": "10.0.26100"}, "10.0.26100"),
    ],
)
def test_the_host_version_is_read_from_the_platform_module(
    host_os: HostOS, patches: dict[str, Any], want: str
) -> None:
    with mock.patch.multiple(
        engine_manifest.platform, **{k: mock.Mock(return_value=v) for k, v in patches.items()}
    ):
        assert host_os_version(host_os) == want


def lock_with(cells: dict[str, PlatformAsset]) -> PinLock:
    return PinLock("llama.cpp", "b12032", "vulkan", cells, ("MIT",), SHA, "server")


def linux() -> Any:
    return collect(system="Linux", machine="x86_64", cpu_info={}, nvidia=None)


def test_a_pin_with_cells_chooses_the_cell_key_of_the_host() -> None:
    lock = lock_with({"linux-x86_64-vulkan": PlatformAsset(BASE + "x.whl", SHA, 1, None)})
    assert host_cell(lock, linux()) == "linux-x86_64-vulkan"


def test_a_pin_with_cells_refuses_a_host_without_one() -> None:
    lock = lock_with({"macos-arm64-vulkan": PlatformAsset(BASE + "x.whl", SHA, 1, None)})
    with pytest.raises(UnsupportedPlatformError, match="Builds for this computer: none"):
        host_cell(lock, linux())


def test_a_pin_with_cells_refuses_a_system_below_the_floor() -> None:
    lock = lock_with({"linux-x86_64-vulkan": PlatformAsset(BASE + "x.whl", SHA, 1, "2.34")})
    with (
        mock.patch.object(engine_manifest, "host_os_version", return_value="2.31"),
        pytest.raises(UnsupportedOsError, match=r"This computer has glibc 2\.31"),
    ):
        host_cell(lock, linux())


def test_a_pin_without_cells_has_no_cell_for_a_pack_install() -> None:
    assert host_cell(lock_with({}), linux()) == ""


def test_a_pin_without_cells_installs_on_a_host_of_unknown_os_and_architecture() -> None:
    odd = collect(system="Plan9", machine="riscv64", cpu_info={}, nvidia=None)
    assert (odd.os, odd.arch) == (None, None)
    assert host_cell(lock_with({}), odd) == ""


def test_a_host_of_unknown_os_has_no_cell_in_a_pin_with_cells() -> None:
    host = collect(system="Plan9", machine="x86_64", cpu_info={}, nvidia=None)
    lock = lock_with({"linux-x86_64-vulkan": PlatformAsset(BASE + "x.whl", SHA, 1, None)})
    with pytest.raises(UnsupportedPlatformError, match="has no vulkan build for this computer"):
        host_cell(lock, host)


def test_a_cell_digest_that_differs_from_the_pin_digest_is_the_digest_of_the_lock() -> None:
    cell = "linux-x86_64-vulkan"
    lock = lock_with({cell: PlatformAsset(BASE + "x.whl", "b" * 64, 1, None)})
    assert host_cell(lock, linux()) == cell
    assert lock.for_cell(cell).sha256 == "b" * 64


def test_a_pin_without_cells_keeps_its_top_digest() -> None:
    lock = lock_with({})
    assert lock.for_cell(host_cell(lock, linux())).sha256 == SHA


def test_a_cell_digest_that_differs_only_in_case_is_accepted() -> None:
    lock = lock_with({"linux-x86_64-vulkan": PlatformAsset(BASE + "x.whl", SHA.upper(), 1, None)})
    assert host_cell(lock, linux()) == "linux-x86_64-vulkan"


def test_the_floor_text_of_an_engine_that_ignores_the_own_engine_setting_omits_it() -> None:
    asset = PlatformAsset(BASE + "x.whl", SHA, 1, "2.34")
    with pytest.raises(UnsupportedOsError) as caught:
        engine_manifest.require_floor("ik_llama.cpp", "1", "cpu", asset, HostOS.LINUX, "2.31")
    assert str(caught.value) == (
        "Engine ik_llama.cpp 1 (cpu) needs glibc 2.34 or later. "
        "This computer has glibc 2.31. Update the system."
    )


def test_pin_lock_turns_the_cells_of_a_pin_into_assets() -> None:
    pin = {
        "engine": "llama.cpp",
        "version": "b12032",
        "backend": "vulkan",
        "sha256": SHA,
        "licenses": ["MIT"],
        "tool": "llama-server",
        "platforms": {"linux-x86_64-vulkan": row("linux", "x86_64", "vulkan", "2.34")},
    }
    lock = choice.pin_lock(pin)
    assert lock.platforms == {
        "linux-x86_64-vulkan": PlatformAsset(
            f"{BASE}lilbee_engine_llama_cpp-1-vulkan-linux.whl", SHA, 1000, "2.34"
        )
    }


def test_pin_lock_refuses_a_cell_without_a_floor_field() -> None:
    cell = row("linux", "x86_64", "vulkan", "2.34")
    del cell["min_os"]
    pin = {
        "engine": "llama.cpp",
        "version": "b1",
        "backend": "vulkan",
        "sha256": SHA,
        "licenses": ["MIT"],
        "platforms": {"linux-x86_64-vulkan": cell},
    }
    with pytest.raises(choice.PinError, match="not usable"):
        choice.pin_lock(pin)


def table_lock(floor: str | None, sha: str = SHA) -> PinLock:
    """The lock of a pin whose platforms come from a table the generator rendered."""
    file = wheel_name(EngineName.LLAMA, "b12032", "linux-x86_64-vulkan", floor)
    rows = render_table([(file, sha, 4)])["llama.cpp"]["b12032"]
    cells = {key: PlatformAsset(**row) for key, row in rows.items()}
    return lock_with(cells)


def test_a_floor_of_the_generated_table_refuses_an_old_system() -> None:
    lock = table_lock("2.40")
    with (
        mock.patch.object(engine_manifest, "host_os_version", return_value="2.31"),
        pytest.raises(UnsupportedOsError, match=r"needs glibc 2\.40 or later"),
    ):
        host_cell(lock, linux())


def test_a_floor_of_the_generated_table_accepts_a_new_system() -> None:
    lock = table_lock("2.40")
    with mock.patch.object(engine_manifest, "host_os_version", return_value="2.41"):
        assert host_cell(lock, linux()) == "linux-x86_64-vulkan"
