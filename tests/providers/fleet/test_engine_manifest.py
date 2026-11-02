"""The engine manifest: its schema, the cell choice for a host, and the OS floor."""

from __future__ import annotations

import copy
from typing import Any
from unittest import mock

import pytest

from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_manifest import (
    Manifest,
    ManifestError,
    UnsupportedOsError,
    host_cell,
    host_os_version,
    parse_manifest,
)
from lilbee.providers.fleet.engine_store import (
    PinLock,
    PlatformAsset,
    UnsupportedPlatformError,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.host_caps import HostOS, collect

BASE = "https://github.com/tobocop2/lilbee-engines/releases/download/llama.cpp-b12032-1/"
SHA = "a" * 64
OWN_ENGINE = "set LILBEE_LLAMA_SERVER_PATH to an engine you built."


def row(os_name: str, arch: str, backend: str, min_os: str | None, version: str = "b12032") -> Any:
    return {
        "os": os_name,
        "arch": arch,
        "backend": backend,
        "version": version,
        "url": f"{BASE}lilbee_engine-1-{backend}-{os_name}.whl",
        "sha256": SHA,
        "size": 1000,
        "min_os": min_os,
    }


def doc() -> dict[str, Any]:
    cells = {
        "macos-arm64-metal": row("macos", "arm64", "metal", "13.3"),
        "linux-x86_64-vulkan": row("linux", "x86_64", "vulkan", "2.34"),
        "linux-x86_64-cu121": row("linux", "x86_64", "cu121", "2.35", "b9665"),
        "windows-x86_64-cpu": row("windows", "x86_64", "cpu", None),
    }
    return {
        "schema": 1,
        "engine": "llama.cpp",
        "release": "llama.cpp-b12032-1",
        "licenses": ["MIT"],
        "cells": cells,
    }


@pytest.fixture
def manifest() -> Manifest:
    return parse_manifest(doc())


def test_a_good_manifest_keeps_its_cells_and_licences(manifest: Manifest) -> None:
    assert (manifest.engine, manifest.release, manifest.licences) == (
        "llama.cpp",
        "llama.cpp-b12032-1",
        ("MIT",),
    )
    cell = manifest.cells["linux-x86_64-cu121"]
    assert (cell.version, cell.asset.min_os, cell.asset.size) == ("b9665", "2.35", 1000)
    assert manifest.cells["windows-x86_64-cpu"].asset.min_os is None


def break_cell(**change: Any) -> dict[str, Any]:
    broken = doc()
    broken["cells"]["linux-x86_64-vulkan"].update(change)
    return broken


def break_top(**change: Any) -> dict[str, Any]:
    broken = doc()
    broken.update(change)
    return broken


BAD = {
    "schema-2": break_top(schema=2),
    "no-cells": break_top(cells={}),
    "cells-not-a-map": break_top(cells=[1]),
    "no-engine": break_top(engine=""),
    "gpl": break_top(licenses=["GPL-3.0-only"]),
    "no-licences": break_top(licenses=[]),
    "licence-not-text": break_top(licenses=[7]),
    "http": break_cell(url=BASE.replace("https", "http") + "x.whl"),
    "other-host": break_cell(url="https://example.com/x.whl"),
    "not-a-wheel": break_cell(url=BASE + "x.tar.gz"),
    "dot-dot": break_cell(url=BASE + "../../../../evil/mal/releases/download/r/x.whl"),
    "encoded-dot-dot": break_cell(url=BASE + "%2e%2E/x.whl"),
    "query-hides-suffix": break_cell(url=BASE + "x?u=.whl"),
    "query-after-suffix": break_cell(url=BASE + "x.whl?u=1"),
    "empty-query": break_cell(url=BASE + "x.whl?"),
    "fragment-hides-suffix": break_cell(url=BASE + "x#.whl"),
    "empty-fragment": break_cell(url=BASE + "x.whl#"),
    "short-sha": break_cell(sha256="a" * 63),
    "non-hex-sha": break_cell(sha256="z" * 64),
    "zero-size": break_cell(size=0),
    "bool-size": break_cell(size=True),
    "empty-min-os": break_cell(min_os=""),
    "text-min-os": break_cell(min_os="new"),
    "no-min-os": {
        **doc(),
        "cells": {
            "k": {k: v for k, v in row("linux", "x86_64", "k", "1").items() if k != "min_os"}
        },
    },
    "key-mismatch": break_top(cells={"linux-x86_64-cpu": row("linux", "x86_64", "vulkan", "2.34")}),
    "cell-not-an-object": break_top(cells={"linux-x86_64-cpu": "x"}),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_a_manifest_that_breaks_the_schema_is_refused(name: str) -> None:
    with pytest.raises(ManifestError):
        parse_manifest(copy.deepcopy(BAD[name]))


def test_the_bad_manifest_table_is_not_empty() -> None:
    assert len(BAD) >= 26


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
    manifest: Manifest, os_name: str, arch: str, backend: str, version: str
) -> None:
    asset = manifest.select(os_name, arch, backend, version)
    assert asset.url == manifest.cells[f"{os_name}-{arch}-{backend}"].asset.url


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
    manifest: Manifest, os_name: str, backend: str, version: str, want: str
) -> None:
    with pytest.raises(UnsupportedOsError) as caught:
        manifest.select(os_name, "arm64" if os_name == "macos" else "x86_64", backend, version)
    assert str(caught.value) == want


def test_a_floor_names_the_version_of_its_own_cell(manifest: Manifest) -> None:
    with pytest.raises(
        UnsupportedOsError, match=r"Engine llama\.cpp b9665 \(cu121\) needs glibc 2\.35"
    ):
        manifest.select("linux", "x86_64", "cu121", "2.34")


def test_a_floor_on_windows_names_windows() -> None:
    asset = PlatformAsset(BASE + "x.whl", SHA, 1, "10.0.19045")
    with pytest.raises(
        UnsupportedOsError, match=r"needs Windows 10\.0\.19045 or later.*Update Windows"
    ):
        engine_manifest.require_floor("e", "1", "cpu", asset, HostOS.WINDOWS, "10.0.17763")


def test_a_missing_backend_lists_the_builds_for_this_computer(manifest: Manifest) -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        manifest.select("windows", "x86_64", "rocm", "10.0.26100")
    assert str(caught.value) == (
        "Engine llama.cpp has no rocm build for windows-x86_64. Builds for this computer: cpu."
    )


def test_a_host_with_no_build_at_all_says_none(manifest: Manifest) -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        manifest.select("linux", "aarch64", "vulkan", "2.39")
    assert str(caught.value) == (
        "Engine llama.cpp has no vulkan build for linux-aarch64. Builds for this computer: none."
    )


def test_builds_for_this_computer_are_sorted_and_leave_out_other_hosts(manifest: Manifest) -> None:
    with pytest.raises(UnsupportedPlatformError) as caught:
        manifest.select("linux", "x86_64", "rocm", "2.39")
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
    return collect(system="Linux", machine="x86_64", cpu_info={}, nvml=None)


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
    odd = collect(system="Plan9", machine="riscv64", cpu_info={}, nvml=None)
    assert (odd.os, odd.arch) == (None, None)
    assert host_cell(lock_with({}), odd) == ""


def test_a_host_of_unknown_os_has_no_cell_in_a_pin_with_cells() -> None:
    host = collect(system="Plan9", machine="x86_64", cpu_info={}, nvml=None)
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
            f"{BASE}lilbee_engine-1-vulkan-linux.whl", SHA, 1000, "2.34"
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
