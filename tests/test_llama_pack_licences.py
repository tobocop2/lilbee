"""The pin of each llama.cpp pack cell names the licence terms that its published wheel ships."""

from __future__ import annotations

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_store import NOTICE_RULES, licence_permitted
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import LlamaCppAdapter
from lilbee.providers.fleet.host_caps import HostCaps, collect
from tests._published import render_table, wheel_name

VERSION = "1.12032.1"
AMD = 0x1002
_CARD = {"driver": "550.135", "cuda_driver": 12040, "gpus": [{"name": "card", "cc": [8, 6]}]}

# License-Expression of each wheel in release engine-packs-20261121-llama, read from its METADATA.
_UNIX = "MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND MIT-0 AND Unlicense"
_WINDOWS = (
    "MIT AND Apache-2.0 WITH LLVM-exception AND LicenseRef-Microsoft-VC-Runtime AND Apache-2.0 "
    "AND BSD-2-Clause AND BSD-3-Clause AND ISC AND MIT-0 AND Unlicense"
)
_LINUX_CUDA = (
    "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND Apache-2.0 WITH LLVM-exception "
    "AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT-0 AND Unlicense"
)
_WINDOWS_CUDA = (
    "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 WITH LLVM-exception "
    "AND LicenseRef-Microsoft-VC-Runtime AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause "
    "AND BSL-1.0 AND ISC AND MIT-0 AND Unlicense"
)
_ROCM = (
    "MIT AND BSD-3-Clause AND NCSA AND Apache-2.0 WITH LLVM-exception AND Apache-2.0 "
    "AND BSD-2-Clause AND ISC AND MIT-0 AND Unlicense"
)


def _gpu_host(
    system: str,
    machine: str,
    *,
    nvidia: dict[str, object] | None = None,
    display_vendors: tuple[int, ...] = (),
    amd_gfx: tuple[str, ...] = (),
    vulkan_loader: bool = False,
    libnuma: bool = False,
) -> HostCaps:
    return collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"]},
        nvidia=nvidia,
        display_vendors=display_vendors,
        amd_gfx=amd_gfx,
        vulkan_loader=vulkan_loader,
        libnuma=libnuma,
    )


_CASES = [
    pytest.param(
        collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None, mac_ver="15.1"),
        "macos-arm64-metal",
        _UNIX,
        id="macos-metal",
    ),
    pytest.param(
        _gpu_host("Linux", "x86_64", nvidia=None), "linux-x86_64-cpu", _UNIX, id="linux-cpu"
    ),
    pytest.param(
        _gpu_host("Linux", "x86_64", nvidia=None, display_vendors=(AMD,), vulkan_loader=True),
        "linux-x86_64-vulkan",
        _UNIX,
        id="linux-vulkan",
    ),
    pytest.param(
        _gpu_host("Linux", "x86_64", nvidia=_CARD),
        "linux-x86_64-cuda",
        _LINUX_CUDA,
        id="linux-cuda",
    ),
    pytest.param(
        _gpu_host(
            "Linux",
            "x86_64",
            nvidia=None,
            display_vendors=(AMD,),
            amd_gfx=("gfx1100",),
            libnuma=True,
        ),
        "linux-x86_64-rocm",
        _ROCM,
        id="linux-rocm",
    ),
    pytest.param(
        _gpu_host("Windows", "AMD64", nvidia=None),
        "windows-x86_64-cpu",
        _WINDOWS,
        id="windows-cpu",
    ),
    pytest.param(
        _gpu_host("Windows", "AMD64", nvidia=None, display_vendors=(AMD,), vulkan_loader=True),
        "windows-x86_64-vulkan",
        _WINDOWS,
        id="windows-vulkan",
    ),
    pytest.param(
        _gpu_host("Windows", "AMD64", nvidia=_CARD),
        "windows-x86_64-cuda",
        _WINDOWS_CUDA,
        id="windows-cuda",
    ),
]


_NO_WHEEL_CASES = [
    pytest.param(
        collect(system="Linux", machine="aarch64", cpu_info={}, nvidia=None),
        id="linux-arm64-cpu",
    ),
    pytest.param(
        collect(system="Darwin", machine="x86_64", cpu_info={}, nvidia=None, mac_ver="15.1"),
        id="macos-x86_64-cpu",
    ),
    pytest.param(
        collect(system="Windows", machine="ARM64", cpu_info={}, nvidia=None),
        id="windows-arm64-cpu",
    ),
    pytest.param(
        collect(system="Plan9", machine="mips", cpu_info={}, nvidia=None),
        id="unread-host",
    ),
]


@pytest.mark.parametrize("host", _NO_WHEEL_CASES)
def test_a_cell_with_no_published_wheel_pins_the_mit_terms(
    host: HostCaps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = LlamaCppAdapter().pin_record()
    assert pin["platforms"] == {}
    assert pin["licenses"] == ["MIT"]


@pytest.mark.parametrize(("host", "cell", "terms"), _CASES)
def test_the_pin_of_each_cell_equals_the_terms_of_its_wheel(
    host: HostCaps, cell: str, terms: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    file = wheel_name(EngineName.LLAMA, VERSION, cell)
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table([(file, "ab" * 32, 1)]))
    pin = LlamaCppAdapter().pack_pin()
    assert list(pin["platforms"]) == [cell]
    assert pin["licenses"] == [terms]
    notices = frozenset(rule.file for rule in NOTICE_RULES.values())
    assert all(licence_permitted(expression, notices) for expression in pin["licenses"])
