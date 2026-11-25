"""Pack digests come from the generated digest table, one per cell."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest, engine_store, host_caps
from lilbee.providers.fleet.engine_manifest import MANIFESTS
from lilbee.providers.fleet.engine_record import PACKAGES_URL
from lilbee.providers.fleet.engines import ik_llama, mistral, tei
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS, NoPublishedPackError, pin_lock
from lilbee.providers.fleet.engines.llama import _WHEEL_VERSION, LLAMA, LlamaCppAdapter
from tests._published import render_table, wheel_name

LINUX = "a" * 64
WINDOWS = "b" * 64
MACOS = "c" * 64


def _host(system: str, machine: str) -> host_caps.HostCaps:
    return host_caps.collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvidia=None,
    )


def _table(name: EngineName, cells: dict[str, str], version: str | None = None) -> dict[str, Any]:
    """The table the generator writes for the wheels of *name* in *cells*."""
    pinned = version or ADAPTERS[name].pin_record()["version"]
    return render_table([(wheel_name(name, pinned, key), sha, 1) for key, sha in cells.items()])


def _pin(monkeypatch: pytest.MonkeyPatch, name: EngineName, system: str, machine: str) -> Any:
    monkeypatch.setattr(runnable, "session_host", lambda: _host(system, machine))
    return ADAPTERS[name].pin_record()


def test_every_row_of_the_shipped_table_states_a_wheel() -> None:
    for versions in engine_manifest.MANIFESTS.values():
        for cells in versions.values():
            for row in cells.values():
                assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                assert row["url"].startswith("https://")
                assert row["size"] > 0
                assert row["min_os"] is None or re.fullmatch(r"\d+\.\d+", row["min_os"])


def _recorded_digest(engine: str, cell: str) -> str:
    """The sha256 that the committed pack record gives the *cell* of *engine*."""
    versions = MANIFESTS[engine]
    assert len(versions) == 1
    return str(next(iter(versions.values()))[cell]["sha256"])


_SUMS_FIXTURE = Path(__file__).parent / "fixtures" / "engine_pack_sums" / "SHA256SUMS"


def _release_sums() -> dict[tuple[str, str], str]:
    """The kept SHA256SUMS of each pack release, by (release tag, file name)."""
    sums: dict[tuple[str, str], str] = {}
    release = ""
    for line in _SUMS_FIXTURE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# release "):
            release = line.removeprefix("# release ")
        elif not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            sums[(release, name)] = digest
    return sums


def _shipped_rows() -> list[Any]:
    return [row for v in MANIFESTS.values() for cells in v.values() for row in cells.values()]


def _disagreements(rows: list[Any], sums: dict[tuple[str, str], str]) -> list[str]:
    """The files whose row digest is not the SHA256SUMS line of the release the url names."""
    found = []
    for row in rows:
        release, name = row["url"].split("/")[-2:]
        if sums.get((release, name)) != row["sha256"]:
            found.append(name)
    return found


def test_every_shipped_digest_is_the_line_of_its_release_sums() -> None:
    rows = _shipped_rows()
    assert rows
    assert len(_release_sums()) >= len(rows)
    assert _disagreements(rows, _release_sums()) == []


def test_a_changed_shipped_digest_is_named_against_the_release_sums() -> None:
    rows = _shipped_rows()
    changed = [{**rows[0], "sha256": "0" * 64}, *rows[1:]]
    name = rows[0]["url"].rsplit("/", 1)[-1]
    assert _disagreements(changed, _release_sums()) == [name]


def test_a_row_whose_release_has_no_sums_line_is_named() -> None:
    rows = _shipped_rows()
    moved = [{**rows[0], "url": rows[0]["url"].replace("/engine-packs-", "/other-")}, *rows[1:]]
    name = rows[0]["url"].rsplit("/", 1)[-1]
    assert _disagreements(moved, _release_sums()) == [name]


def test_every_shipped_row_lives_under_the_pack_index_and_matches_its_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert engine_manifest.MANIFESTS
    for engine, versions in engine_manifest.MANIFESTS.items():
        pin = _pin(monkeypatch, EngineName(engine), "Linux", "x86_64")
        assert pin["engine"] == engine
        assert list(versions) == [_WHEEL_VERSION if engine == EngineName.LLAMA else pin["version"]]
        for cells in versions.values():
            for row in cells.values():
                assert row["url"].startswith(PACKAGES_URL)


def test_an_empty_table_publishes_no_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    for name in (EngineName.MISTRAL, EngineName.TEI, EngineName.IK_LLAMA):
        pin = _pin(monkeypatch, name, "Linux", "x86_64")
        assert (pin["sha256"], pin["platforms"]) == (None, {})


def test_cell_key_joins_os_arch_and_backend() -> None:
    host = _host("Linux", "x86_64")
    assert engine_manifest.cell_key(host, "cuda128-sm89") == "linux-x86_64-cuda128-sm89"
    assert engine_manifest.cell_key(host, None) is None


def test_an_unreadable_host_has_no_cell() -> None:
    host = _host("Plan9", "mips")
    assert host.os is None
    assert engine_manifest.cell_key(host, "cpu") is None
    assert engine_manifest.pack_digest("mistral.rs", "0.9.5", host, "cpu") is None


def test_mistral_cpu_digests_differ_between_linux_and_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = {"linux-x86_64-cpu": LINUX, "windows-x86_64-cpu": WINDOWS}
    monkeypatch.setattr(engine_manifest, "MANIFESTS", _table(EngineName.MISTRAL, cells))
    linux = _pin(monkeypatch, EngineName.MISTRAL, "Linux", "x86_64")
    windows = _pin(monkeypatch, EngineName.MISTRAL, "Windows", "AMD64")
    assert (linux["backend"], linux["sha256"]) == ("cpu", LINUX)
    assert (windows["backend"], windows["sha256"]) == ("cpu", WINDOWS)


def test_ik_llama_digest_follows_the_host_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = {"linux-x86_64-cpu": LINUX, "macos-arm64-cpu": MACOS}
    monkeypatch.setattr(engine_manifest, "MANIFESTS", _table(EngineName.IK_LLAMA, cells))
    assert _pin(monkeypatch, EngineName.IK_LLAMA, "Linux", "x86_64")["sha256"] == LINUX
    assert _pin(monkeypatch, EngineName.IK_LLAMA, "Darwin", "arm64")["sha256"] == MACOS
    assert _pin(monkeypatch, EngineName.IK_LLAMA, "Windows", "AMD64")["sha256"] is None


def test_tei_digest_follows_the_host_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_manifest, "MANIFESTS", _table(EngineName.TEI, {"linux-x86_64-cpu": LINUX})
    )
    assert _pin(monkeypatch, EngineName.TEI, "Linux", "x86_64")["sha256"] == LINUX
    assert _pin(monkeypatch, EngineName.TEI, "Windows", "AMD64")["sha256"] is None


_TEI_RELEASE = PACKAGES_URL + "engine-packs-20261118/"
_TEI_CELLS = {
    "cpu": "linux-x86_64-cpu",
    "metal": "macos-arm64-metal",
    "sm75": "linux-x86_64-cuda-sm75",
    "sm80": "linux-x86_64-cuda-sm80",
    "sm86": "linux-x86_64-cuda-sm86",
    "sm89": "linux-x86_64-cuda-sm89",
    "sm90": "linux-x86_64-cuda-sm90",
    "sm100": "linux-x86_64-cuda-sm100",
    "sm120": "linux-x86_64-cuda-sm120",
}
_TEI_HOSTS = {
    "cpu": ("Linux", "x86_64", None),
    "metal": ("Darwin", "arm64", None),
    "sm75": ("Linux", "x86_64", [7, 5]),
    "sm80": ("Linux", "x86_64", [8, 0]),
    "sm86": ("Linux", "x86_64", [8, 6]),
    "sm89": ("Linux", "x86_64", [8, 9]),
    "sm90": ("Linux", "x86_64", [9, 0]),
    "sm100": ("Linux", "x86_64", [10, 0]),
    "sm120": ("Linux", "x86_64", [12, 0]),
}


@pytest.mark.parametrize("build", _TEI_CELLS)
def test_each_host_resolves_its_own_tei_pack_from_the_shipped_table(
    build: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _TEI_CELLS[build]
    digest = _recorded_digest("tei", cell)
    system, machine, compute = _TEI_HOSTS[build]
    facts = (
        {"driver": "570.00", "cuda_driver": 12080, "gpus": [{"name": "gpu", "cc": compute}]}
        if compute
        else None
    )
    host = host_caps.collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvidia=facts,
    )
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = tei.TeiAdapter().pin_record()
    backend = cell.split("-", 2)[2]
    assert pin["backend"] == backend
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    url = pin["platforms"][cell]["url"]
    assert url.startswith(_TEI_RELEASE)
    assert f"-1.{backend.replace('-', '_')}-py3-none-" in url


def test_a_cell_with_no_digest_keeps_the_no_pack_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = {"linux-x86_64-cpu": LINUX}
    monkeypatch.setattr(engine_manifest, "MANIFESTS", _table(EngineName.MISTRAL, cells))
    pin = _pin(monkeypatch, EngineName.MISTRAL, "Darwin", "arm64")
    with pytest.raises(NoPublishedPackError) as exc:
        pin_lock(pin)
    assert str(exc.value) == "No pack of mistral.rs is published yet. llama.cpp stays in use."


@pytest.mark.parametrize(
    ("name", "other"),
    [
        (EngineName.TEI, "1.9.5"),
        (EngineName.MISTRAL, "0.9.6"),
        (EngineName.IK_LLAMA, "0+git0123456789ab"),
    ],
)
def test_each_pin_gets_the_digest_of_its_own_version(
    monkeypatch: pytest.MonkeyPatch, name: EngineName, other: str
) -> None:
    pinned = ADAPTERS[name].pin_record()["version"]
    file = wheel_name(name, pinned, "linux-x86_64-cpu")
    newer = wheel_name(name, other, "linux-x86_64-cpu")
    table = render_table([(newer, MACOS, 2), (file, LINUX, 1)])
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)
    pin = _pin(monkeypatch, name, "Linux", "x86_64")
    assert pin["sha256"] == LINUX
    assert pin["platforms"]["linux-x86_64-cpu"]["sha256"] == LINUX


def test_a_pin_whose_version_is_not_in_the_table_has_no_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    newer = wheel_name(EngineName.TEI, "1.9.5", "linux-x86_64-cpu")
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table([(newer, MACOS, 2)]))
    pin = _pin(monkeypatch, EngineName.TEI, "Linux", "x86_64")
    with pytest.raises(NoPublishedPackError, match=r"No pack of tei is published yet\."):
        pin_lock(pin)


def test_a_published_pack_reaches_the_pin_with_its_url_size_and_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = wheel_name(EngineName.TEI, "1.9.4", "linux-x86_64-cpu", "2.34")
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table([(file, LINUX, 777)]))
    lock = pin_lock(_pin(monkeypatch, EngineName.TEI, "Linux", "x86_64"))
    asset = lock.platforms["linux-x86_64-cpu"]
    assert (asset.sha256, asset.size, asset.min_os) == (LINUX, 777, "2.34")
    assert asset.url.endswith(file)


def test_ik_llama_version_is_a_pep_440_version() -> None:
    from packaging.version import Version

    assert ik_llama.VERSION == "0+git13cffc006b82"
    assert str(Version(ik_llama.VERSION)) == ik_llama.VERSION


def test_ik_llama_pin_record_holds_its_own_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    pin = _pin(monkeypatch, EngineName.IK_LLAMA, "Linux", "x86_64")
    assert pin["launch"][0] == "{binary}"
    assert pin["licenses"] == ["MIT AND Apache-2.0 AND MIT-0 AND Unlicense"]
    assert pin["memory"]["rule"] is None
    shipped = engine_manifest.MANIFESTS["ik_llama.cpp"][ik_llama.VERSION]
    assert pin["platforms"] == {"linux-x86_64-cpu": shipped["linux-x86_64-cpu"]}
    assert pin["sha256"] == shipped["linux-x86_64-cpu"]["sha256"]


_IK_L4 = {"driver": "550.135", "cuda_driver": 12040, "gpus": [{"name": "NVIDIA L4", "cc": [8, 9]}]}
SHIPPED_IK = {
    "cpu": (
        None,
        "linux-x86_64-cpu",
        "96d7a448b522bde5c970767094cb4d8c0e6add596e9ed5f9bb681dfde97e734c",
        "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-3.cpu-py3-none-manylinux_2_34_x86_64.whl",
    ),
    "cuda": (
        _IK_L4,
        "linux-x86_64-cuda",
        "4407cb7986ceb42ce2c1c2c743be5f074317af37ab88f0f33ab17ab589142d1d",
        "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-2.cuda-py3-none-manylinux_2_34_x86_64.whl",
    ),
}


@pytest.mark.parametrize("build", SHIPPED_IK)
def test_each_linux_host_resolves_its_own_ik_pack_from_the_shipped_table(
    build: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvidia, cell, digest, file = SHIPPED_IK[build]
    cpu_info = {"flags": ["avx2"], "brand_raw": "Test CPU"}
    host = host_caps.collect(system="Linux", machine="x86_64", cpu_info=cpu_info, nvidia=nvidia)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = ADAPTERS[EngineName.IK_LLAMA].pin_record()
    assert pin["backend"] == build
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    assert pin["platforms"][cell]["url"] == f"{PACKAGES_URL}engine-packs-20261119-ik2/{file}"


def test_ik_llama_pin_record_does_not_read_the_llama_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(type(LLAMA), "pin_record", lambda _self: {"llama-only-field": True})
    pin = _pin(monkeypatch, EngineName.IK_LLAMA, "Linux", "x86_64")
    assert "llama-only-field" not in pin
    assert set(pin) == {
        "engine",
        "version",
        "upstream_ref",
        "backend",
        "sha256",
        "tool",
        "launch",
        "platforms",
        "memory",
        "licenses",
    }


def test_no_adapter_keeps_a_single_pack_digest() -> None:
    for module in (mistral, tei, ik_llama):
        assert not hasattr(module, "PACK_SHA256")


SHIPPED_LLAMA = {
    ("Darwin", "arm64"): (
        "macos-arm64-metal",
        "99058e20d0e5cb8241187c2fc95f52f9ef840e6aedfb03dd25ba44ac7b4f8537",
        "engine-packs-20261121-llama",
        3,
    ),
    ("Linux", "x86_64"): (
        "linux-x86_64-cpu",
        "4c5dfa351e8baa59e99df0b08cf0ec097f97a5c9cdfdf80ebcb5665cfb90770b",
        "engine-packs-20261121-llama",
        3,
    ),
    ("Windows", "AMD64"): (
        "windows-x86_64-cpu",
        "3ada06318c4b2d4a63b2ac772a0ccc90bb7de23b48f2e87bcf52d427711a0cd6",
        "engine-packs-20261121-llama",
        4,
    ),
}


@pytest.mark.parametrize(("system", "machine"), SHIPPED_LLAMA)
def test_each_host_resolves_its_own_llama_pack_from_the_shipped_table(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest, release, build = SHIPPED_LLAMA[(system, machine)]
    host = _host(system, machine)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = LlamaCppAdapter().pack_pin()
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    url = pin["platforms"][cell]["url"]
    assert f"/{release}/" in url
    assert f"-{build}.{cell.rsplit('-', 1)[1]}-py3-none-" in url


SHIPPED_LLAMA_CUDA = {
    ("Linux", "x86_64"): (
        "linux-x86_64-cuda",
        "39bc41d7ae80c5ae3971a576581cf5849d2cd76714bdc2fc53b735aa3dd0664d",
        "engine-packs-20261121-llama",
    ),
    ("Windows", "AMD64"): (
        "windows-x86_64-cuda",
        "ae205af3bfd9224760884630cfb0bde328597cf4f6378c9631fd28fc01a377fd",
        "engine-packs-20261121-llama",
    ),
}


@pytest.mark.parametrize(("system", "machine"), SHIPPED_LLAMA_CUDA)
def test_a_host_with_an_nvidia_driver_resolves_the_cuda_llama_pack_from_the_shipped_table(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest, release = SHIPPED_LLAMA_CUDA[(system, machine)]
    facts = {"driver": "550.135", "cuda_driver": 12040, "gpus": [{"name": "c", "cc": [6, 1]}]}
    host = host_caps.collect(system=system, machine=machine, cpu_info={}, nvidia=facts)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == "cuda"
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    assert pin["platforms"][cell]["url"].startswith(
        f"https://github.com/tobocop2/lilbee-engine-packs/releases/download/{release}/"
    )


_MISTRAL_RELEASE = (
    "https://github.com/tobocop2/lilbee-engine-packs/releases/download/engine-packs-20261125-mistralrs/"
)

SHIPPED_MISTRAL = {
    ("Darwin", "arm64"): (
        "macos-arm64-metal",
        "f21ef879a4072e4ba3a98910eb87908ae7a219c8d8c0cfbe4a715de3e5f1019a",
    ),
    ("Linux", "x86_64"): (
        "linux-x86_64-cpu",
        "35c70ed067923c185a5275d57c8582c9efac087022fc2b7d5486f6385e9b7ee6",
    ),
    ("Windows", "AMD64"): (
        "windows-x86_64-cpu",
        "f4c22d36171ff99d9e8ec9a11fd46cc0d2edb84fd31bc6194e981304ee377cd7",
    ),
}


@pytest.mark.parametrize(("system", "machine"), SHIPPED_MISTRAL)
def test_each_host_resolves_its_own_mistral_pack_from_the_shipped_table(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest = SHIPPED_MISTRAL[(system, machine)]
    pin = _pin(monkeypatch, EngineName.MISTRAL, system, machine)
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    assert pin["platforms"][cell]["url"].startswith(_MISTRAL_RELEASE)


def test_the_metal_mistral_pack_states_the_macos_15_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    pin = _pin(monkeypatch, EngineName.MISTRAL, "Darwin", "arm64")
    assert pin["platforms"]["macos-arm64-metal"]["min_os"] == "15.0"


SHIPPED_MISTRAL_CUDA = {
    80: "07976991691ce2188eaa4a43293ae7fbb85dac253e8f79bc3cc3fc2f3ee43eae",
    86: "74f15bf25cb9112947175b513533a6438d87016a66c5edaa4dc9de19813b3e2f",
    89: "fbd58aa532ba8f1a6491b4be9c90a247534e25f97d7474f0b0c2fc4d41c3629e",
    90: "9547e8afe2ddcb8a94eea4ac20cc2ba76222adc324d2aa51491b90110173cb44",
    100: "a027766de06b18e2ede8234c08f17d1d4236e5034e62a9eb4950aa7f20402f81",
    120: "83bbfc1c7e20780707b5990fba0386df3a625abf6dd6ca60341efe0cbb42de01",
}


@pytest.mark.parametrize("cc", SHIPPED_MISTRAL_CUDA)
def test_a_linux_host_with_a_cuda_12_8_driver_resolves_the_mistral_pack_of_its_card(
    cc: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = {"driver": "570.86", "cuda_driver": 12080, "gpus": [{"name": "c", "cc": [cc // 10, cc % 10]}]}
    host = host_caps.collect(system="Linux", machine="x86_64", cpu_info={}, nvidia=facts)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    pin = ADAPTERS[EngineName.MISTRAL].pin_record()
    cell = f"linux-x86_64-cuda-sm{cc}"
    assert pin["sha256"] == SHIPPED_MISTRAL_CUDA[cc]
    assert list(pin["platforms"]) == [cell]
    assert pin["platforms"][cell]["url"] == (
        f"{_MISTRAL_RELEASE}lilbee_engine_mistral_rs-0.9.5-2.cuda_sm{cc}-py3-none-manylinux_2_35_x86_64.whl"
    )


class _Reached(Exception):
    """The download check passed and the fetch went on to the disk check."""


def _fetch(monkeypatch: pytest.MonkeyPatch, url: str, tmp_path: Path) -> None:
    def reached(*_args: object) -> str:
        raise _Reached

    monkeypatch.setattr(engine_store, "disk_shortfall", reached)
    asset = engine_store.PlatformAsset(url=url, sha256="0" * 64, size=1, min_os=None)
    lock = engine_store.PinLock(
        engine="llama.cpp",
        version="1",
        backend="cpu",
        platforms={"linux-x86_64-cpu": asset},
        licences=(),
        sha256="0" * 64,
        tool="llama-server",
    )
    engine_store.fetch_frozen(lock, "linux-x86_64-cpu", dest=tmp_path)


def test_the_shipped_pack_base_is_the_release_downloads_of_the_packs_repository() -> None:
    assert PACKAGES_URL == "https://github.com/tobocop2/lilbee-engine-packs/releases/download/"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/other/repo/releases/download/x/y.whl",
        "https://github.com/tobocop2/lilbee/releases/download/x/y.whl",
        "https://github.com/tobocop2/lilbee-engine-packs/archive/y.whl",
        "https://raw.githubusercontent.com/tobocop2/lilbee-engine-packs/main/y.whl",
    ],
)
def test_the_shipped_pack_base_refuses_a_url_outside_the_packs_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str
) -> None:
    with pytest.raises(engine_store.PackUrlError, match="not on the pack index"):
        _fetch(monkeypatch, url, tmp_path)


def test_the_shipped_pack_base_accepts_every_committed_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        row
        for versions in MANIFESTS.values()
        for cells in versions.values()
        for row in cells.values()
    ]
    assert rows
    for row in rows:
        with pytest.raises(_Reached):
            _fetch(monkeypatch, row["url"], tmp_path)
