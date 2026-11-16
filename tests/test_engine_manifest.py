"""Pack digests come from the generated digest table, one per cell."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest, engine_store, first_use, host_caps
from lilbee.providers.fleet.engine_digests import MANIFESTS, PACKAGES_URL
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


PUBLISHED_SIZES: dict[str, int] = json.loads(
    (Path(__file__).parent / "fixtures" / "published_pack_sizes.json").read_text(encoding="utf-8")
)


def _shipped_rows() -> dict[str, int]:
    return {
        row["url"].removeprefix(PACKAGES_URL): row["size"]
        for versions in engine_manifest.MANIFESTS.values()
        for cells in versions.values()
        for row in cells.values()
    }


@pytest.mark.parametrize("asset", sorted(_shipped_rows()))
def test_a_shipped_row_has_the_size_of_its_published_asset(asset: str) -> None:
    assert _shipped_rows()[asset] == PUBLISHED_SIZES[asset]


def test_the_recorded_sizes_and_the_shipped_rows_name_the_same_assets() -> None:
    assert PUBLISHED_SIZES
    assert set(_shipped_rows()) == set(PUBLISHED_SIZES)
    assert {asset.split("/")[1].split("-")[0] for asset in PUBLISHED_SIZES} == {
        "lilbee_engine_llama_cpp",
        "lilbee_engine_ik_llama_cpp",
        "lilbee_engine_mistral_rs",
        "lilbee_engine_tei",
    }


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


_TEI_RELEASE = "https://github.com/tobocop2/lilbee/releases/download/engine-packs-20261116-tei-2/"
SHIPPED_TEI = {
    "cpu": (
        "linux-x86_64-cpu",
        "a92f666c0cdc825f173c3e29486c409a4b5435d037f57d3a60f970844eaf09b8",
    ),
    "metal": (
        "macos-arm64-metal",
        "adef1ba024f7fd18e65f335820cd9d6169eee5a3ca823108650bf67a45afed04",
    ),
    "sm75": (
        "linux-x86_64-cuda-sm75",
        "1245143a609c9669950602affa9133e1ec5730f334c4f0f585015e84e1f518f4",
    ),
    "sm80": (
        "linux-x86_64-cuda-sm80",
        "1fd3fd7a9b50db3f0b5bae956f6fb8c2865dd23c0c454ef7e4d6a54740343cfd",
    ),
    "sm86": (
        "linux-x86_64-cuda-sm86",
        "538c78b6280c25df33acd6bf9fb0c02291a1056ff0091aad20c529b2b824fba1",
    ),
    "sm89": (
        "linux-x86_64-cuda-sm89",
        "e697869e269c936a99e39244a2098d2c0c5ee7e8ae7798de48d3714ad4a4a19b",
    ),
    "sm90": (
        "linux-x86_64-cuda-sm90",
        "61258fdcccd2322665ab3523c309204a986c2f461e843449cf1e53aeac63dca7",
    ),
    "sm100": (
        "linux-x86_64-cuda-sm100",
        "9fc0a9bff5b4a69b290858f6d761c629e4050727e76de22a29d216d20e920880",
    ),
    "sm120": (
        "linux-x86_64-cuda-sm120",
        "0482fe2efecfaed8cea2e90f6dd6f0978aa7b73878d93fd972b4bf86741ac656",
    ),
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


@pytest.mark.parametrize("build", SHIPPED_TEI)
def test_each_host_resolves_its_own_tei_pack_from_the_shipped_table(
    build: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest = SHIPPED_TEI[build]
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
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
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
    assert pin["licenses"] == ["MIT"]
    assert pin["memory"]["rule"] is None
    shipped = engine_manifest.MANIFESTS["ik_llama.cpp"][ik_llama.VERSION]
    assert pin["platforms"] == shipped
    assert pin["sha256"] == shipped["linux-x86_64-cpu"]["sha256"]


def test_ik_llama_pin_record_does_not_read_the_llama_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(type(LLAMA), "pin_record", lambda _self: {"llama-only-field": True})
    pin = _pin(monkeypatch, EngineName.IK_LLAMA, "Linux", "x86_64")
    assert "llama-only-field" not in pin
    assert set(pin) == {
        "engine",
        "version",
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
        "ddec445038beea9c83ff50a540ddf0e88a3b70c06edec301ef458cd1a46580ff",
    ),
    ("Linux", "x86_64"): (
        "linux-x86_64-cpu",
        "b306bfa9ede1eaef1ce762fe1c772905ec2e10cfc315ea02b58138e77688899b",
    ),
    ("Windows", "AMD64"): (
        "windows-x86_64-cpu",
        "91e9670756cf84252d8382050c74901ef4fc5f07fa7af803fef8bda5438c938d",
    ),
}


@pytest.mark.parametrize(("system", "machine"), SHIPPED_LLAMA)
def test_each_host_resolves_its_own_llama_pack_from_the_shipped_table(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest = SHIPPED_LLAMA[(system, machine)]
    host = _host(system, machine)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    pin = LlamaCppAdapter().pack_pin()
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    assert f"-1.{cell.rsplit('-', 1)[1]}-py3-none-" in pin["platforms"][cell]["url"]


SHIPPED_LLAMA_CUDA = {
    ("Linux", "x86_64"): (
        "linux-x86_64-cuda",
        "6cdd44291d036a352fd496b2a2c146f0a5aaf35cd38f73e8f795ca5a2d87edd4",
    ),
    ("Windows", "AMD64"): (
        "windows-x86_64-cuda",
        "975be314c81ab5c518c40be68fd72edb080b4e4d3f3ba4adbbb3380985cd117c",
    ),
}


@pytest.mark.parametrize(("system", "machine"), SHIPPED_LLAMA_CUDA)
def test_a_host_with_an_nvidia_driver_resolves_the_cuda_llama_pack_from_the_shipped_table(
    system: str, machine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, digest = SHIPPED_LLAMA_CUDA[(system, machine)]
    facts = {"driver": "550.135", "cuda_driver": 12040, "gpus": [{"name": "c", "cc": [6, 1]}]}
    host = host_caps.collect(system=system, machine=machine, cpu_info={}, nvidia=facts)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == "cuda"
    assert pin["sha256"] == digest
    assert list(pin["platforms"]) == [cell]
    assert pin["platforms"][cell]["url"].startswith(
        "https://github.com/tobocop2/lilbee/releases/download/engine-packs-20261105-cuda/"
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


def test_the_shipped_pack_base_is_the_release_downloads_of_lilbee() -> None:
    assert PACKAGES_URL == "https://github.com/tobocop2/lilbee/releases/download/"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/other/repo/releases/download/x/y.whl",
        "https://github.com/tobocop2/lilbee/archive/y.whl",
        "https://raw.githubusercontent.com/tobocop2/lilbee/main/y.whl",
    ],
)
def test_the_shipped_pack_base_refuses_a_url_outside_the_lilbee_releases(
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
