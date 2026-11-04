"""Pack digests come from the generated digest table, one per cell."""

from __future__ import annotations

import re
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest, host_caps
from lilbee.providers.fleet.engine_digests import PACKAGES_URL
from lilbee.providers.fleet.engines import ik_llama, mistral, tei
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS, NoPublishedPackError, pin_lock
from lilbee.providers.fleet.engines.llama import LLAMA
from tests._published import render_table, wheel_name

LINUX = "a" * 64
WINDOWS = "b" * 64
MACOS = "c" * 64


def _host(system: str, machine: str) -> host_caps.HostCaps:
    return host_caps.collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"], "brand_raw": "Test CPU"},
        nvml=None,
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


def test_every_shipped_row_lives_under_the_pack_index_and_matches_its_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert engine_manifest.MANIFESTS
    for engine, versions in engine_manifest.MANIFESTS.items():
        pin = _pin(monkeypatch, EngineName(engine), "Linux", "x86_64")
        assert pin["engine"] == engine
        assert list(versions) == [pin["version"]]
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
    monkeypatch.setattr(LLAMA, "pin_record", lambda: {"llama-only-field": True})
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
