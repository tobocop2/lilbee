"""The advice and the install share one OS floor check: the ``min_os`` of the pin's row."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineRefusedError
from lilbee.catalog import runnable
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_record import RECORD_FILE, build_manifests
from lilbee.providers.fleet.engine_store import EngineFolderError, EngineNotInstalledError
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    floor_gate,
    install_advice,
    install_any_advice,
    install_offer,
    install_offered,
)
from lilbee.providers.fleet.engines.host_gate import PASS
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.host_caps import HostCaps, collect, version_numbers

_MAC = {"system": "Darwin", "machine": "arm64", "cpu_info": {}}
_LINUX = {"system": "Linux", "machine": "x86_64", "cpu_info": {"flags": ["avx2"]}}
_LABEL = {"Darwin": "macOS", "Linux": "glibc"}


class _Reached(Exception):
    """The install passed every refusal and went to fetch the pack."""


def _host(monkeypatch: pytest.MonkeyPatch, kind: dict[str, Any], version: str) -> HostCaps:
    """Make this the session host, with *version* as the OS version the floor reads."""
    mac_ver = version if kind is _MAC else ""
    host = collect(**kind, nvidia=None, mac_ver=mac_ver)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: version)
    return host


def _mistral_floor(monkeypatch: pytest.MonkeyPatch, floor: str) -> None:
    """Publish the shipped pack record with the mistral.rs Metal wheel tagged for *floor*."""
    record = resources.files("lilbee.providers.fleet").joinpath(RECORD_FILE)
    text = record.read_text(encoding="utf-8")
    tag = f"macosx_{floor.replace('.', '_')}_arm64"
    table = build_manifests(json.loads(text.replace("macosx_11_0_arm64", tag)))
    rows = table[EngineName.MISTRAL.value]
    assert any(r["min_os"] == floor for cells in rows.values() for r in cells.values())
    monkeypatch.setitem(engine_manifest.MANIFESTS, EngineName.MISTRAL.value, rows)


def _install_refusal(monkeypatch: pytest.MonkeyPatch, name: str) -> str | None:
    """The refusal of an install of *name*, or None when it reaches the fetch."""

    def reach(*_args: object, **_kwargs: object) -> None:
        raise _Reached

    monkeypatch.setattr(engine_service.EngineStore, "ensure", reach)
    try:
        engine_service.install(name)
    except _Reached:
        return None
    except EngineRefusedError as exc:
        return str(exc)
    raise AssertionError("install neither refused nor reached the fetch")


def _binary_error(adapter: EngineAdapter) -> str:
    with pytest.raises((EngineNotInstalledError, ProviderError)) as caught:
        adapter.binary()
    return str(caught.value)


def _check_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    adapter: EngineAdapter,
    kind: dict[str, Any],
    version: str,
    floor: str,
) -> None:
    host = _host(monkeypatch, kind, version)
    name = str(adapter.name)
    label = _LABEL[kind["system"]]
    advised = bool(version) and version_numbers(version) >= version_numbers(floor)
    advice = install_advice(adapter, host)
    refusal = _install_refusal(monkeypatch, name)
    assert (name in engine_service.not_installed()) is advised
    assert install_offer(adapter, host).ok is advised
    assert advice in _binary_error(adapter)
    assert install_offered(name, host) is advised
    assert (f"lilbee engine install {name}" in install_any_advice(host)) is advised
    damaged = str(EngineFolderError(Path("/engines") / name, name))
    assert (f"run 'lilbee engine install {name}'" in damaged) is advised
    if advised:
        assert advice == f"Run lilbee engine install {name}."
        assert refusal is None
        return
    have = f"{label} {version}" if version else f"an unreadable {label} version"
    assert f"needs {label} {floor} or later. This computer has {have}." in advice
    assert refusal is not None
    assert advice in refusal


@pytest.mark.parametrize("version", ["13.6", "14.0"])
def test_tei_is_advised_only_on_a_mac_that_meets_its_published_metal_floor(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    _check_surfaces(monkeypatch, TEI, _MAC, version, "14.0")


@pytest.mark.parametrize("version", ["14.6.1", "15.0"])
def test_mistral_is_advised_only_on_a_mac_that_meets_a_15_metal_floor(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    _mistral_floor(monkeypatch, "15.0")
    _check_surfaces(monkeypatch, MISTRAL, _MAC, version, "15.0")


def test_an_unread_macos_version_is_refused_by_the_advice_as_by_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _check_surfaces(monkeypatch, TEI, _MAC, "", "14.0")


@pytest.mark.parametrize("glibc", ["2.31", "2.34", "2.39"])
def test_tei_is_advised_only_on_a_linux_that_meets_its_published_glibc_floor(
    monkeypatch: pytest.MonkeyPatch, glibc: str
) -> None:
    _check_surfaces(monkeypatch, TEI, _LINUX, glibc, "2.34")


_NOT_A_FLOOR: list[Callable[[dict[str, Any]], dict[str, Any]]] = [
    lambda pin: {**pin, "backend": None},
    lambda pin: {**pin, "platforms": {"windows-x86_64-cpu": next(iter(pin["platforms"].values()))}},
]


@pytest.mark.parametrize("change", _NOT_A_FLOOR, ids=["unusable-pin", "no-cell-here"])
def test_an_unusable_pin_or_a_missing_cell_is_not_a_floor_refusal(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    host = _host(monkeypatch, _LINUX, "2.31")
    assert floor_gate(change(dict(TEI.pin_record())), host) == PASS


def _just_below(floor: str) -> str:
    major, minor, *_ = (*version_numbers(floor), 0)
    return f"{major}.{minor - 1}" if minor else f"{major - 1}.9"


@pytest.mark.parametrize("kind", [_MAC, _LINUX], ids=["macos", "linux"])
def test_the_install_and_the_advice_agree_on_every_row_each_pin_reads(
    monkeypatch: pytest.MonkeyPatch, kind: dict[str, Any]
) -> None:
    checked: set[str] = set()
    for adapter in (a for a in ADAPTERS.values() if not a.bundled):
        _host(monkeypatch, kind, "99.0")
        rows = dict(adapter.pin_record()["platforms"])
        for row in rows.values():
            for version in (_just_below(row["min_os"]), row["min_os"]):
                _check_surfaces(monkeypatch, adapter, kind, version, row["min_os"])
            checked.add(str(adapter.name))
    assert EngineName.TEI.value in checked
