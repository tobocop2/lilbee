"""One patch of the session host reaches every reader of it."""

from __future__ import annotations

import pytest

from lilbee.app import engine_choice, engine_service
from lilbee.catalog import runnable
from lilbee.catalog.query import get_catalog
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.fleet import first_use
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import HostCaps, collect

_MAC = collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None)
_GATED = {
    EngineName.LLAMA: frozenset({ModelFormat.GGUF}),
    EngineName.MLX: frozenset({ModelFormat.MLX}),
}


@pytest.fixture
def mac_reads(monkeypatch: pytest.MonkeyPatch) -> list[HostCaps]:
    """Patch only the runnable module; the list records each read of the host.

    MLX is not bundled and the default choice asks the host about it, so every reader reads it.
    """
    reads: list[HostCaps] = []

    def read() -> HostCaps:
        reads.append(_MAC)
        return _MAC

    monkeypatch.setattr(runnable, "session_host", read)
    monkeypatch.setattr(engine_choice, "installed_engines", lambda: _GATED)
    monkeypatch.setattr(engine_service, "installed_engines", lambda: _GATED)
    return reads


def test_engine_choice_reads_the_patched_host(mac_reads: list[HostCaps]) -> None:
    engine_choice.engine_for_format(ModelFormat.GGUF, cfg)
    assert mac_reads


def test_catalog_query_reads_the_patched_host(
    mac_reads: list[HostCaps], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lilbee.catalog.query.get_picks", lambda: [])
    get_catalog(featured=True)
    assert mac_reads


def test_engine_service_reads_the_patched_host(mac_reads: list[HostCaps]) -> None:
    engine_service.offered_engines()
    assert mac_reads


def test_first_use_reads_the_patched_host(mac_reads: list[HostCaps]) -> None:
    first_use.warn_if_gpu_unserved()
    assert mac_reads
