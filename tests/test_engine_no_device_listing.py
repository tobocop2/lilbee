"""An engine with no device listing says so, so planning never warns about its empty probe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lilbee.providers.fleet.engines import tei as tei_module
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.engines.openvino import OPENVINO
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.engines.tei import TEI

_BINARY = Path("engine-binary")
_CARD = object()


@pytest.mark.parametrize("adapter", [MLX, OPENVINO, LEMONADE, OVMS], ids=lambda a: a.name.value)
def test_an_engine_with_no_probe_reports_no_device_listing(adapter: Any) -> None:
    probe = adapter.discover_devices(_BINARY)
    assert (probe.devices, probe.has_listing) == ([], False)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_an_engine_that_reads_cards_without_a_probe_reports_no_device_listing(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        type(TEI), "pick_build", lambda _self, _host: SimpleNamespace(backend=backend)
    )
    monkeypatch.setattr(tei_module, "visible_nvidia_cards", lambda: [_CARD])
    probe = TEI.discover_devices(_BINARY)
    assert probe.devices == ([_CARD] if backend == "cuda" else [])
    assert probe.has_listing is False


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_mistral_reads_no_card_and_reports_no_device_listing(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        type(MISTRAL), "pick_build", lambda _self, _host: SimpleNamespace(backend=backend)
    )
    probe = MISTRAL.discover_devices(_BINARY)
    assert (probe.devices, probe.has_listing) == ([], False)
