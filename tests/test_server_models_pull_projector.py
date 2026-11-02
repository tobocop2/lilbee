"""HTTP pull: a pulled vision model gets the projector check the CLI pull runs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app import models as app_models
from lilbee.catalog.download import ProjectorListingError
from lilbee.catalog.types import ModelTask
from lilbee.server import handlers

REF = "acme/Vision-GGUF"


async def _pull_events() -> list[str]:
    manager = MagicMock()
    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=manager),
    ):
        events = [e async for e in handlers.models_pull(REF, source="native")]
    manager.pull.assert_called_once()
    return [e for e in events if e]


def _stub_entry(monkeypatch: pytest.MonkeyPatch, task: ModelTask = ModelTask.VISION) -> None:
    monkeypatch.setattr("lilbee.catalog.resolve_pull_target", lambda _q: SimpleNamespace(task=task))


@pytest.mark.asyncio
async def test_pull_fetches_the_missing_projector(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_entry(monkeypatch)
    missing = iter([True, False])
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: next(missing))
    fetched: list[object] = []
    monkeypatch.setattr(
        "lilbee.catalog.download_mmproj", lambda e: fetched.append(e) or Path("mmproj.gguf")
    )
    events = await _pull_events()
    assert len(fetched) == 1
    assert not any("error" in e for e in events)


@pytest.mark.asyncio
async def test_unreadable_listing_streams_the_cli_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_entry(monkeypatch)
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: True)

    def _raise(_entry: object) -> None:
        raise ProjectorListingError("offline")

    monkeypatch.setattr("lilbee.catalog.download_mmproj", _raise)
    events = await _pull_events()
    assert any(app_models._unchecked_projector_warning(REF) in e for e in events)


@pytest.mark.asyncio
async def test_repo_without_projector_streams_the_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_entry(monkeypatch)
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: True)
    monkeypatch.setattr("lilbee.catalog.download_mmproj", lambda _e: None)
    events = await _pull_events()
    assert any(app_models._no_projector_warning(REF) in e for e in events)
