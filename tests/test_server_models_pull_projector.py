"""HTTP pull: a pulled vision model gets the projector check the CLI pull runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app import models as app_models
from lilbee.catalog.download import ProjectorListingError
from lilbee.catalog.types import ModelTask
from lilbee.server import handlers

REF = "acme/Vision-GGUF"


async def _pull_events(*, installed: bool = False) -> list[tuple[str, dict]]:
    manager = MagicMock()
    manager.is_installed.return_value = installed
    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=manager),
    ):
        frames = [e async for e in handlers.models_pull(REF, source="native")]
    manager.pull.assert_called_once()
    events: list[tuple[str, dict]] = []
    for frame in frames:
        if not frame:
            continue
        lines = frame.splitlines()
        name = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = next(ln[6:] for ln in lines if ln.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def _stub_entry(monkeypatch: pytest.MonkeyPatch, task: ModelTask = ModelTask.VISION) -> None:
    monkeypatch.setattr("lilbee.catalog.resolve_pull_target", lambda _q: SimpleNamespace(task=task))


def _stub_listing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_entry: object, **_: object) -> None:
        raise ProjectorListingError("offline")

    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: True)
    monkeypatch.setattr("lilbee.catalog.download_mmproj", _raise)


@pytest.mark.asyncio
async def test_pull_fetches_the_missing_projector(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_entry(monkeypatch)
    missing = iter([True, False])
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: next(missing))
    fetched: list[object] = []
    monkeypatch.setattr(
        "lilbee.catalog.download_mmproj", lambda e, **_: fetched.append(e) or Path("mmproj.gguf")
    )
    events = await _pull_events()
    assert len(fetched) == 1
    assert events == [("done", {})]


@pytest.mark.asyncio
async def test_unreadable_listing_ends_in_done_with_the_cli_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_entry(monkeypatch)
    _stub_listing_error(monkeypatch)
    events = await _pull_events()
    assert events == [("done", {"warning": app_models._unchecked_projector_warning(REF)})]


@pytest.mark.asyncio
async def test_installed_model_with_unreadable_listing_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_entry(monkeypatch)
    _stub_listing_error(monkeypatch)
    events = await _pull_events(installed=True)
    assert events == [("done", {})]


@pytest.mark.asyncio
async def test_repo_without_projector_ends_in_done_with_the_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_entry(monkeypatch)
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: True)
    monkeypatch.setattr("lilbee.catalog.download_mmproj", lambda _e, **_: None)
    events = await _pull_events()
    assert events == [("done", {"warning": app_models._no_projector_warning(REF)})]


@pytest.mark.asyncio
async def test_fetched_projector_the_engine_misses_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_entry(monkeypatch)
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: True)
    monkeypatch.setattr("lilbee.catalog.download_mmproj", lambda _e, **_: Path("mmproj.gguf"))
    events = await _pull_events()
    assert [name for name, _ in events] == ["error"]


@pytest.mark.asyncio
async def test_pull_passes_the_client_cancel_to_the_projector_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_entry(monkeypatch)
    missing = iter([True, False])
    monkeypatch.setattr(app_models, "_vision_projector_missing", lambda _r: next(missing))
    signals: list[object] = []

    def _fetch(_entry: object, *, cancel: object = None, **_: object) -> Path:
        signals.append(cancel)
        return Path("mmproj.gguf")

    monkeypatch.setattr("lilbee.catalog.download_mmproj", _fetch)
    await _pull_events()
    assert len(signals) == 1
    assert signals[0] is not None
    assert hasattr(signals[0], "is_set")
