"""Placement and linked_roots writers leave cfg at the resolver's value."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lilbee.app import ingest as app_ingest
from lilbee.app import placement as app_placement
from lilbee.app.reset import perform_reset
from lilbee.config_meta import MODEL_ROLE_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core import settings
from lilbee.core.config import Config, cfg, config_scope
from lilbee.core.config.resolve import ROOT_DERIVED_FIELDS
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3
PINNED = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
SAVED = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(1,))})


def _diverged() -> dict[str, tuple[Any, Any]]:
    """Every overlayable key whose live value differs from a freshly built Config."""
    fresh = Config()
    keys = sorted((set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - ROOT_DERIVED_FIELDS)
    assert "placement" in keys
    assert "linked_roots" in keys
    pairs = {key: (getattr(cfg, key), getattr(fresh, key)) for key in keys}
    return {key: pair for key, pair in pairs.items() if pair[0] != pair[1]}


def _resolved(spec: PlacementSpec | None, **_kw: Any) -> ResolvedPlacement:
    return ResolvedPlacement(
        devices=(
            FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 80 * GIB),
            FleetDevice("CUDA", 1, "NVIDIA A100", 80 * GIB, 80 * GIB),
        ),
        instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0,), tensor_split=None),),
        unplaceable_roles=(),
        model_refs={WorkerRole.CHAT: "org/chat.gguf"},
    )


@pytest.fixture
def no_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan placement without hardware and with nothing running."""
    monkeypatch.setattr(app_placement, "resolve_placement_plan", _resolved)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    monkeypatch.setattr(app_placement, "clear_read_device_cache", lambda: None)
    settings.overlay_persisted_settings(cfg.data_root)


@pytest.fixture
def placement_pinned(monkeypatch: pytest.MonkeyPatch, no_fleet: None) -> None:
    """Pin placement through LILBEE_PLACEMENT, as a restart would read it."""
    monkeypatch.setenv("LILBEE_PLACEMENT", PINNED.to_json())
    settings.overlay_persisted_settings(cfg.data_root)
    assert cfg.placement == PINNED.to_json()


class TestPlacementWriter:
    def test_set_under_env_pin_keeps_env_value_and_records_saved_spec(self, placement_pinned):
        view = app_placement.set_placement(SAVED)
        assert cfg.placement == PINNED.to_json()
        assert settings.load(cfg.data_root)["placement"] == SAVED.to_json()
        assert view.spec_json == PINNED.to_json()
        assert _diverged() == {}

    def test_clear_under_env_pin_keeps_env_value(self, placement_pinned):
        settings.set_value(cfg.data_root, "placement", SAVED.to_json())
        app_placement.set_placement(None)
        assert cfg.placement == PINNED.to_json()
        assert "placement" not in settings.load(cfg.data_root)
        assert _diverged() == {}

    def test_set_without_pin_takes_saved_spec(self, no_fleet):
        view = app_placement.set_placement(SAVED)
        assert cfg.placement == SAVED.to_json()
        assert view.spec_json == SAVED.to_json()
        assert _diverged() == {}

    def test_clear_without_pin_returns_to_auto(self, no_fleet):
        app_placement.set_placement(SAVED)
        view = app_placement.set_placement(None)
        assert cfg.placement is None
        assert view.manual is False
        assert _diverged() == {}


def _corpus(tmp_path: Path, name: str = "corpus") -> Path:
    corpus = tmp_path / name
    corpus.mkdir()
    return corpus


class TestLinkedRootsWriters:
    def test_register_takes_saved_roots(self, tmp_path):
        corpus = _corpus(tmp_path)
        app_ingest.register_sources([corpus])
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}
        assert _diverged() == {}

    def test_register_picks_up_a_root_another_process_saved(self, tmp_path):
        other = _corpus(tmp_path, "other")
        settings.set_value(cfg.data_root, "linked_roots", {"other": str(other.resolve())})
        corpus = _corpus(tmp_path)
        app_ingest.register_sources([corpus])
        assert set(cfg.linked_roots) == {"corpus", "other"}
        assert _diverged() == {}

    def test_register_leaves_cfg_alone_when_the_save_fails(self, tmp_path, monkeypatch):
        def refuse(root: Path, values: dict[str, Any]) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(settings, "save", refuse)
        with pytest.raises(OSError, match="disk full"):
            app_ingest.register_sources([_corpus(tmp_path)])
        assert cfg.linked_roots == {}
        assert _diverged() == {}

    def test_unregister_takes_saved_roots(self, tmp_path):
        app_ingest.register_sources([_corpus(tmp_path)])
        assert app_ingest.unregister_roots(["corpus"]) == ["corpus"]
        assert cfg.linked_roots == {}
        assert _diverged() == {}

    def test_unregister_leaves_cfg_alone_when_the_save_fails(self, tmp_path, monkeypatch):
        corpus = _corpus(tmp_path)
        app_ingest.register_sources([corpus])

        def refuse(root: Path, values: dict[str, Any]) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(settings, "save", refuse)
        with pytest.raises(OSError, match="disk full"):
            app_ingest.unregister_roots(["corpus"])
        assert cfg.linked_roots == {"corpus": str(corpus.resolve())}
        assert _diverged() == {}

    def test_register_under_a_library_scope_sets_the_scoped_config(self, tmp_path):
        scoped = cfg.model_copy(update={"data_root": tmp_path / "library"})
        scoped.data_root.mkdir()
        corpus = _corpus(tmp_path)
        with config_scope(scoped):
            app_ingest.register_sources([corpus])
        assert scoped.linked_roots == {"corpus": str(corpus.resolve())}
        assert settings.load(scoped.data_root)["linked_roots"] == scoped.linked_roots
        assert cfg.linked_roots == {}

    def test_reset_takes_saved_roots(self, tmp_path):
        app_ingest.register_sources([_corpus(tmp_path)])
        perform_reset()
        assert cfg.linked_roots == {}
        assert "linked_roots" not in settings.load(cfg.data_root)
        assert _diverged() == {}
