"""A safetensors embedding model sets the store width from its own config files, on every path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from lilbee.app import settings as appset
from lilbee.app.services import set_services
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry, register_downloaded_model
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS, NoEngineError

_REPO = "acme/wide-embedder"
_REF = f"{_REPO}#safetensors"
_REV = "0" * 40
_WIDTH = 1024
_DEFAULT_WIDTH = 768


def _blob(cache: Path, name: str, content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    (cache / "blobs").mkdir(parents=True, exist_ok=True)
    (cache / "blobs" / digest).write_bytes(content)
    link = cache / "snapshots" / _REV / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(cache / "blobs" / digest)


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    """A registry holding one installed safetensors embedder of width 1024."""
    models = tmp_path / "models"
    cache = models / "models--acme--wide-embedder"
    _blob(cache, "model.safetensors", b"w" * 100)
    _blob(cache, "config.json", json.dumps({"hidden_size": _WIDTH}).encode())
    entry = CatalogModel(
        hf_repo=_REPO,
        gguf_filename="model.safetensors",
        size_gb=0.0,
        min_ram_gb=1.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.EMBEDDING,
        format=ModelFormat.SAFETENSORS,
    )
    monkeypatch.setattr(cfg, "models_dir", models)
    register_downloaded_model(entry, cache / "snapshots" / _REV / "model.safetensors")
    reg = ModelRegistry(models)
    set_services(mock.MagicMock(registry=reg, store=mock.MagicMock()))
    cfg.embedding_dim = _DEFAULT_WIDTH
    return reg


def test_the_installed_folder_is_the_width_source(registry: ModelRegistry) -> None:
    assert appset._embedder_dim(_REF, registry) == _WIDTH


def test_startup_reconcile_pins_the_folder_width(registry: ModelRegistry) -> None:
    cfg.embedding_model = _REF
    appset.reconcile_embedding_dim(registry)
    assert cfg.embedding_dim == _WIDTH


def test_a_settings_write_derives_the_folder_width(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(appset, "_pin_legacy_store_meta", lambda: None)
    monkeypatch.setattr(appset, "_invalidate_caches", lambda keys: None)
    monkeypatch.setattr(appset, "_embed_reindex_required", lambda: False)
    appset.apply_settings_update({"embedding_model": _REF})
    assert (cfg.embedding_model, cfg.embedding_dim) == (_REF, _WIDTH)


def test_a_one_run_swap_derives_the_folder_width(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(appset, "_pin_legacy_store_meta", lambda: None)
    appset.apply_ephemeral_model_swap("embedding_model", _REF)
    assert (cfg.embedding_model, cfg.embedding_dim) == (_REF, _WIDTH)


def test_the_picker_asks_the_store_with_the_folder_width(registry: ModelRegistry) -> None:
    store = mock.MagicMock()
    store.index_mismatch.return_value = None
    set_services(mock.MagicMock(registry=registry, store=store))
    appset.embed_pick_needs_rebuild(_REF)
    store.index_mismatch.assert_called_once_with(model=_REF, dim=_WIDTH)


def test_a_one_run_chat_swap_sets_the_chat_model(registry: ModelRegistry) -> None:
    cfg.chat_model = "acme/before-GGUF/before.gguf"
    appset.apply_ephemeral_model_swap("chat_model", "acme/after-GGUF/after.gguf")
    assert cfg.chat_model == "acme/after-GGUF/after.gguf"


def test_startup_reconcile_uses_the_registry_it_is_given(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    other = ModelRegistry(tmp_path / "empty")
    set_services(mock.MagicMock(registry=other, store=mock.MagicMock()))
    cfg.embedding_model = _REF
    appset.reconcile_embedding_dim(registry)
    assert cfg.embedding_dim == _WIDTH


def test_the_header_reader_gets_the_resolved_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def read(path: Path) -> dict[str, str]:
        seen.append(path)
        return {"embedding_length": "1"}

    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda r, g=None: Path("/m/e.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", read)
    assert appset._embedder_dim("ref") == 1
    assert seen == [Path("/m/e.gguf")]


def test_a_header_reader_that_finds_nothing_gives_no_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda r, g=None: Path("/m/e.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda p: None)
    assert appset._embedder_dim("ref") is None


def test_an_ephemeral_swap_to_a_model_with_no_known_width_keeps_the_stored_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appset, "_pin_legacy_store_meta", lambda: None)
    monkeypatch.setattr(appset, "_embedder_dim", lambda ref: None)
    monkeypatch.setattr(cfg, "embedding_model", "old/ref")
    monkeypatch.setattr(cfg, "embedding_dim", 768)
    appset.apply_ephemeral_model_swap("embedding_model", "new/ref")
    assert cfg.embedding_model == "new/ref"
    assert cfg.embedding_dim == 768


_POOLED_WIDTH = 512


@pytest.fixture
def two_mode_registry(registry: ModelRegistry) -> ModelRegistry:
    """The installed embedder, with a pooling step that enables two modes over 512."""
    snapshot = cfg.models_dir / "models--acme--wide-embedder" / "snapshots" / _REV
    (snapshot / "modules.json").write_text(
        json.dumps([{"type": "pipeline.step.Pooling", "path": "1_Pooling"}]), encoding="utf-8"
    )
    (snapshot / "1_Pooling").mkdir()
    (snapshot / "1_Pooling" / "config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": _POOLED_WIDTH,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_lasttoken": True,
            }
        ),
        encoding="utf-8",
    )
    return registry


def _served_by(monkeypatch: pytest.MonkeyPatch, name: EngineName) -> None:
    monkeypatch.setattr(
        "lilbee.app.engine_choice.engine_for_ref", lambda ref, config: ADAPTERS[name]
    )


def test_an_engine_that_serves_one_pooling_mode_gets_one_block(
    two_mode_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served_by(monkeypatch, EngineName.TEI)
    assert appset._embedder_dim(_REF, two_mode_registry) == _POOLED_WIDTH


def test_an_engine_that_concatenates_pooling_modes_gets_one_block_per_mode(
    two_mode_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _served_by(monkeypatch, EngineName.MISTRAL)
    assert appset._embedder_dim(_REF, two_mode_registry) == 2 * _POOLED_WIDTH


def test_no_engine_for_the_model_gets_one_block(
    two_mode_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(ref: str, config: object) -> object:
        raise NoEngineError("no engine")

    monkeypatch.setattr("lilbee.app.engine_choice.engine_for_ref", refuse)
    assert appset._embedder_dim(_REF, two_mode_registry) == _POOLED_WIDTH


def test_get_services_pins_the_width_before_the_store_is_built(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.app import services as services_mod
    from tests.conftest import make_mock_services

    seen: list[int] = []

    def build(config, provider=None, registry=None, interactive=False):  # type: ignore[no-untyped-def]
        seen.append(config.embedding_dim)
        return make_mock_services()

    monkeypatch.setattr(services_mod, "build_services", build)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    monkeypatch.setattr(services_mod._state, "singleton", None)
    cfg.embedding_model = _REF
    services_mod.get_services()
    assert seen == [_WIDTH]


def test_the_picker_judges_the_canonical_ref_of_a_bare_pick(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With role validation on, a bare repo pick reaches the store as the installed ref."""
    monkeypatch.delenv("LILBEE_SKIP_MODEL_TASK_VALIDATION")
    monkeypatch.setattr(
        "lilbee.modelhub.role_validator._installed_ref_and_task",
        lambda ref, want: ("acme/old-GGUF/old.gguf", ModelTask.EMBEDDING),
    )
    store = mock.MagicMock()
    store.index_mismatch.return_value = None
    set_services(mock.MagicMock(registry=registry, store=store))
    appset.embed_pick_needs_rebuild("acme/old-GGUF")
    assert store.index_mismatch.call_args.kwargs["model"] == "acme/old-GGUF/old.gguf"
