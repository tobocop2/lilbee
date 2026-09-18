"""Model-role refs that reach cfg without going through the settings boundary.

``LILBEE_CHAT_MODEL``, ``--model`` and ``config.toml`` all write cfg directly,
so the installed check that PATCH / PUT / MCP run never fires for them. These
tests pin the parity rule and the registry's single definition of installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lilbee.catalog.types import ModelSource, ModelTask
from lilbee.core.config import Config, cfg
from lilbee.modelhub.model_manager import ModelManager
from lilbee.modelhub.registry import ModelManifest, ModelRegistry
from lilbee.modelhub.role_validator import (
    _UNREGISTERED_ROLE_WARNING,
    configured_role_refs,
    unregistered_role_refs,
    validate_model_task_assignment,
    warn_unregistered_role_refs,
)
from lilbee.providers.roles import MODEL_ROLE_FIELDS

_REPO = "Qwen/Qwen3-0.6B-GGUF"
_FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"
_REF = f"{_REPO}/{_FILENAME}"
_MISSING_REF = "other/Repo-GGUF/other.gguf"
_BLOB = b"GGUF-bytes"


def _install(models_dir: Path, ref: str = _REF) -> None:
    """Install *ref* the way a real pull does: blob in the cache plus a manifest."""
    registry = ModelRegistry(models_dir)
    hf_repo, filename = ref.rsplit("/", 1)
    source = models_dir / "source.gguf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(_BLOB)
    registry.install(
        hf_repo,
        filename,
        source,
        ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=filename,
            size_bytes=len(_BLOB),
            task=ModelTask.CHAT,
            downloaded_at="2026-04-25T00:00:00+00:00",
        ),
    )
    source.unlink()


def _handbuilt_repo_layout(models_dir: Path, ref: str = _REF) -> None:
    """Place a GGUF at ``models_dir/<ref>`` by hand, with no manifest."""
    target = models_dir / ref
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_BLOB)


class TestPullReportsWhatItDid:
    """`lilbee model pull` must not claim a model that wrote no manifest."""

    def test_handbuilt_layout_is_not_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _handbuilt_repo_layout(models_dir)
        manager = ModelManager(models_dir)

        assert manager.is_installed(_REF, ModelSource.NATIVE) is False
        assert ModelRegistry(models_dir).list_installed() == []
        assert not (models_dir / "manifests").exists()

    def test_installed_answer_matches_the_listing(self, tmp_path: Path) -> None:
        """The pull short-circuit and the listing read one definition of installed."""
        models_dir = tmp_path / "models"
        _handbuilt_repo_layout(models_dir, "other/Repo/other.gguf")
        _install(models_dir)
        manager = ModelManager(models_dir)

        listed = manager.list_installed(ModelSource.NATIVE)
        assert listed == [_REF]
        for ref in (_REF, "other/Repo/other.gguf"):
            assert manager.is_installed(ref, ModelSource.NATIVE) is (ref in listed)

    def test_loose_gguf_is_not_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama3-8b.gguf").write_bytes(_BLOB)
        manager = ModelManager(models_dir)

        assert manager.is_installed("llama3-8b.gguf", ModelSource.NATIVE) is False
        assert manager.get_source("llama3-8b.gguf") is None

    def test_rm_still_deletes_a_loose_gguf(self, tmp_path: Path) -> None:
        """Removal keeps its stray-file sweep so an unregistered GGUF is reclaimable."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        stray = models_dir / "llama3-8b.gguf"
        stray.write_bytes(_BLOB)
        manager = ModelManager(models_dir)

        assert manager.remove("llama3-8b.gguf") is True
        assert not stray.exists()

    def test_registered_model_is_installed(self, tmp_path: Path) -> None:
        """Control: the same check answers True once a manifest exists."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install(models_dir)

        assert ModelManager(models_dir).is_installed(_REF, ModelSource.NATIVE) is True


def _build_config(tmp_path: Path, roles: dict[str, str]) -> Config:
    """A Config built from env only, bound to *tmp_path*, carrying *roles*."""
    env: dict[str, str] = {
        "LILBEE_DATA": str(tmp_path),
        "LILBEE_SKIP_TOML_CONFIG": "1",
    }
    for field_name, ref in roles.items():
        env[f"LILBEE_{field_name.upper()}"] = ref
    with mock.patch.dict(os.environ, env, clear=True):
        return Config()


class TestUnregisteredRoleRefsAreReported:
    """An env-named model that no listing will show is named at startup."""

    def test_absolute_path_chat_model_is_flagged(self, tmp_path: Path) -> None:
        """An absolute GGUF path is named even under the suite-wide validation bypass."""
        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        config = _build_config(tmp_path, {"chat_model": str(gguf)})

        flagged = unregistered_role_refs(config, ModelRegistry(models_dir))

        assert flagged == {"chat_model": str(gguf)}

    def test_every_registry_role_reaches_the_report(self, tmp_path: Path) -> None:
        """Every role the registry declares is reported, with no field list to edit."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        roles = {name: f"missing/Repo-GGUF/{name}.gguf" for name in MODEL_ROLE_FIELDS}
        config = _build_config(tmp_path, roles)

        assert set(configured_role_refs(config)) == MODEL_ROLE_FIELDS
        assert unregistered_role_refs(config, ModelRegistry(models_dir)) == roles

    def test_only_the_unregistered_role_is_flagged(self, tmp_path: Path) -> None:
        """A pulled model is registered; a sibling role that is not is still named."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install(models_dir)
        config = _build_config(tmp_path, {"chat_model": _REF, "vision_model": _MISSING_REF})

        flagged = unregistered_role_refs(config, ModelRegistry(models_dir))

        assert flagged == {"vision_model": _MISSING_REF}

    @pytest.mark.parametrize("ref", ["", "   ", "ollama/qwen3:0.6b", "openai/gpt-4o"])
    def test_blank_and_prefixed_refs_are_not_flagged(self, tmp_path: Path, ref: str) -> None:
        """Excluded refs drop out while an unregistered sibling in the same config stays."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        config = _build_config(tmp_path, {"chat_model": ref, "vision_model": _MISSING_REF})

        flagged = unregistered_role_refs(config, ModelRegistry(models_dir))

        assert flagged == {"vision_model": _MISSING_REF}

    def test_warning_names_the_ref_and_the_fix(self, tmp_path: Path, caplog) -> None:
        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        config = _build_config(tmp_path, {"chat_model": str(gguf)})

        with caplog.at_level(logging.WARNING, logger="lilbee.modelhub.role_validator"):
            warn_unregistered_role_refs(config, ModelRegistry(models_dir))

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert str(gguf) in message
        assert "chat_model" in message
        assert "lilbee model pull" in message

    def test_warning_skips_the_registered_ref(self, tmp_path: Path, caplog) -> None:
        """One warning for the unregistered role, none for the registered one."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install(models_dir)
        config = _build_config(tmp_path, {"chat_model": _REF, "vision_model": _MISSING_REF})

        with caplog.at_level(logging.WARNING, logger="lilbee.modelhub.role_validator"):
            warn_unregistered_role_refs(config, ModelRegistry(models_dir))

        assert [record.getMessage() for record in caplog.records] == [
            _UNREGISTERED_ROLE_WARNING % ("vision_model", _MISSING_REF)
        ]

    def test_services_construction_reports_it(self, tmp_path: Path) -> None:
        """The one call site: every surface builds a container before it serves."""
        from lilbee.app.services import build_services

        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        config = _build_config(tmp_path, {"chat_model": str(gguf)})
        config.models_dir = models_dir

        with (
            mock.patch("lilbee.modelhub.role_validator.warn_unregistered_role_refs") as warn,
            mock.patch("lilbee.data.extract.backends.sync_xberg_backends"),
        ):
            build_services(config, provider=mock.MagicMock())

        assert warn.call_args.args[0] is config


class TestEntryPointParity:
    """Env var, CLI flag and config.toml must reach the same verdict."""

    def _refs_from_env(self, tmp_path: Path, ref: str) -> dict[str, str]:
        return configured_role_refs(_build_config(tmp_path, {"chat_model": ref}))

    def _refs_from_cli(self, tmp_path: Path, ref: str) -> dict[str, str]:
        from lilbee.cli.app import apply_overrides

        config = _build_config(tmp_path, {})
        with mock.patch("lilbee.cli.app.cfg", config):
            apply_overrides(data_dir=tmp_path, model=ref)
        return configured_role_refs(config)

    def _refs_from_toml(self, tmp_path: Path, ref: str) -> dict[str, str]:
        from lilbee.core import settings as persistent_settings

        root = tmp_path / "toml-root"
        root.mkdir()
        persistent_settings.save(root, {"chat_model": ref})
        config = _build_config(tmp_path, {})
        with (
            mock.patch("lilbee.core.settings.cfg", config),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            persistent_settings.overlay_persisted_settings(root)
        return configured_role_refs(config)

    @pytest.mark.parametrize("ref", [_REF, "ollama/qwen3:0.6b"])
    def test_every_entry_point_lands_the_same_ref(self, tmp_path: Path, ref: str) -> None:
        from_env = self._refs_from_env(tmp_path, ref)
        assert self._refs_from_cli(tmp_path, ref) == from_env
        assert self._refs_from_toml(tmp_path, ref) == from_env

    def test_unregistered_verdict_matches_the_write_boundary(self, tmp_path: Path) -> None:
        """The startup report and the settings boundary agree on the same ref.

        Asserted against each other, not against a literal, so the two cannot
        drift apart.
        """
        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        config = _build_config(tmp_path, {"chat_model": str(gguf)})

        flagged = unregistered_role_refs(config, ModelRegistry(models_dir))
        with (
            mock.patch("lilbee.modelhub.role_validator.cfg", config),
            mock.patch("lilbee.modelhub.role_validator.find_pick", return_value=None),
            pytest.raises(ValueError, match="not installed"),
        ):
            validate_model_task_assignment("chat_model", str(gguf), allow_bypass=False)
        assert "chat_model" in flagged


class TestConfigPatchKeepsEnvModels:
    """PATCH /api/config must not disturb a role the environment configured."""

    def _apply(self, tmp_path: Path, updates: dict[str, Any]) -> None:
        from lilbee.app.settings import apply_settings_update

        apply_settings_update(updates, allow_model_roles=False)

    def test_patch_leaves_the_env_role_alone(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "data_root", tmp_path)
        monkeypatch.setattr(cfg, "chat_model", _REF)

        self._apply(tmp_path, {"num_ctx": 4096})

        assert cfg.chat_model == _REF
        assert "chat_model" not in (tmp_path / "config.toml").read_text(encoding="utf-8")

    def test_patch_refuses_a_role_write(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "data_root", tmp_path)
        monkeypatch.setattr(cfg, "chat_model", _REF)

        with pytest.raises(ValueError, match="dedicated model route"):
            self._apply(tmp_path, {"chat_model": "other/Repo/other.gguf"})
        assert cfg.chat_model == _REF

    def test_role_route_still_changes_the_model(self, tmp_path: Path, monkeypatch) -> None:
        """Control: the dedicated route takes effect, so the arm is not immobility."""
        from lilbee.app.settings import apply_settings_update

        monkeypatch.setattr(cfg, "data_root", tmp_path)
        monkeypatch.setattr(cfg, "chat_model", _REF)

        apply_settings_update({"chat_model": "other/Repo/other.gguf"})

        assert cfg.chat_model == "other/Repo/other.gguf"
