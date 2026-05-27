"""Direct tests for the engine-neutral parameter helpers.

FleetProvider monkeypatches these in its own tests, so the real bodies need
direct coverage here.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers import engine_params as ep
from lilbee.providers.base import ProviderError


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestResolveModelPath:
    def test_absolute_existing_path_when_registry_misses(self, tmp_path) -> None:
        from lilbee.app.services import set_services
        from tests.conftest import make_mock_services

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        registry = MagicMock()
        registry.resolve.side_effect = KeyError("not in registry")
        set_services(make_mock_services(registry=registry))
        try:
            assert ep.resolve_model_path(str(model)) == model
        finally:
            set_services(None)

    def test_absolute_missing_path_raises(self, tmp_path) -> None:
        from lilbee.app.services import set_services
        from tests.conftest import make_mock_services

        # A child of tmp_path is absolute on every OS (a leading-slash literal is
        # not absolute on Windows) and missing because we never create it.
        missing = tmp_path / "missing" / "model.gguf"
        registry = MagicMock()
        registry.resolve.side_effect = KeyError("not in registry")
        set_services(make_mock_services(registry=registry))
        try:
            with pytest.raises(ProviderError, match="Model file not found") as exc_info:
                ep.resolve_model_path(str(missing))
            from lilbee.providers.base import ProviderErrorKind

            assert exc_info.value.kind is ProviderErrorKind.NOT_FOUND
        finally:
            set_services(None)

    def test_registry_miss_raises_not_found_naming_model(self) -> None:
        """A registry miss for a relative ref is a NOT_FOUND ProviderError whose
        message names the model and the pull command (F3 root cause)."""
        from lilbee.app.services import set_services
        from lilbee.providers.base import ProviderErrorKind
        from tests.conftest import make_mock_services

        registry = MagicMock()
        registry.resolve.side_effect = KeyError("not in registry")
        set_services(make_mock_services(registry=registry))
        try:
            with pytest.raises(ProviderError) as exc_info:
                ep.resolve_model_path("nomic-ai/embed/embed.gguf")
            assert exc_info.value.kind is ProviderErrorKind.NOT_FOUND
            message = str(exc_info.value)
            assert "nomic-ai/embed/embed.gguf" in message
            assert "lilbee model pull nomic-ai/embed/embed.gguf" in message
        finally:
            set_services(None)


class TestResolveChatCtx:
    def test_uses_dynamic_sizing(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 100)
        monkeypatch.setattr(ep, "train_ctx_from_meta", lambda *a, **k: 8192)
        monkeypatch.setattr(ep, "get_available_memory", lambda _frac: 10**10)
        monkeypatch.setattr(ep, "kv_bytes_per_token", lambda _meta, _b: 1000)
        monkeypatch.setattr(ep, "compute_dynamic_ctx", lambda **_k: 6000)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 8192)
        assert ep.resolve_chat_ctx(model, {"arch": "x"}) == 6000

    def test_honors_num_ctx_max_ceiling(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 100)
        seen: dict = {}
        monkeypatch.setattr(ep, "train_ctx_from_meta", lambda *a, **k: 99999)
        monkeypatch.setattr(ep, "get_available_memory", lambda _frac: 10**10)
        monkeypatch.setattr(ep, "kv_bytes_per_token", lambda _meta, _b: 1000)

        def _capture(**kwargs):
            seen.update(kwargs)
            return 4096

        monkeypatch.setattr(ep, "compute_dynamic_ctx", _capture)
        monkeypatch.setattr(cfg, "num_ctx_max", 4096)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 8192)
        ep.resolve_chat_ctx(model, None)
        assert seen["ceiling"] == 4096  # explicit num_ctx_max caps below training_ctx

    def test_falls_back_to_static_cap_on_stat_error(self, monkeypatch) -> None:
        monkeypatch.setattr(ep, "train_ctx_from_meta", lambda *a, **k: 8192)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 4096)
        # A nonexistent path makes .stat() raise OSError -> static min(training, target).
        assert ep.resolve_chat_ctx(Path("/nonexistent/x.gguf"), None) == 4096


class TestResolveNGpuLayers:
    def test_embedding_offloads_all(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "n_gpu_layers", 20)
        assert ep.resolve_n_gpu_layers(embedding=True) == -1

    def test_chat_uses_configured_value(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "n_gpu_layers", 20)
        assert ep.resolve_n_gpu_layers(embedding=False) == 20

    def test_none_offloads_all(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "n_gpu_layers", None)
        assert ep.resolve_n_gpu_layers(embedding=False) == -1


class TestResolveVisionCtx:
    def test_reads_training_ctx(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(ep, "read_gguf_metadata", lambda _p: {"arch": "y"})
        monkeypatch.setattr(ep, "train_ctx_from_meta", lambda _meta, **_k: 4321)
        assert ep.resolve_vision_ctx(model) == 4321

    def test_falls_back_when_metadata_unreadable(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x")

        def _boom(_p: Path) -> dict:
            raise OSError("unreadable")

        monkeypatch.setattr(ep, "read_gguf_metadata", _boom)
        monkeypatch.setattr(ep, "train_ctx_from_meta", lambda _meta, *, fallback, **_k: fallback)
        assert ep.resolve_vision_ctx(model) == 4096  # _VISION_FALLBACK_N_CTX


def test_apply_vulkan_loader_safety_disables_layers_and_icds(monkeypatch) -> None:
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_select import VulkanIcdEnvVar

    monkeypatch.setattr(gpu_env.sys, "platform", "linux")
    monkeypatch.setattr(
        "lilbee.providers.fleet.gpu_select.disable_conflicting_vulkan_icds",
        lambda: "/etc/vulkan/icd.d/other.json",
    )
    monkeypatch.delenv(gpu_env._VK_LOADER_LAYERS_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(VulkanIcdEnvVar.LOADER_DRIVERS_DISABLE, raising=False)

    gpu_env._apply_vulkan_loader_safety()

    assert (
        os.environ[gpu_env._VK_LOADER_LAYERS_DISABLE_ENV_VAR]
        == gpu_env._VK_LOADER_LAYERS_DISABLE_VALUE
    )
    assert os.environ[VulkanIcdEnvVar.LOADER_DRIVERS_DISABLE] == "/etc/vulkan/icd.d/other.json"
