"""Tests for per-server local-model-server URL resolution."""

import pytest

from lilbee.core.config import cfg
from lilbee.providers.local_servers import LM_STUDIO, OLLAMA
from lilbee.providers.local_servers.config_urls import base_url_for, configured_local_servers


class TestBaseUrlFor:
    def test_blank_ollama_uses_spec_default(self, monkeypatch):
        monkeypatch.setattr(cfg, "ollama_base_url", "")
        assert base_url_for("ollama") == OLLAMA.default_base_url

    def test_blank_lm_studio_uses_spec_default(self, monkeypatch):
        monkeypatch.setattr(cfg, "lm_studio_base_url", "")
        assert base_url_for("lm_studio") == LM_STUDIO.default_base_url

    def test_configured_value_overrides_default(self, monkeypatch):
        monkeypatch.setattr(cfg, "ollama_base_url", "http://box:11434")
        assert base_url_for("ollama") == "http://box:11434"

    def test_unknown_server_key_raises(self):
        with pytest.raises(KeyError):
            base_url_for("not-a-server")


class TestConfiguredLocalServers:
    def test_resolves_url_for_every_server(self, monkeypatch):
        monkeypatch.setattr(cfg, "ollama_base_url", "http://box:11434")
        monkeypatch.setattr(cfg, "lm_studio_base_url", "")
        resolved = {spec.key: url for spec, url in configured_local_servers()}
        assert resolved == {
            "ollama": "http://box:11434",
            "lm_studio": LM_STUDIO.default_base_url,
        }
