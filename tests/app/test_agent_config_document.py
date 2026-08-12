"""The per-client config document shared by `lilbee agent-config` and /api/agent-config."""

from __future__ import annotations

import pytest
import yaml

from lilbee.app.agent_configs import detect as detect_mod
from lilbee.app.agent_configs.detect import detect_agent_clients
from lilbee.app.agent_configs.document import (
    AgentClient,
    AgentSurface,
    ConfigFormat,
    build_agent_config,
    client_serves_models,
    parse_agent_client,
)

_BASE_URL = "http://127.0.0.1:8765"
_TOKEN = "live-token-abc"
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"


def _build(client: AgentClient):
    return build_agent_config(
        client,
        base_url=_BASE_URL,
        api_key=_TOKEN,
        model_refs=[_CHAT_REF],
        default_ref=_CHAT_REF,
        chat_ctx=32768,
    )


class TestClientParsing:
    def test_every_supported_client_parses(self):
        assert [parse_agent_client(name) for name in ("claude", "hermes", "opencode")] == [
            AgentClient.CLAUDE,
            AgentClient.HERMES,
            AgentClient.OPENCODE,
        ]

    def test_an_unknown_client_names_the_valid_ones(self):
        with pytest.raises(ValueError) as exc:
            parse_agent_client("cursor")
        message = str(exc.value)
        assert "cursor" in message
        for name in ("claude", "hermes", "opencode"):
            assert name in message

    def test_the_error_stays_user_facing(self):
        """No module, function, or dispatch vocabulary leaks into the message."""
        with pytest.raises(ValueError) as exc:
            parse_agent_client("cursor")
        lowered = str(exc.value).lower()
        for internal in ("builder", "dispatch", "handler", "lilbee.app", "enum"):
            assert internal not in lowered


class TestClaudeDocument:
    def test_it_registers_only_the_mcp_surface(self):
        """Claude Code brings its own model, so there is no provider block."""
        document = _build(AgentClient.CLAUDE)
        assert document.surfaces == (AgentSurface.MCP,)
        assert not client_serves_models(AgentClient.CLAUDE)

    def test_the_http_form_carries_the_live_url_and_token(self):
        server = _build(AgentClient.CLAUDE).config["mcpServers"]["lilbee"]
        assert server["type"] == "http"
        assert server["url"] == "http://127.0.0.1:8765/mcp"
        assert server["headers"]["Authorization"] == "Bearer live-token-abc"

    def test_the_stdio_form_spawns_the_lilbee_cli(self):
        server = _build(AgentClient.CLAUDE).stdio_config["mcpServers"]["lilbee"]
        assert server == {"command": "lilbee", "args": ["mcp"]}

    def test_it_is_json_and_carries_no_yaml_content(self):
        document = _build(AgentClient.CLAUDE)
        assert document.format is ConfigFormat.JSON
        assert document.content is None


class TestOpencodeDocument:
    def test_it_matches_what_the_cli_prints(self):
        document = _build(AgentClient.OPENCODE)
        assert document.format is ConfigFormat.JSON
        options = document.config["provider"]["lilbee"]["options"]
        assert options["baseURL"] == "http://127.0.0.1:8765/v1"
        assert options["apiKey"] == _TOKEN
        assert document.config["mcp"]["lilbee"]["url"] == "http://127.0.0.1:8765/mcp"

    def test_it_wires_both_surfaces(self):
        assert _build(AgentClient.OPENCODE).surfaces == (
            AgentSurface.MODEL_PROVIDER,
            AgentSurface.MCP,
        )
        assert client_serves_models(AgentClient.OPENCODE)

    def test_it_carries_the_served_window(self):
        models = _build(AgentClient.OPENCODE).config["provider"]["lilbee"]["models"]
        assert models["Qwen3-0.6B"]["limit"]["context"] == 32768

    def test_it_has_no_stdio_alternative(self):
        assert _build(AgentClient.OPENCODE).stdio_config is None


class TestHermesDocument:
    def test_it_is_rendered_yaml_text(self):
        document = _build(AgentClient.HERMES)
        assert document.format is ConfigFormat.YAML
        assert document.config is None
        parsed = yaml.safe_load(document.content)
        assert parsed["providers"]["lilbee"]["api"] == "http://127.0.0.1:8765/v1"
        assert parsed["providers"]["lilbee"]["api_key"] == _TOKEN
        assert parsed["mcp_servers"]["lilbee"]["url"] == "http://127.0.0.1:8765/mcp"

    def test_it_wires_both_surfaces(self):
        assert AgentSurface.MODEL_PROVIDER in _build(AgentClient.HERMES).surfaces


class TestMissingModelInputs:
    def test_an_mcp_only_document_needs_no_model_state(self):
        """The route skips the registry walk for claude, so the defaults must hold."""
        document = build_agent_config(AgentClient.CLAUDE, base_url=_BASE_URL, api_key=_TOKEN)
        assert document.config["mcpServers"]["lilbee"]["url"] == "http://127.0.0.1:8765/mcp"

    def test_a_provider_document_with_no_models_still_builds(self):
        document = build_agent_config(AgentClient.OPENCODE, base_url=_BASE_URL, api_key=_TOKEN)
        assert document.config["provider"]["lilbee"]["models"] == {}


class TestClientDetection:
    def test_it_reports_a_client_whose_cli_is_installed(self, monkeypatch):
        monkeypatch.setattr(detect_mod, "find_executable", lambda name: f"/opt/homebrew/bin/{name}")
        found = {d.client: d for d in detect_agent_clients()}
        assert found[AgentClient.OPENCODE].cli_detected
        assert found[AgentClient.OPENCODE].cli_path == "/opt/homebrew/bin/opencode"

    def test_it_reports_a_client_whose_cli_is_missing(self, monkeypatch):
        monkeypatch.setattr(detect_mod, "find_executable", lambda name: None)
        for detection in detect_agent_clients():
            assert not detection.cli_detected
            assert detection.cli_path is None

    def test_it_probes_every_supported_client_by_its_own_name(self, monkeypatch):
        probed: list[str] = []

        def _probe(name: str) -> str | None:
            probed.append(name)
            return None

        monkeypatch.setattr(detect_mod, "find_executable", _probe)
        detect_agent_clients()
        assert probed == ["claude", "hermes", "opencode"]
