"""GET /api/agent-config and /api/agent-config/{client}: auth, detection, live config."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml
from litestar.testing import TestClient, create_test_client

from lilbee.app.agent_configs import detect as detect_mod
from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.server import auth as auth_mod
from lilbee.server.app import create_app
from lilbee.server.auth import authenticates_itself, session_manager
from lilbee.server.routes.agent_config import agent_config_index_route, agent_config_route
from tests.conftest import make_mock_services

_TOKEN = "agent-config-token-" + "z" * 40
_PORT = 8765
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_CHAT_CTX = 32768
_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_HTTP_NOT_FOUND = 404


def _manifest(ref: str, task: str) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = task
    return manifest


@pytest.fixture()
def served_state(tmp_path, monkeypatch):
    """A server holding a live token, one installed chat model, and a served window."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.chat_model = _CHAT_REF
    registry = MagicMock()
    registry.list_installed.return_value = [
        _manifest(_CHAT_REF, "chat"),
        _manifest(_EMBED_REF, "embedding"),
    ]
    services = make_mock_services(registry=registry)
    services.provider.served_chat_ctx.return_value = _CHAT_CTX
    set_services(services)
    previous_token = auth_mod.session_manager.token
    previous_init = auth_mod.session_manager._initialized
    session_manager.token = _TOKEN
    session_manager._initialized = True
    monkeypatch.setattr(detect_mod, "find_executable", lambda name: None)
    yield
    session_manager.token = previous_token
    session_manager._initialized = previous_init
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def client(served_state):
    with create_test_client(
        [agent_config_index_route, agent_config_route], base_url=_BASE_URL
    ) as test_client:
        yield test_client


def _bearer() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


class TestAuth:
    """The document embeds the bearer token, so no caller reads it unauthenticated."""

    @pytest.mark.parametrize("path", ["/api/agent-config", "/api/agent-config/opencode"])
    def test_no_token_is_rejected(self, served_state, path):
        with TestClient(app=create_app()) as served:
            assert served.get(path).status_code == _HTTP_UNAUTHORIZED

    @pytest.mark.parametrize("path", ["/api/agent-config", "/api/agent-config/opencode"])
    def test_the_session_token_is_accepted(self, served_state, path):
        with TestClient(app=create_app()) as served:
            # The lifespan mints the boot token; the route answers to that one.
            live = {"Authorization": f"Bearer {session_manager.token}"}
            assert served.get(path, headers=live).status_code == _HTTP_OK

    def test_neither_route_checks_auth_in_the_handler(self):
        """Middleware owns the check; an opt-out here would hand out the token."""
        assert not authenticates_itself(agent_config_index_route.fn)
        assert not authenticates_itself(agent_config_route.fn)


class TestIndex:
    def test_it_lists_every_supported_client(self, client):
        body = client.get("/api/agent-config", headers=_bearer()).json()
        assert [entry["client"] for entry in body["clients"]] == [
            "claude",
            "hermes",
            "opencode",
        ]

    def test_an_installed_cli_reports_where_it_was_found(self, client, monkeypatch):
        monkeypatch.setattr(
            detect_mod,
            "find_executable",
            lambda name: "/opt/homebrew/bin/claude" if name == "claude" else None,
        )
        body = client.get("/api/agent-config", headers=_bearer()).json()
        found = {entry["client"]: entry for entry in body["clients"]}
        assert found["claude"] == {
            "client": "claude",
            "cli_detected": True,
            "cli_path": "/opt/homebrew/bin/claude",
        }
        assert found["opencode"]["cli_detected"] is False
        assert found["opencode"]["cli_path"] is None


class TestOpencodeDocument:
    def test_it_carries_the_live_port_and_current_token(self, client):
        body = client.get("/api/agent-config/opencode", headers=_bearer()).json()
        options = body["config"]["provider"]["lilbee"]["options"]
        assert options["baseURL"] == f"{_BASE_URL}/v1"
        assert options["apiKey"] == _TOKEN

    def test_it_follows_a_rotated_token(self, client):
        """The token is minted per boot; the served document tracks it."""
        session_manager.token = "rotated-token-" + "q" * 40
        body = client.get(
            "/api/agent-config/opencode",
            headers={"Authorization": f"Bearer {session_manager.token}"},
        ).json()
        assert body["config"]["provider"]["lilbee"]["options"]["apiKey"] == session_manager.token

    def test_it_reports_json_and_both_surfaces(self, client):
        body = client.get("/api/agent-config/opencode", headers=_bearer()).json()
        assert body["client"] == "opencode"
        assert body["format"] == "json"
        assert body["surfaces"] == ["model_provider", "mcp"]
        assert body["content"] is None
        assert body["stdio_config"] is None

    def test_it_lists_the_installed_chat_model_with_the_served_window(self, client):
        body = client.get("/api/agent-config/opencode", headers=_bearer()).json()
        models = body["config"]["provider"]["lilbee"]["models"]
        assert list(models) == ["Qwen3-0.6B"]
        assert models["Qwen3-0.6B"]["limit"]["context"] == _CHAT_CTX


class TestHermesDocument:
    def test_it_serves_rendered_yaml(self, client):
        body = client.get("/api/agent-config/hermes", headers=_bearer()).json()
        assert body["format"] == "yaml"
        assert body["config"] is None
        parsed = yaml.safe_load(body["content"])
        assert parsed["providers"]["lilbee"]["api"] == f"{_BASE_URL}/v1"
        assert parsed["providers"]["lilbee"]["api_key"] == _TOKEN
        assert parsed["mcp_servers"]["lilbee"]["headers"]["Authorization"] == f"Bearer {_TOKEN}"


class TestClaudeDocument:
    def test_it_registers_the_running_server_over_http(self, client):
        body = client.get("/api/agent-config/claude", headers=_bearer()).json()
        server = body["config"]["mcpServers"]["lilbee"]
        assert server["type"] == "http"
        assert server["url"] == f"{_BASE_URL}/mcp"
        assert server["headers"]["Authorization"] == f"Bearer {_TOKEN}"

    def test_it_also_offers_the_stdio_registration(self, client):
        body = client.get("/api/agent-config/claude", headers=_bearer()).json()
        assert body["stdio_config"]["mcpServers"]["lilbee"] == {
            "command": "lilbee",
            "args": ["mcp"],
        }

    def test_it_declares_the_mcp_surface_only(self, client):
        """Claude Code brings its own model, so lilbee is not its provider."""
        body = client.get("/api/agent-config/claude", headers=_bearer()).json()
        assert body["surfaces"] == ["mcp"]
        assert "provider" not in body["config"]

    def test_it_does_not_walk_the_model_registry(self, client, served_state):
        """An MCP-only client needs no model state, so the blocking walk is skipped."""
        from lilbee.app.services import get_services

        get_services().registry.list_installed.reset_mock()
        client.get("/api/agent-config/claude", headers=_bearer())
        get_services().registry.list_installed.assert_not_called()


class TestUnknownClient:
    def test_it_is_a_404(self, client):
        assert (
            client.get("/api/agent-config/cursor", headers=_bearer()).status_code == _HTTP_NOT_FOUND
        )

    def test_the_message_names_the_valid_clients(self, client):
        detail = client.get("/api/agent-config/cursor", headers=_bearer()).json()["detail"]
        for name in ("claude", "hermes", "opencode"):
            assert name in detail


class TestBaseUrl:
    def test_the_config_points_back_at_the_host_the_caller_reached(self, served_state):
        """A LAN client must get a URL it can call, not the server's loopback."""
        with create_test_client(
            [agent_config_route], base_url="http://192.168.1.50:9100"
        ) as lan_client:
            body = lan_client.get("/api/agent-config/claude", headers=_bearer()).json()
        assert body["config"]["mcpServers"]["lilbee"]["url"] == "http://192.168.1.50:9100/mcp"


class TestAuthDisabled:
    def test_a_server_with_auth_off_hands_out_an_empty_bearer(self, client):
        """validate() accepts any header once disabled, so an empty token is honest."""
        session_manager.disable()
        body = client.get("/api/agent-config/claude").json()
        assert body["config"]["mcpServers"]["lilbee"]["headers"]["Authorization"] == "Bearer "
