"""No HTTP route answers a caller who sent no token.

Middleware covers everything, so a new route is authenticated by default. The
only way out is the registry below, and anything in it must prove it checks
the token itself.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.server.app import create_app
from lilbee.server.auth import session_manager
from tests.conftest import make_mock_services

# Routes whose token check runs inside the handler instead of in middleware.
# Both belong to the OpenAI-compatible surface, which must answer a bad token
# with the OpenAI error envelope rather than Litestar's 401 shape. They are
# still authenticated: the assertions below prove it by calling them.
SELF_AUTHENTICATING = {"/v1/models", "/v1/chat/completions"}


@pytest.fixture()
def client(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.wiki = True
    svc_mod.set_services(make_mock_services())
    session_manager.token = "a-real-token"
    session_manager._initialized = True
    with TestClient(app=create_app()) as c:
        yield c
    session_manager.token = None
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _get_routes(app) -> list[tuple[str, str]]:
    """Every (method, path) the app serves, excluding schema and the MCP mount."""
    found = []
    for route in app.routes:
        path = route.path
        if path.startswith(("/schema", "/mcp")):
            continue
        for method in getattr(route, "methods", set()):
            if method in ("HEAD", "OPTIONS"):
                continue
            found.append((method, path))
    return sorted(set(found))


def _concrete(path: str) -> str:
    """Fill path params with a value that will 404, not 401, if auth passes."""
    out = []
    for part in path.split("/"):
        out.append("x" if part.startswith("{") else part)
    return "/".join(out)


class TestNoRouteAnswersWithoutAToken:
    def test_the_app_serves_the_routes_this_test_thinks_it_does(self, client):
        """Guard against the sweep silently covering nothing."""
        routes = _get_routes(client.app)
        assert len(routes) > 30
        paths = {p for _, p in routes}
        for expected in ("/api/health", "/api/export", "/api/search", "/api/wiki"):
            assert expected in paths, expected

    def test_no_route_answers_an_unauthenticated_caller(self, client):
        """A 401 is the only acceptable answer to a request with no token."""
        answered = []
        for method, path in _get_routes(client.app):
            if path in SELF_AUTHENTICATING:
                continue
            resp = client.request(method, _concrete(path), json={})
            if resp.status_code != 401:
                answered.append(f"{method} {path} -> {resp.status_code}")
        assert not answered, "routes answered without a token:\n" + "\n".join(answered)

    @pytest.mark.parametrize("path", sorted(SELF_AUTHENTICATING))
    def test_the_self_authenticating_routes_still_reject(self, client, path):
        """These skip middleware, so they carry the whole burden themselves."""
        method = "GET" if path == "/v1/models" else "POST"
        resp = client.request(method, path, json={"model": "m", "messages": []})
        assert resp.status_code == 401
        # And in the OpenAI envelope, which is the reason they opt out at all.
        assert resp.json()["error"]["code"] == "invalid_api_key"
