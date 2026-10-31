"""Tests for the /api/presets routes, run against the real service and a temp config."""

import json
import logging

import pytest

from lilbee.app.presets import engines_refused_message
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.server.routes import presets as routes
from tests._litestar_test_client import create_test_client

KEY = "SENTINEL-rest-key-5e21"
LOCAL = "http://127.0.0.1:8000/v1"
OTHER = "http://127.0.0.1:9000/v1"
ALL_ROUTES = [
    routes.presets_list_route,
    routes.presets_add_route,
    routes.presets_edit_route,
    routes.presets_remove_route,
]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.server_presets = None
    cfg.allow_http_engines = True
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def client():
    with create_test_client(ALL_ROUTES) as cl:
        yield cl


def _stored() -> list[dict]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


def _add(client, **extra):
    body = {"name": "omlx", "url": LOCAL, "roles": ["chat", "embed"], **extra}
    return client.post("/api/presets", json=body)


class TestList:
    def test_empty(self, client):
        res = client.get("/api/presets")
        assert (res.status_code, res.json()) == (200, {"presets": []})

    def test_a_stored_key_shows_as_has_key_only(self, client):
        _add(client, api_key=KEY)
        res = client.get("/api/presets")
        assert res.json()["presets"][0]["has_key"] is True
        assert KEY not in res.text

    def test_list_is_open_when_the_flag_is_off(self, client):
        cfg.allow_http_engines = False
        assert client.get("/api/presets").status_code == 200


class TestAdd:
    def test_creates_and_returns_the_view(self, client):
        res = _add(client, rerank_wire="tei")
        assert res.status_code == 201
        assert res.json() == {
            "name": "omlx",
            "base_url": LOCAL,
            "roles": ["chat", "embed"],
            "has_key": False,
            "valid": True,
            "rerank_wire": "tei",
        }
        assert [e["name"] for e in _stored()] == ["omlx"]

    def test_key_is_stored_and_not_returned(self, client):
        res = _add(client, api_key=KEY)
        assert res.json()["has_key"] is True
        assert KEY not in res.text

    def test_duplicate_is_409_with_the_service_text(self, client):
        _add(client)
        res = _add(client)
        assert (res.status_code, res.json()["detail"]) == (
            409,
            "A server preset named 'omlx' already exists.",
        )

    def test_bad_input_is_422_with_the_service_text(self, client):
        res = _add(client, roles=["chatt"])
        assert res.status_code == 422
        assert res.json()["detail"].startswith("Unknown role 'chatt'")
        assert _stored() == []

    def test_a_locked_config_file_is_422_with_the_write_failure_text(self, client, monkeypatch):
        def locked(*args, **kwargs):
            raise OSError("locked")

        monkeypatch.setattr(settings, "mutate_value", locked)
        res = _add(client)
        assert res.status_code == 422
        assert "locked" in res.json()["detail"]

    def test_wrong_typed_body_does_not_echo_the_key(self, client):
        res = client.post(
            "/api/presets",
            json={"name": "x1", "url": LOCAL, "roles": "chat", "api_key": [KEY]},
        )
        assert res.status_code == 400
        assert KEY not in res.text


class TestEdit:
    def test_changes_roles_and_wire(self, client):
        _add(client)
        res = client.put("/api/presets/omlx", json={"roles": ["rerank"], "rerank_wire": "tei"})
        assert res.status_code == 200
        assert (res.json()["roles"], res.json()["rerank_wire"]) == (["rerank"], "tei")

    def test_clear_key(self, client):
        _add(client, api_key=KEY)
        res = client.put("/api/presets/omlx", json={"clear_key": True})
        assert (res.status_code, res.json()["has_key"]) == (200, False)

    def test_new_origin_without_a_key_is_422(self, client):
        _add(client, api_key=KEY)
        res = client.put("/api/presets/omlx", json={"url": OTHER})
        assert res.status_code == 422
        assert KEY not in res.text

    def test_no_field_is_422(self, client):
        _add(client)
        res = client.put("/api/presets/omlx", json={})
        assert res.status_code == 422
        assert res.json()["detail"].startswith("Nothing to change")

    def test_missing_preset_is_404(self, client):
        res = client.put("/api/presets/nope", json={"url": LOCAL})
        assert (res.status_code, res.json()["detail"]) == (
            404,
            "No server preset is named 'nope'.",
        )


class TestRemove:
    def test_removes(self, client):
        _add(client)
        res = client.delete("/api/presets/omlx")
        assert (res.status_code, res.json()) == (200, {"removed": "omlx"})
        assert _stored() == []

    def test_missing_preset_is_404(self, client):
        assert client.delete("/api/presets/nope").status_code == 404


class TestGate:
    def test_each_write_is_409_when_the_flag_is_off(self, client):
        _add(client, api_key=KEY)
        before = _stored()
        cfg.allow_http_engines = False
        answers = [
            _add(client, name="second"),
            client.put("/api/presets/omlx", json={"roles": ["chat"]}),
            client.delete("/api/presets/omlx"),
        ]
        assert [(r.status_code, r.json()["detail"]) for r in answers] == [
            (409, engines_refused_message())
        ] * 3
        assert _stored() == before

    def test_the_gate_runs_before_the_service_checks(self, client):
        cfg.allow_http_engines = False
        res = _add(client, roles=["chatt"])
        assert (res.status_code, res.json()["detail"]) == (409, engines_refused_message())


def test_no_log_line_holds_the_key(client, caplog):
    caplog.set_level(logging.DEBUG)
    _add(client, api_key=KEY)
    client.put("/api/presets/omlx", json={"api_key": KEY, "url": OTHER})
    client.put("/api/presets/omlx", json={"clear_key": True, "roles": ["chat"]})
    assert caplog.records
    assert KEY not in caplog.text
