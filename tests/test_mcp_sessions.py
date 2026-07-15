"""Tests for the MCP session tools (list, get, rename, delete)."""

from __future__ import annotations

import pytest

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.mcp_server import session_delete, session_get, session_rename, sessions_list
from lilbee.sessions import MessageRole, SessionMessage, TitleSource
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def store():
    services = make_mock_services()
    svc_mod.set_services(services)
    yield services.session_store
    svc_mod.set_services(None)


def _seed(store) -> str:
    session_id = store.create(model_ref="gpt-oss-20b", scope="both")
    store.set_title(session_id, "Torque specs", TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="what specs?"))
    store.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="85 Nm.", sources=("manual.pdf",)),
    )
    return session_id


def test_sessions_list(store):
    session_id = _seed(store)
    result = sessions_list()
    assert result["total"] == 1
    assert result["sessions"][0]["id"] == session_id
    assert result["sessions"][0]["message_count"] == 2


def test_sessions_list_empty(store):
    assert sessions_list() == {"sessions": [], "total": 0}


def test_session_get_returns_transcript(store):
    session_id = _seed(store)
    result = session_get(session_id)
    assert result["meta"]["title"] == "Torque specs"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["sources"] == ["manual.pdf"]


def test_session_get_unknown_errors(store):
    assert "error" in session_get("nope")


def test_session_rename(store):
    session_id = _seed(store)
    assert session_rename(session_id, "Renamed") == {"id": session_id, "title": "Renamed"}
    assert session_get(session_id)["meta"]["title"] == "Renamed"


def test_session_rename_unknown_errors(store):
    assert "error" in session_rename("nope", "x")


def test_session_delete(store):
    session_id = _seed(store)
    assert session_delete(session_id) == {"id": session_id, "deleted": True}
    assert sessions_list()["total"] == 0


def test_session_delete_unknown_errors(store):
    assert "error" in session_delete("nope")
