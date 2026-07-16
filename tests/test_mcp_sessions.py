"""Tests for the MCP session tools (list, get, rename, delete)."""

from __future__ import annotations

import pytest

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.mcp_server import (
    session_add_message,
    session_create,
    session_delete,
    session_get,
    session_rename,
    session_set_summary,
    sessions_list,
)
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


def test_session_get_carries_the_summary(store):
    """An agent resuming a compacted conversation needs the summary."""
    session_id = _seed(store)
    store.set_summary(session_id, "earlier: torque is 85 Nm")
    assert session_get(session_id)["summary"] == "earlier: torque is 85 Nm"


def test_session_create_then_append_and_read_back(store):
    """An agent can own a conversation end to end over MCP."""
    created = session_create(model_ref="qwen3-4b", scope="both")
    session_id = created["id"]
    session_add_message(session_id, "user", "what torque?", ["manual.pdf"])
    session_add_message(session_id, "assistant", "85 Nm.")
    got = session_get(session_id)
    assert got["meta"]["model_ref"] == "qwen3-4b"
    assert [m["content"] for m in got["messages"]] == ["what torque?", "85 Nm."]
    assert got["messages"][0]["sources"] == ["manual.pdf"]


def test_session_add_message_bad_role_errors(store):
    session_id = _seed(store)
    assert "error" in session_add_message(session_id, "wizard", "hi")


def test_session_add_message_unknown_errors(store):
    assert "error" in session_add_message("nope", "user", "hi")


def test_session_set_summary_round_trips(store):
    session_id = _seed(store)
    session_set_summary(session_id, "folded: 85 Nm")
    assert session_get(session_id)["summary"] == "folded: 85 Nm"


def test_session_set_summary_unknown_errors(store):
    assert "error" in session_set_summary("nope", "x")


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
