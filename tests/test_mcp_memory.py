"""Tests for the MCP memory tools and agent-owner derivation."""

from unittest.mock import MagicMock

from lilbee import mcp_server
from lilbee.app.memory import MEMORY_DISABLED_HINT
from lilbee.data.store import MemoryKind, MemoryRow, MemorySource


def _row(text: str, *, kind: MemoryKind = MemoryKind.FACT, shared: bool = False) -> MemoryRow:
    return MemoryRow(
        id="abc123",
        owner="agent:opencode",
        shared=shared,
        kind=kind,
        source=MemorySource.AGENT,
        text=text,
        vector=[0.0],
        created_at="t",
        updated_at="t",
    )


def _ctx(name: str | None) -> MagicMock:
    ctx = MagicMock()
    if name is None:
        ctx.session.client_params = None
    else:
        ctx.session.client_params.clientInfo.name = name
    return ctx


class TestSlug:
    def test_lowercases_and_hyphenates(self):
        assert mcp_server._slug("Open Code") == "open-code"
        assert mcp_server._slug("claude.ai") == "claude-ai"

    def test_empty_falls_back_to_generic(self):
        assert mcp_server._slug("") == "generic"
        assert mcp_server._slug("***") == "generic"


class TestClientName:
    def test_none_ctx(self):
        assert mcp_server._client_name(None) == ""

    def test_uninitialized_session(self):
        assert mcp_server._client_name(_ctx(None)) == ""

    def test_reads_client_info(self):
        assert mcp_server._client_name(_ctx("opencode")) == "opencode"


class TestDeriveOwner:
    def test_explicit_agent_id_wins(self, monkeypatch):
        monkeypatch.setenv("LILBEE_AGENT_ID", "from-env")
        assert mcp_server._derive_owner("Explicit", _ctx("client")) == "agent:explicit"

    def test_env_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("LILBEE_AGENT_ID", "from-env")
        assert mcp_server._derive_owner("", _ctx("client")) == "agent:from-env"

    def test_client_name_when_no_explicit_or_env(self, monkeypatch):
        monkeypatch.delenv("LILBEE_AGENT_ID", raising=False)
        assert mcp_server._derive_owner("", _ctx("OpenCode")) == "agent:opencode"

    def test_generic_when_nothing(self, monkeypatch):
        monkeypatch.delenv("LILBEE_AGENT_ID", raising=False)
        assert mcp_server._derive_owner("", None) == "agent:generic"


class TestMemoryRememberTool:
    def test_disabled_returns_hint(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: False)
        assert mcp_server.memory_remember("x") == {"error": MEMORY_DISABLED_HINT}

    def test_stores_under_derived_agent_owner(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: True)
        remember = MagicMock(return_value="mid")
        monkeypatch.setattr(mcp_server, "remember", remember)
        result = mcp_server.memory_remember(
            "uses rust", kind=MemoryKind.PREFERENCE, shared=True, agent_id="opencode"
        )
        assert result == {"ok": True, "id": "mid", "owner": "agent:opencode"}
        remember.assert_called_once_with(
            "uses rust",
            owner="agent:opencode",
            kind=MemoryKind.PREFERENCE,
            source=MemorySource.AGENT,
            shared=True,
        )


class TestMemoryRecallTool:
    def test_disabled_returns_hint(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: False)
        assert mcp_server.memory_recall("q") == {"error": MEMORY_DISABLED_HINT}

    def test_returns_serialized_memories(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: True)
        monkeypatch.setattr(mcp_server, "recall", MagicMock(return_value=[_row("uses rust")]))
        result = mcp_server.memory_recall("q", agent_id="opencode")
        assert result == {
            "memories": [
                {"id": "abc123", "text": "uses rust", "kind": "fact", "owner": "agent:opencode"}
            ]
        }


class TestMemoryListTool:
    def test_returns_serialized_with_shared(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: True)
        monkeypatch.setattr(
            mcp_server, "list_memories", MagicMock(return_value=[_row("note", shared=True)])
        )
        result = mcp_server.memory_list(agent_id="opencode")
        assert result == {
            "memories": [{"id": "abc123", "text": "note", "kind": "fact", "shared": True}]
        }

    def test_disabled_returns_hint(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: False)
        assert mcp_server.memory_list() == {"error": MEMORY_DISABLED_HINT}


class TestMemoryForgetTool:
    def test_forgets(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: True)
        forget = MagicMock()
        monkeypatch.setattr(mcp_server, "forget", forget)
        assert mcp_server.memory_forget("d1") == {"ok": True, "id": "d1"}
        forget.assert_called_once_with("d1")

    def test_disabled_returns_hint(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "memory_enabled", lambda: False)
        assert mcp_server.memory_forget("d1") == {"error": MEMORY_DISABLED_HINT}
