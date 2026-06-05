"""Tests for the app-layer memory service and owner predicate builders."""

from unittest.mock import MagicMock

import pytest

from lilbee.app import memory as app_memory
from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.data.store import (
    LOCAL_OWNER,
    MemoryKind,
    MemorySource,
    agent_owner,
    agent_recall_predicate,
    local_owner_predicate,
)
from tests.conftest import make_mock_services


@pytest.fixture()
def svc():
    store = MagicMock()
    store.add_memory.return_value = "stored-id"
    store.update_memory.return_value = True
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2]
    services = make_mock_services(
        store=store,
        embedder=embedder,
        worker_pool=MagicMock(),
        pool_runtime=MagicMock(),
        pool_health_ticker=MagicMock(),
    )
    set_services(services)
    yield services
    set_services(None)


class TestPredicates:
    def test_local_predicate(self):
        assert local_owner_predicate() == f"owner = '{LOCAL_OWNER}'"

    def test_agent_predicate_unions_shared_local(self):
        pred = agent_recall_predicate(agent_owner("opencode"))
        assert "owner = 'agent:opencode'" in pred
        assert f"shared = true AND owner = '{LOCAL_OWNER}'" in pred

    def test_agent_predicate_escapes_quotes(self):
        pred = agent_recall_predicate("agent:o'hare")
        assert "agent:o''hare" in pred


class TestMemoryEnabled:
    def test_reflects_cfg(self):
        cfg.memory_enabled = False
        assert app_memory.memory_enabled() is False
        cfg.memory_enabled = True
        assert app_memory.memory_enabled() is True


class TestRemember:
    def test_embeds_and_stores_with_metadata(self, svc):
        result = app_memory.remember(
            "prefers rust",
            kind=MemoryKind.PREFERENCE,
            source=MemorySource.MANUAL,
            shared=True,
        )
        assert result == "stored-id"
        svc.embedder.embed.assert_called_once_with("prefers rust")
        record = svc.store.add_memory.call_args.args[0]
        assert record.text == "prefers rust"
        assert record.kind is MemoryKind.PREFERENCE
        assert record.owner == LOCAL_OWNER
        assert record.shared is True
        assert record.vector == [0.1, 0.2]
        assert len(record.id) == 32  # uuid4 hex

    def test_agent_owner_extracted_memory_defaults(self, svc):
        app_memory.remember(
            "uses lancedb",
            owner=agent_owner("opencode"),
            source=MemorySource.AGENT,
        )
        record = svc.store.add_memory.call_args.args[0]
        assert record.owner == "agent:opencode"
        assert record.source is MemorySource.AGENT


class TestRecall:
    def test_local_uses_local_predicate(self, svc):
        cfg.memory_top_k = 7
        cfg.memory_max_distance = 0.4
        app_memory.recall("where is auth")
        svc.embedder.embed.assert_called_once_with("where is auth")
        kwargs = svc.store.search_memories.call_args.kwargs
        assert kwargs["owner_predicate"] == local_owner_predicate()
        assert kwargs["top_k"] == 7
        assert kwargs["max_distance"] == 0.4

    def test_agent_uses_recall_predicate(self, svc):
        app_memory.recall("q", owner=agent_owner("x"))
        kwargs = svc.store.search_memories.call_args.kwargs
        assert kwargs["owner_predicate"] == agent_recall_predicate(agent_owner("x"))

    def test_explicit_top_k_overrides(self, svc):
        app_memory.recall("q", top_k=2)
        assert svc.store.search_memories.call_args.kwargs["top_k"] == 2


class TestListForgetFlags:
    def test_list_local(self, svc):
        app_memory.list_memories()
        assert svc.store.get_memories.call_args.kwargs["owner_predicate"] == local_owner_predicate()

    def test_list_agent_owns_only(self, svc):
        app_memory.list_memories(agent_owner("x"))
        assert svc.store.get_memories.call_args.kwargs["owner_predicate"] == "owner = 'agent:x'"

    def test_forget(self, svc):
        app_memory.forget("d1")
        svc.store.delete_memory.assert_called_once_with("d1")

    def test_set_shared(self, svc):
        assert app_memory.set_memory_shared("u1", shared=True) is True
        svc.store.update_memory.assert_called_once_with("u1", shared=True)


class TestAutoExtract:
    @pytest.fixture(autouse=True)
    def _restore_cfg(self):
        snapshot = (cfg.memory_enabled, cfg.memory_auto_extract)
        yield
        cfg.memory_enabled, cfg.memory_auto_extract = snapshot

    def test_enabled_requires_both_gates(self):
        cfg.memory_enabled = True
        cfg.memory_auto_extract = False
        assert app_memory.auto_extract_enabled() is False
        cfg.memory_auto_extract = True
        assert app_memory.auto_extract_enabled() is True
        cfg.memory_enabled = False
        assert app_memory.auto_extract_enabled() is False

    def test_disabled_does_not_call_model(self, svc):
        cfg.memory_enabled = True
        cfg.memory_auto_extract = False
        assert app_memory.auto_extract("q", "a") == []
        svc.provider.chat.assert_not_called()

    def test_stores_extracted(self, svc):
        cfg.memory_enabled = True
        cfg.memory_auto_extract = True
        svc.provider.chat.return_value = '[{"text": "the user prefers rust", "kind": "fact"}]'
        stored = app_memory.auto_extract("I love rust", "Rust is great.")
        assert [m.text for m in stored] == ["the user prefers rust"]
        assert stored[0].id == "stored-id"
        assert stored[0].kind is MemoryKind.FACT
        record = svc.store.add_memory.call_args.args[0]
        assert record.source is MemorySource.EXTRACTED

    def test_nothing_extracted_stores_nothing(self, svc):
        cfg.memory_enabled = True
        cfg.memory_auto_extract = True
        svc.provider.chat.return_value = "[]"
        assert app_memory.auto_extract("hi", "hello") == []
        svc.store.add_memory.assert_not_called()
