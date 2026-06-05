"""Tests for memory recall: the block formatter and searcher system-prompt injection."""

from unittest.mock import MagicMock

from lilbee.core.config import cfg
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow, MemorySource
from lilbee.retrieval.query import Searcher
from lilbee.retrieval.query.history_window import estimate_text_tokens
from lilbee.retrieval.query.memory import (
    MEMORY_BLOCK_FOOTER,
    MEMORY_BLOCK_HEADER,
    format_memory_block,
)


def _mem(text: str, kind: MemoryKind = MemoryKind.FACT) -> MemoryRow:
    return MemoryRow(
        id="i",
        owner=LOCAL_OWNER,
        shared=False,
        kind=kind,
        source=MemorySource.MANUAL,
        text=text,
        vector=[0.0],
        created_at="t",
        updated_at="t",
    )


def _searcher(store: MagicMock, embedder: MagicMock) -> Searcher:
    return Searcher(cfg, MagicMock(), store, embedder, MagicMock(), MagicMock())


class TestEstimateTextTokens:
    def test_floor_is_one(self):
        assert estimate_text_tokens("") == 1
        assert estimate_text_tokens("abc") == 1

    def test_char_over_four(self):
        assert estimate_text_tokens("a" * 40) == 10


class TestFormatMemoryBlock:
    def test_empty_when_nothing(self):
        assert format_memory_block([], [], 512) == ""

    def test_preferences_before_facts_with_framing(self):
        block = format_memory_block(
            [_mem("be terse", MemoryKind.PREFERENCE)], [_mem("uses rust")], 512
        )
        assert MEMORY_BLOCK_HEADER in block
        assert MEMORY_BLOCK_FOOTER in block
        assert block.index("be terse") < block.index("uses rust")

    def test_budget_drops_overflow_facts(self):
        prefs = [_mem("p" * 80, MemoryKind.PREFERENCE)]
        facts = [_mem("f" * 80)]
        # Budget fits exactly the header + the preference line, so the fact overflows.
        budget = estimate_text_tokens(MEMORY_BLOCK_HEADER) + estimate_text_tokens("- " + "p" * 80)
        block = format_memory_block(prefs, facts, budget)
        assert "p" * 80 in block
        assert "f" * 80 not in block

    def test_tiny_budget_returns_empty(self):
        assert format_memory_block([_mem("anything")], [], 1) == ""


class TestSearcherMemoryBlock:
    def test_disabled_returns_empty_and_no_store_calls(self):
        cfg.memory_enabled = False
        store, embedder = MagicMock(), MagicMock()
        assert _searcher(store, embedder)._memory_block("q") == ""
        store.get_memories.assert_not_called()

    def test_preferences_injected_without_embedding(self):
        cfg.memory_enabled = True
        store = MagicMock()
        store.get_memories.return_value = [_mem("be terse", MemoryKind.PREFERENCE)]
        embedder = MagicMock()
        embedder.embedding_available.return_value = False
        block = _searcher(store, embedder)._memory_block("q")
        assert "be terse" in block
        embedder.embed.assert_not_called()
        store.search_memories.assert_not_called()

    def test_facts_recalled_when_embedding_available(self):
        cfg.memory_enabled = True
        cfg.memory_top_k = 5
        store = MagicMock()
        store.get_memories.return_value = []
        store.search_memories.return_value = [_mem("uses rust")]
        embedder = MagicMock()
        embedder.embedding_available.return_value = True
        embedder.embed.return_value = [0.1, 0.2]
        block = _searcher(store, embedder)._memory_block("q")
        assert "uses rust" in block
        embedder.embed.assert_called_once_with("q")

    def test_top_k_zero_skips_fact_recall(self):
        cfg.memory_enabled = True
        cfg.memory_top_k = 0
        store = MagicMock()
        store.get_memories.return_value = []
        embedder = MagicMock()
        embedder.embedding_available.return_value = True
        _searcher(store, embedder)._memory_block("q")
        store.search_memories.assert_not_called()
        embedder.embed.assert_not_called()


class TestSystemWithMemory:
    def test_appends_block_when_present(self):
        cfg.memory_enabled = True
        store = MagicMock()
        store.get_memories.return_value = [_mem("be terse", MemoryKind.PREFERENCE)]
        embedder = MagicMock()
        embedder.embedding_available.return_value = False
        result = _searcher(store, embedder)._system_with_memory("BASE", "q")
        assert result.startswith("BASE\n\n")
        assert "be terse" in result

    def test_returns_base_when_empty(self):
        cfg.memory_enabled = False
        result = _searcher(MagicMock(), MagicMock())._system_with_memory("BASE", "q")
        assert result == "BASE"
