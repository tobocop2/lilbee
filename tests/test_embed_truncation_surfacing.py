"""Truncation is observable, not silent, and the char limit never clips a full-budget chunk.

Token-dense input (minified JSON, dense code, CJK) underestimates under the
4-chars/token heuristic, so a finished chunk can land over the embedder's char
limit. These tests pin two correctness invariants:

1. The embedder's char limit is never below the chunker's max chunk size, so a
   full-budget chunk is embedded whole rather than silently losing its tail.
2. When the embedder does truncate, it reports a count that the ingest pipeline
   surfaces in ``SyncResult`` instead of swallowing it in a log line.
"""

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.extract.chunk import CHARS_PER_TOKEN
from lilbee.data.types import SyncResult
from lilbee.retrieval.embedder import Embedder


@pytest.fixture()
def mock_provider():
    provider = MagicMock()
    provider.embed.side_effect = lambda texts: [[0.0] * cfg.embedding_dim for _ in texts]
    return provider


@pytest.fixture()
def embedder(mock_provider):
    return Embedder(cfg, mock_provider)


class TestCharBudgetCoversChunkBudget:
    """A finished full-budget chunk must not be clipped at the embedder char limit."""

    def test_full_budget_chunk_not_truncated(self, embedder):
        chunk_budget = cfg.chunk_size * CHARS_PER_TOKEN
        text = "a" * chunk_budget
        assert embedder.truncate(text) == text

    def test_effective_limit_at_least_chunk_budget(self, embedder):
        assert embedder.embed_char_budget >= cfg.chunk_size * CHARS_PER_TOKEN


class TestTruncationIsSurfaced:
    """Over-budget input is counted and the count reaches the caller.

    The count the pipeline surfaces is the delta of the lock-protected
    cumulative counter across the run, which is the only reading that stays
    correct while several ingest threads embed through the shared Embedder.
    """

    def test_embed_batch_counts_truncations(self, embedder):
        over = "x" * (embedder.embed_char_budget + 500)
        before = embedder.truncated_total
        embedder.embed_batch(["short", over, over])
        assert embedder.truncated_total - before == 2

    def test_no_truncation_counts_nothing(self, embedder):
        before = embedder.truncated_total
        embedder.embed_batch(["short", "also short"])
        assert embedder.truncated_total - before == 0


class TestSyncResultCarriesTruncation:
    def test_sync_result_has_truncated_field(self):
        result = SyncResult(truncated=3)
        assert result.truncated == 3
        assert "Truncated: 3" in str(result)
