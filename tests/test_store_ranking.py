"""Tests for the store's MMR ranking helper's config scoping."""

from __future__ import annotations

from lilbee.core.config import cfg, config_scope
from lilbee.data.store import SearchChunk, mmr_rerank


def _chunk(source: str, vector: list[float]) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=source,
        chunk_index=0,
        distance=0.5,
        vector=vector,
    )


class TestMmrRerankConfigScoping:
    def test_default_lambda_honors_the_scoped_config(self):
        """Omitting mmr_lambda must read the ACTIVE config, not the process
        global: under the library API a config_scope binding is the caller's
        config, and every sibling in the data path uses active_config()."""
        # Two near-duplicates and one distinct doc, query aligned with the pair.
        results = [
            _chunk("dup_a.md", [1.0, 0.0]),
            _chunk("dup_b.md", [0.99, 0.01]),
            _chunk("other.md", [0.0, 1.0]),
        ]
        query = [1.0, 0.0]

        scoped = cfg.model_copy()
        # Pure relevance: both near-duplicates win the top two slots.
        scoped.mmr_lambda = 1.0
        with config_scope(scoped):
            relevance_only = mmr_rerank(query, results, top_k=2)

        # Maximum diversity: the distinct doc displaces the near-duplicate.
        scoped_diverse = cfg.model_copy()
        scoped_diverse.mmr_lambda = 0.0
        with config_scope(scoped_diverse):
            diversity_only = mmr_rerank(query, results, top_k=2)

        assert [r.source for r in relevance_only] == ["dup_a.md", "dup_b.md"]
        assert "other.md" in [r.source for r in diversity_only]
