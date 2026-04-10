"""Tests for the services container — focused on the clusterer factory."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from lilbee.clustering_embedding import EmbeddingClusterer
from lilbee.config import cfg
from lilbee.services import _build_clusterer


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestBuildClusterer:
    def test_default_returns_embedding_clusterer(self):
        cfg.wiki_clusterer = "embedding"
        store = MagicMock()
        result = _build_clusterer(cfg, store)
        assert isinstance(result, EmbeddingClusterer)

    def test_unknown_choice_falls_back_to_embedding(self):
        cfg.wiki_clusterer = "no-such-backend"
        store = MagicMock()
        result = _build_clusterer(cfg, store)
        assert isinstance(result, EmbeddingClusterer)

    def test_concepts_choice_returns_graph_clusterer_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from lilbee.concepts import ConceptGraphClusterer

        cfg.wiki_clusterer = "concepts"
        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: True)
        store = MagicMock()
        result = _build_clusterer(cfg, store)
        assert isinstance(result, ConceptGraphClusterer)

    def test_concepts_choice_falls_back_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        from lilbee.concepts import ConceptGraphClusterer

        cfg.wiki_clusterer = "concepts"
        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: False)
        store = MagicMock()
        with caplog.at_level(logging.WARNING, logger="lilbee.services"):
            result = _build_clusterer(cfg, store)
        assert isinstance(result, EmbeddingClusterer)
        assert any("falling back" in record.message.lower() for record in caplog.records)
