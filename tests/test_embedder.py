"""Tests for the embedding wrapper (mocked -- no live server needed)."""

from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.chunk import CHARS_PER_TOKEN
from lilbee.retrieval.embedder import EMBED_BATCH_TARGET_SEQUENCES, Embedder


@pytest.fixture()
def mock_provider():
    return MagicMock()


@pytest.fixture()
def embedder(mock_provider):
    return Embedder(cfg, mock_provider)


class TestTruncate:
    def test_short_text_unchanged(self, embedder):
        text = "short text"
        assert embedder.truncate(text) == text

    def test_long_text_truncated(self, embedder):
        text = "x" * (embedder.embed_char_budget + 500)
        result = embedder.truncate(text)
        assert len(result) == embedder.embed_char_budget

    def test_exact_limit_unchanged(self, embedder):
        text = "a" * embedder.embed_char_budget
        assert embedder.truncate(text) == text


class TestEmbed:
    def test_returns_vector(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.1] * 768]
        vec = embedder.embed("test")
        assert vec == [0.1] * 768

    def test_passes_truncated_text(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.0] * 768]
        embedder.embed("hello")
        mock_provider.embed.assert_called_once_with(["hello"])

    def test_truncates_long_input(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.0] * 768]
        long_text = "a" * (embedder.embed_char_budget + 1000)
        embedder.embed(long_text)
        call_args = mock_provider.embed.call_args[0][0]
        assert len(call_args[0]) == embedder.embed_char_budget


class TestEmbedBatch:
    def test_returns_multiple_vectors(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.1] * 768, [0.2] * 768]
        result = embedder.embed_batch(["a", "b"])
        assert len(result) == 2

    def test_empty_input_returns_empty(self, embedder):
        assert embedder.embed_batch([]) == []

    def test_passes_list_as_input(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.0] * 768, [0.0] * 768]
        embedder.embed_batch(["hello", "world"])
        mock_provider.embed.assert_called_once_with(["hello", "world"])

    def test_batches_large_input(self, embedder, mock_provider):
        """Texts exceeding the batch char budget split into multiple API calls."""
        budget = embedder.batch_char_budget
        chunk_size = min(embedder.embed_char_budget, budget // 2 + 1)
        n_to_fill = budget // chunk_size + 1
        texts = ["x" * chunk_size for _ in range(n_to_fill + 1)]
        mock_provider.embed.side_effect = [
            [[0.1] * 768 for _ in range(n_to_fill)],
            [[0.1] * 768],
        ]
        result = embedder.embed_batch(texts)
        assert len(result) == n_to_fill + 1
        assert mock_provider.embed.call_count == 2

    def test_batch_char_budget_feeds_full_engine_batches(self, embedder):
        """The app-layer cap allows a full packed batch of max-size chunks."""
        assert (
            embedder.batch_char_budget
            == EMBED_BATCH_TARGET_SEQUENCES * cfg.chunk_size * CHARS_PER_TOKEN
        )

    def test_many_default_chunks_fit_one_request(self, embedder, mock_provider):
        """A typical bulk-ingest batch is no longer split into 3-8 chunk requests."""
        chunk_chars = cfg.chunk_size * CHARS_PER_TOKEN
        texts = ["x" * (chunk_chars // 2) for _ in range(EMBED_BATCH_TARGET_SEQUENCES)]
        mock_provider.embed.return_value = [[0.1] * 768 for _ in texts]
        embedder.embed_batch(texts)
        assert mock_provider.embed.call_count == 1

    def test_truncates_long_texts_in_batch(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.0] * 768, [0.0] * 768]
        texts = ["short", "x" * (embedder.embed_char_budget + 500)]
        embedder.embed_batch(texts)
        mock_provider.embed.assert_called_once()
        call_input = mock_provider.embed.call_args[0][0]
        assert call_input[0] == "short"
        assert len(call_input[1]) == embedder.embed_char_budget


class TestValidateVector:
    def test_valid_vector_passes(self, embedder):
        embedder.validate_vector([0.1] * 768)

    def test_embed_wrong_dim_raises(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.1, 0.2]]
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedder.embed("test")

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_embed_invalid_value_raises(self, embedder, mock_provider, bad_value):
        mock_provider.embed.return_value = [[bad_value] + [0.1] * 767]
        with pytest.raises(ValueError, match="invalid value"):
            embedder.embed("test")

    def test_embed_batch_wrong_dim_raises(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.1, 0.2]]
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedder.embed_batch(["test"])


class TestValidateModel:
    def test_validate_returns_true_when_model_available(self, embedder, mock_provider):
        mock_provider.list_models.return_value = [cfg.embedding_model]
        assert embedder.validate_model() is True

    def test_validate_returns_false_when_model_missing(self, embedder, mock_provider):
        mock_provider.list_models.return_value = []
        assert embedder.validate_model() is False

    def test_validate_returns_false_on_provider_error(self, embedder, mock_provider):
        mock_provider.list_models.side_effect = RuntimeError("no connection")
        assert embedder.validate_model() is False

    def test_embedding_available_true(self, embedder, mock_provider):
        mock_provider.list_models.return_value = [cfg.embedding_model]
        assert embedder.embedding_available() is True

    def test_embedding_available_false(self, embedder, mock_provider):
        mock_provider.list_models.return_value = []
        assert embedder.embedding_available() is False

    def test_embedding_available_empty_string(self, mock_provider):
        """embedding_available returns False when model is empty string."""
        from lilbee.core.config import cfg

        old = cfg.embedding_model
        # Bypass pydantic validation to simulate edge case
        object.__setattr__(cfg, "embedding_model", "")
        try:
            embedder = Embedder(cfg, mock_provider)
            assert embedder.embedding_available() is False
        finally:
            cfg.embedding_model = old

    def test_embedding_available_true_for_prefixed_model_matching_bare_list(self, mock_provider):
        """ollama/ ref matches against bare tags returned by list_models.

        provider.list_models returns raw /api/tags entries without the
        ollama/ prefix, so the availability check must compare on the
        stripped name.
        """
        from lilbee.core.config import cfg

        old = cfg.embedding_model
        cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
        try:
            mock_provider.list_models.return_value = ["nomic-embed-text:v1.5"]
            embedder = Embedder(cfg, mock_provider)
            assert embedder.embedding_available() is True
        finally:
            cfg.embedding_model = old

    def test_embedding_available_false_for_prefixed_model_skips_native_probe(self, mock_provider):
        """ollama/ ref not in list_models returns False without resolve_model_path."""
        from lilbee.core.config import cfg

        old = cfg.embedding_model
        cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
        try:
            mock_provider.list_models.return_value = []
            embedder = Embedder(cfg, mock_provider)
            with mock.patch("lilbee.providers.engine_params.resolve_model_path") as resolve:
                assert embedder.embedding_available() is False
                resolve.assert_not_called()
        finally:
            cfg.embedding_model = old


class TestAsymmetricEmbed:
    """bb-7z8: query vs document embedding gets the configured embedder's instruction."""

    def test_symmetric_model_applies_no_prefix(self, embedder, mock_provider, monkeypatch):
        monkeypatch.setattr(cfg, "embedding_model", "gpustack/bge-m3-GGUF/b.gguf")
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim]
        embedder.embed_query("hello")
        assert mock_provider.embed.call_args[0][0] == ["hello"]

    def test_instruct_model_prefixes_query_not_document(self, embedder, mock_provider, monkeypatch):
        monkeypatch.setattr(cfg, "embedding_model", "Qwen/Qwen3-Embedding-8B-GGUF/q.gguf")
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim]
        embedder.embed_query("astar grid")
        sent = mock_provider.embed.call_args[0][0][0]
        assert sent.startswith("Instruct:") and sent.endswith("astar grid")

        mock_provider.reset_mock()
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim]
        embedder.embed("a document")  # qwen3 doc_prefix is empty
        assert mock_provider.embed.call_args[0][0] == ["a document"]

    def test_e5_model_prefixes_both_sides(self, embedder, mock_provider, monkeypatch):
        monkeypatch.setattr(cfg, "embedding_model", "intfloat/e5-large-v2-GGUF/e.gguf")
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim]
        embedder.embed_query("q")
        assert mock_provider.embed.call_args[0][0] == ["query: q"]

        mock_provider.reset_mock()
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim]
        embedder.embed_batch(["d"])
        assert mock_provider.embed.call_args[0][0] == ["passage: d"]

    def test_embed_query_batch_prefixes_each(self, embedder, mock_provider, monkeypatch):
        monkeypatch.setattr(cfg, "embedding_model", "intfloat/e5-large-v2-GGUF/e.gguf")
        mock_provider.embed.return_value = [[0.0] * cfg.embedding_dim] * 2
        embedder.embed_query_batch(["a", "b"])
        assert mock_provider.embed.call_args[0][0] == ["query: a", "query: b"]
