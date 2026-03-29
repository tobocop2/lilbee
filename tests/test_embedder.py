"""Tests for the embedding wrapper (mocked -- no live server needed)."""

from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.config import cfg
from lilbee.embedder import MAX_BATCH_CHARS, Embedder


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
        text = "x" * (cfg.max_embed_chars + 500)
        result = embedder.truncate(text)
        assert len(result) == cfg.max_embed_chars

    def test_exact_limit_unchanged(self, embedder):
        text = "a" * cfg.max_embed_chars
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
        long_text = "a" * (cfg.max_embed_chars + 1000)
        embedder.embed(long_text)
        call_args = mock_provider.embed.call_args[0][0]
        assert len(call_args[0]) == cfg.max_embed_chars


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
        """Texts exceeding MAX_BATCH_CHARS split into multiple API calls."""
        chunk_size = min(cfg.max_embed_chars, MAX_BATCH_CHARS // 2 + 1)
        n_to_fill = MAX_BATCH_CHARS // chunk_size + 1
        texts = ["x" * chunk_size for _ in range(n_to_fill + 1)]
        mock_provider.embed.side_effect = [
            [[0.1] * 768 for _ in range(n_to_fill)],
            [[0.1] * 768],
        ]
        result = embedder.embed_batch(texts)
        assert len(result) == n_to_fill + 1
        assert mock_provider.embed.call_count == 2

    def test_truncates_long_texts_in_batch(self, embedder, mock_provider):
        mock_provider.embed.return_value = [[0.0] * 768, [0.0] * 768]
        texts = ["short", "x" * (cfg.max_embed_chars + 500)]
        embedder.embed_batch(texts)
        mock_provider.embed.assert_called_once()
        call_input = mock_provider.embed.call_args[0][0]
        assert call_input[0] == "short"
        assert len(call_input[1]) == cfg.max_embed_chars


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
    def test_validate_returns_true_when_model_available(self, embedder):
        with mock.patch("lilbee.providers.llama_cpp_provider._resolve_model_path"):
            assert embedder.validate_model() is True

    def test_validate_returns_false_when_model_missing(self, embedder):
        from lilbee.providers.base import ProviderError

        with mock.patch(
            "lilbee.providers.llama_cpp_provider._resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            assert embedder.validate_model() is False

    def test_embedding_available_true(self, embedder):
        with mock.patch("lilbee.providers.llama_cpp_provider._resolve_model_path"):
            assert embedder.embedding_available() is True

    def test_embedding_available_false(self, embedder):
        from lilbee.providers.base import ProviderError

        with mock.patch(
            "lilbee.providers.llama_cpp_provider._resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            assert embedder.embedding_available() is False
