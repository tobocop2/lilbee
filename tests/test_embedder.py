"""Tests for the embedding wrapper (mocked — no live server needed)."""

from unittest import mock

import pytest

from lilbee.config import cfg


class TestTruncate:
    def test_short_text_unchanged(self):
        from lilbee.embedder import truncate

        text = "short text"
        assert truncate(text) == text

    def test_long_text_truncated(self):
        from lilbee.embedder import truncate

        text = "x" * (cfg.max_embed_chars + 500)
        result = truncate(text)
        assert len(result) == cfg.max_embed_chars

    def test_exact_limit_unchanged(self):
        from lilbee.embedder import truncate

        text = "a" * cfg.max_embed_chars
        assert truncate(text) == text


class TestEmbed:
    @mock.patch("lilbee.embedder.get_provider")
    def test_returns_vector(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.1] * 768]
        from lilbee.embedder import embed

        vec = embed("test")
        assert vec == [0.1] * 768

    @mock.patch("lilbee.embedder.get_provider")
    def test_passes_truncated_text(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.0] * 768]
        from lilbee.embedder import embed

        embed("hello")
        mock_get_provider.return_value.embed.assert_called_once_with(["hello"])

    @mock.patch("lilbee.embedder.get_provider")
    def test_truncates_long_input(self, mock_get_provider):
        from lilbee.embedder import embed

        mock_get_provider.return_value.embed.return_value = [[0.0] * 768]
        long_text = "a" * (cfg.max_embed_chars + 1000)
        embed(long_text)
        call_args = mock_get_provider.return_value.embed.call_args[0][0]
        assert len(call_args[0]) == cfg.max_embed_chars


class TestEmbedBatch:
    @mock.patch("lilbee.embedder.get_provider")
    def test_returns_multiple_vectors(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.1] * 768, [0.2] * 768]
        from lilbee.embedder import embed_batch

        result = embed_batch(["a", "b"])
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        from lilbee.embedder import embed_batch

        assert embed_batch([]) == []

    @mock.patch("lilbee.embedder.get_provider")
    def test_passes_list_as_input(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.0] * 768, [0.0] * 768]
        from lilbee.embedder import embed_batch

        embed_batch(["hello", "world"])
        mock_get_provider.return_value.embed.assert_called_once_with(["hello", "world"])

    @mock.patch("lilbee.embedder.get_provider")
    def test_batches_large_input(self, mock_get_provider):
        """Texts exceeding MAX_BATCH_CHARS split into multiple API calls."""
        from lilbee.embedder import MAX_BATCH_CHARS, embed_batch

        chunk_size = min(cfg.max_embed_chars, MAX_BATCH_CHARS // 2 + 1)
        n_to_fill = MAX_BATCH_CHARS // chunk_size + 1
        texts = ["x" * chunk_size for _ in range(n_to_fill + 1)]
        mock_get_provider.return_value.embed.side_effect = [
            [[0.1] * 768 for _ in range(n_to_fill)],
            [[0.1] * 768],
        ]
        result = embed_batch(texts)
        assert len(result) == n_to_fill + 1
        assert mock_get_provider.return_value.embed.call_count == 2

    @mock.patch("lilbee.embedder.get_provider")
    def test_truncates_long_texts_in_batch(self, mock_get_provider):
        from lilbee.embedder import embed_batch

        mock_get_provider.return_value.embed.return_value = [[0.0] * 768, [0.0] * 768]
        texts = ["short", "x" * (cfg.max_embed_chars + 500)]
        embed_batch(texts)
        mock_get_provider.return_value.embed.assert_called_once()
        call_input = mock_get_provider.return_value.embed.call_args[0][0]
        assert call_input[0] == "short"
        assert len(call_input[1]) == cfg.max_embed_chars


class TestValidateVector:
    def test_valid_vector_passes(self):
        from lilbee.embedder import validate_vector

        validate_vector([0.1] * 768)

    @mock.patch("lilbee.embedder.get_provider")
    def test_embed_wrong_dim_raises(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.1, 0.2]]
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed("test")

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    @mock.patch("lilbee.embedder.get_provider")
    def test_embed_invalid_value_raises(self, mock_get_provider, bad_value):
        mock_get_provider.return_value.embed.return_value = [[bad_value] + [0.1] * 767]
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="invalid value"):
            embed("test")

    @mock.patch("lilbee.embedder.get_provider")
    def test_embed_batch_wrong_dim_raises(self, mock_get_provider):
        mock_get_provider.return_value.embed.return_value = [[0.1, 0.2]]
        from lilbee.embedder import embed_batch

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed_batch(["test"])


class TestValidateModel:
    def test_model_found(self):
        with mock.patch("lilbee.model_manager.get_model_manager") as mock_get_mm:
            mock_get_mm.return_value.is_installed.return_value = True
            from lilbee.embedder import validate_model

            validate_model()

    def test_auto_pull_when_model_missing(self):
        with (
            mock.patch("lilbee.model_manager.get_model_manager") as mock_get_mm,
            mock.patch("lilbee.embedder.get_provider") as mock_get_prov,
        ):
            mock_get_mm.return_value.is_installed.return_value = False
            from lilbee.embedder import validate_model

            validate_model()
            mock_get_prov.return_value.pull_model.assert_called_once_with(
                cfg.embedding_model, on_progress=mock.ANY
            )

    def test_auto_pull_failure_propagates(self):
        with (
            mock.patch("lilbee.model_manager.get_model_manager") as mock_get_mm,
            mock.patch("lilbee.embedder.get_provider") as mock_get_prov,
        ):
            mock_get_mm.return_value.is_installed.return_value = False
            mock_get_prov.return_value.pull_model.side_effect = RuntimeError("model not found")
            from lilbee.embedder import validate_model

            with pytest.raises(RuntimeError):
                validate_model()

    def test_connection_error(self):
        with mock.patch("lilbee.model_manager.get_model_manager") as mock_get_mm:
            mock_get_mm.return_value.is_installed.side_effect = ConnectionError("refused")
            from lilbee.embedder import validate_model

            with pytest.raises(RuntimeError, match="Cannot connect"):
                validate_model()
