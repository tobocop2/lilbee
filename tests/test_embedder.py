"""Tests for the Ollama embedding wrapper (mocked — no live server needed)."""

from unittest import mock

import ollama
import pytest


class TestTruncate:
    def test_short_text_unchanged(self):
        from lilbee.embedder import _truncate

        text = "short text"
        assert _truncate(text) == text

    def test_long_text_truncated(self):
        from lilbee.embedder import _MAX_EMBED_CHARS, _truncate

        text = "x" * (_MAX_EMBED_CHARS + 500)
        result = _truncate(text)
        assert len(result) == _MAX_EMBED_CHARS

    def test_exact_limit_unchanged(self):
        from lilbee.embedder import _MAX_EMBED_CHARS, _truncate

        text = "a" * _MAX_EMBED_CHARS
        assert _truncate(text) == text


class TestEmbed:
    @mock.patch("ollama.embed")
    def test_returns_vector(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.1] * 768]}
        from lilbee.embedder import embed

        vec = embed("test")
        assert vec == [0.1] * 768
        mock_ollama.assert_called_once()

    @mock.patch("ollama.embed")
    def test_passes_correct_model(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.0] * 768]}
        from lilbee.embedder import embed

        embed("hello")
        call_kwargs = mock_ollama.call_args
        assert call_kwargs[1]["input"] == "hello"

    @mock.patch("ollama.embed")
    def test_truncates_long_input(self, mock_ollama):
        from lilbee.embedder import _MAX_EMBED_CHARS, embed

        mock_ollama.return_value = {"embeddings": [[0.0] * 768]}
        long_text = "a" * (_MAX_EMBED_CHARS + 1000)
        embed(long_text)
        actual_input = mock_ollama.call_args[1]["input"]
        assert len(actual_input) == _MAX_EMBED_CHARS


class TestEmbedBatch:
    @mock.patch("ollama.embed")
    def test_returns_multiple_vectors(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.1] * 768, [0.2] * 768]}
        from lilbee.embedder import embed_batch

        result = embed_batch(["a", "b"])
        assert len(result) == 2
        mock_ollama.assert_called_once()

    def test_empty_input_returns_empty(self):
        from lilbee.embedder import embed_batch

        assert embed_batch([]) == []

    @mock.patch("ollama.embed")
    def test_passes_list_as_input(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.0] * 768, [0.0] * 768]}
        from lilbee.embedder import embed_batch

        embed_batch(["hello", "world"])
        assert mock_ollama.call_args[1]["input"] == ["hello", "world"]

    @mock.patch("ollama.embed")
    def test_batches_large_input(self, mock_ollama):
        """Texts exceeding _MAX_BATCH_CHARS split into multiple API calls."""
        from lilbee.embedder import _MAX_BATCH_CHARS, _MAX_EMBED_CHARS, embed_batch

        # Use chunks under _MAX_EMBED_CHARS so they don't get truncated
        chunk_size = min(_MAX_EMBED_CHARS, _MAX_BATCH_CHARS // 2 + 1)
        # Need enough chunks so total chars > _MAX_BATCH_CHARS
        n_to_fill = _MAX_BATCH_CHARS // chunk_size + 1
        texts = ["x" * chunk_size for _ in range(n_to_fill + 1)]
        mock_ollama.side_effect = [
            {"embeddings": [[0.1] * 768 for _ in range(n_to_fill)]},
            {"embeddings": [[0.1] * 768]},
        ]
        result = embed_batch(texts)
        assert len(result) == n_to_fill + 1
        assert mock_ollama.call_count == 2

    @mock.patch("ollama.embed")
    def test_truncates_long_texts_in_batch(self, mock_ollama):
        from lilbee.embedder import _MAX_EMBED_CHARS, embed_batch

        mock_ollama.return_value = {"embeddings": [[0.0] * 768, [0.0] * 768]}
        texts = ["short", "x" * (_MAX_EMBED_CHARS + 500)]
        embed_batch(texts)
        # Both fit in one batch after truncation (total < _MAX_BATCH_CHARS)
        mock_ollama.assert_called_once()
        call_input = mock_ollama.call_args[1]["input"]
        assert call_input[0] == "short"
        assert len(call_input[1]) == _MAX_EMBED_CHARS


class TestValidateVector:
    def test_valid_vector_passes(self):
        from lilbee.embedder import _validate_vector

        _validate_vector([0.1] * 768)  # Should not raise

    @mock.patch("ollama.embed")
    def test_embed_wrong_dim_raises(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.1, 0.2]]}  # Wrong dim
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed("test")

    @mock.patch("ollama.embed")
    def test_embed_nan_raises(self, mock_ollama):
        import math

        mock_ollama.return_value = {"embeddings": [[math.nan] + [0.1] * 767]}
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="invalid value"):
            embed("test")

    @mock.patch("ollama.embed")
    def test_embed_batch_wrong_dim_raises(self, mock_ollama):
        mock_ollama.return_value = {"embeddings": [[0.1, 0.2]]}  # Wrong dim
        from lilbee.embedder import embed_batch

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed_batch(["test"])

    @mock.patch("ollama.embed")
    def test_embed_inf_raises(self, mock_ollama):
        import math

        mock_ollama.return_value = {"embeddings": [[math.inf] + [0.1] * 767]}
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="invalid value"):
            embed("test")


class TestValidateModel:
    def test_model_found(self):
        mock_model = mock.MagicMock()
        mock_model.model = "nomic-embed-text:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with mock.patch("ollama.list", return_value=mock_response):
            from lilbee.embedder import validate_model

            validate_model()  # Should not raise

    def test_model_found_by_base_name(self):
        mock_model = mock.MagicMock()
        mock_model.model = "nomic-embed-text:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with mock.patch("ollama.list", return_value=mock_response):
            from lilbee.embedder import validate_model

            validate_model()  # "nomic-embed-text" matches base of "nomic-embed-text:latest"

    def test_auto_pull_when_model_missing(self):
        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull") as mock_pull,
        ):
            from lilbee.embedder import validate_model

            validate_model()
            mock_pull.assert_called_once_with("nomic-embed-text")

    def test_auto_pull_failure_propagates(self):
        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull", side_effect=ollama.ResponseError("model not found")),
        ):
            from lilbee.embedder import validate_model

            with pytest.raises(ollama.ResponseError):
                validate_model()

    def test_auto_pull_logs_info(self, caplog):
        import logging

        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull"),
            caplog.at_level(logging.INFO, logger="lilbee.embedder"),
        ):
            from lilbee.embedder import validate_model

            validate_model()
            assert "Pulling embedding model" in caplog.text

    def test_connection_error(self):
        with mock.patch("ollama.list", side_effect=ConnectionError("refused")):
            from lilbee.embedder import validate_model

            with pytest.raises(RuntimeError, match="Cannot connect"):
                validate_model()


class TestRetry:
    @mock.patch("time.sleep")  # Don't actually sleep
    @mock.patch("ollama.embed")
    def test_retry_on_connection_error(self, mock_ollama, mock_sleep):
        mock_ollama.side_effect = [
            ConnectionError("refused"),
            {"embeddings": [[0.1] * 768]},
        ]
        from lilbee.embedder import embed

        vec = embed("test")
        assert len(vec) == 768
        assert mock_ollama.call_count == 2

    @mock.patch("time.sleep")
    @mock.patch("ollama.embed")
    def test_retry_exhaustion_raises(self, mock_ollama, mock_sleep):
        mock_ollama.side_effect = ConnectionError("refused")
        from lilbee.embedder import embed

        with pytest.raises(ConnectionError):
            embed("test")
        assert mock_ollama.call_count == 3

    @mock.patch("ollama.embed")
    def test_no_retry_on_value_error(self, mock_ollama):
        mock_ollama.side_effect = ValueError("bad input")
        from lilbee.embedder import embed

        with pytest.raises(ValueError):
            embed("test")
        assert mock_ollama.call_count == 1
