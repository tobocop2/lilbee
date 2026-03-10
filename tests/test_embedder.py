"""Tests for the Ollama embedding wrapper (mocked — no live server needed)."""

from unittest import mock

import ollama
import pytest

from lilbee.config import cfg


class TestTruncate:
    def test_short_text_unchanged(self):
        from lilbee.embedder import truncate

        text = "short text"
        assert truncate(text) == text

    def test_long_texttruncated(self):

        from lilbee.embedder import truncate

        text = "x" * (cfg.max_embed_chars + 500)
        result = truncate(text)
        assert len(result) == cfg.max_embed_chars

    def test_exact_limit_unchanged(self):

        from lilbee.embedder import truncate

        text = "a" * cfg.max_embed_chars
        assert truncate(text) == text


class TestEmbed:
    @mock.patch("ollama.embed")
    def test_returns_vector(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.1] * 768])
        from lilbee.embedder import embed

        vec = embed("test")
        assert vec == [0.1] * 768
        mock_ollama.assert_called_once()

    @mock.patch("ollama.embed")
    def test_passes_correct_model(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.0] * 768])
        from lilbee.embedder import embed

        embed("hello")
        call_kwargs = mock_ollama.call_args
        assert call_kwargs[1]["input"] == "hello"

    @mock.patch("ollama.embed")
    def testtruncates_long_input(self, mock_ollama):

        from lilbee.embedder import embed

        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.0] * 768])
        long_text = "a" * (cfg.max_embed_chars + 1000)
        embed(long_text)
        actual_input = mock_ollama.call_args[1]["input"]
        assert len(actual_input) == cfg.max_embed_chars


class TestEmbedBatch:
    @mock.patch("ollama.embed")
    def test_returns_multiple_vectors(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.1] * 768, [0.2] * 768])
        from lilbee.embedder import embed_batch

        result = embed_batch(["a", "b"])
        assert len(result) == 2
        mock_ollama.assert_called_once()

    def test_empty_input_returns_empty(self):
        from lilbee.embedder import embed_batch

        assert embed_batch([]) == []

    @mock.patch("ollama.embed")
    def test_passes_list_as_input(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.0] * 768, [0.0] * 768])
        from lilbee.embedder import embed_batch

        embed_batch(["hello", "world"])
        assert mock_ollama.call_args[1]["input"] == ["hello", "world"]

    @mock.patch("ollama.embed")
    def test_batches_large_input(self, mock_ollama):
        """Texts exceeding MAX_BATCH_CHARS split into multiple API calls."""

        from lilbee.embedder import MAX_BATCH_CHARS, embed_batch

        # Use chunks under cfg.max_embed_chars so they don't get truncated
        chunk_size = min(cfg.max_embed_chars, MAX_BATCH_CHARS // 2 + 1)
        # Need enough chunks so total chars > MAX_BATCH_CHARS
        n_to_fill = MAX_BATCH_CHARS // chunk_size + 1
        texts = ["x" * chunk_size for _ in range(n_to_fill + 1)]
        mock_ollama.side_effect = [
            mock.MagicMock(embeddings=[[0.1] * 768 for _ in range(n_to_fill)]),
            mock.MagicMock(embeddings=[[0.1] * 768]),
        ]
        result = embed_batch(texts)
        assert len(result) == n_to_fill + 1
        assert mock_ollama.call_count == 2

    @mock.patch("ollama.embed")
    def testtruncates_long_texts_in_batch(self, mock_ollama):

        from lilbee.embedder import embed_batch

        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.0] * 768, [0.0] * 768])
        texts = ["short", "x" * (cfg.max_embed_chars + 500)]
        embed_batch(texts)
        # Both fit in one batch after truncation (total < MAX_BATCH_CHARS)
        mock_ollama.assert_called_once()
        call_input = mock_ollama.call_args[1]["input"]
        assert call_input[0] == "short"
        assert len(call_input[1]) == cfg.max_embed_chars


class TestValidateVector:
    def test_valid_vector_passes(self):
        from lilbee.embedder import validate_vector

        validate_vector([0.1] * 768)  # Should not raise

    @mock.patch("ollama.embed")
    def test_embed_wrong_dim_raises(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.1, 0.2]])  # Wrong dim
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed("test")

    @mock.patch("ollama.embed")
    def test_embed_nan_raises(self, mock_ollama):
        import math

        mock_ollama.return_value = mock.MagicMock(embeddings=[[math.nan] + [0.1] * 767])
        from lilbee.embedder import embed

        with pytest.raises(ValueError, match="invalid value"):
            embed("test")

    @mock.patch("ollama.embed")
    def test_embed_batch_wrong_dim_raises(self, mock_ollama):
        mock_ollama.return_value = mock.MagicMock(embeddings=[[0.1, 0.2]])  # Wrong dim
        from lilbee.embedder import embed_batch

        with pytest.raises(ValueError, match="dimension mismatch"):
            embed_batch(["test"])

    @mock.patch("ollama.embed")
    def test_embed_inf_raises(self, mock_ollama):
        import math

        mock_ollama.return_value = mock.MagicMock(embeddings=[[math.inf] + [0.1] * 767])
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
        event = mock.MagicMock(status="done", total=100, completed=100)
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull", return_value=iter([event])) as mock_pull,
        ):
            from lilbee.embedder import validate_model

            validate_model()
            mock_pull.assert_called_once_with("nomic-embed-text", stream=True)

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

    def test_auto_pull_shows_ready_message(self, capsys):
        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        event = mock.MagicMock(status="downloading", total=1_000_000, completed=1_000_000)
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull", return_value=iter([event])),
        ):
            from lilbee.embedder import validate_model

            validate_model()
            stderr = capsys.readouterr().err
            assert "ready" in stderr

    def test_auto_pull_handles_events_without_total(self):
        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        event = mock.MagicMock(status="pulling manifest", total=0, completed=0)
        with (
            mock.patch("ollama.list", return_value=mock_response),
            mock.patch("ollama.pull", return_value=iter([event])),
        ):
            from lilbee.embedder import validate_model

            validate_model()  # Should not raise

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
            mock.MagicMock(embeddings=[[0.1] * 768]),
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
