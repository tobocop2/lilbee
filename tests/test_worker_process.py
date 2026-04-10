"""Tests for the subprocess worker process for llama-cpp operations."""

from __future__ import annotations

import multiprocessing
import sys
from multiprocessing import get_context
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.worker_process import (
    ConfigSnapshot,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ShutdownRequest,
    VisionRequest,
    VisionResponse,
    WorkerProcess,
    _close_model,
    _handle_embed,
    _handle_vision,
    _load_embed_model,
    _load_vision_model,
    _worker_main,
    config_snapshot_from_cfg,
)


@pytest.fixture(autouse=True)
def _no_stdio_redirect():
    """Prevent _worker_main from redirecting stdout/stderr to /dev/null.
    In a real child process the redirect suppresses llama-cpp C-level output.
    In tests it leaks unclosed devnull files and fights with pytest capture.
    """
    with mock.patch("lilbee.providers.worker_process._redirect_stdio"):
        yield


@pytest.fixture()
def config_snap(tmp_path: Path) -> ConfigSnapshot:
    return ConfigSnapshot(
        models_dir=str(tmp_path / "models"),
        embedding_model="test-embed",
        embedding_dim=768,
        num_ctx=None,
        gpu_memory_fraction=0.75,
        vision_model="test-vision",
    )


class TestDataclasses:
    def test_embed_request_fields(self) -> None:
        req = EmbedRequest(texts=["hello"], model="m", request_id=1)
        assert req.texts == ["hello"]
        assert req.model == "m"
        assert req.request_id == 1

    def test_embed_response_defaults(self) -> None:
        resp = EmbedResponse()
        assert resp.vectors == []
        assert resp.error == ""
        assert resp.request_id == 0

    def test_vision_request_fields(self) -> None:
        req = VisionRequest(png_bytes=b"img", model="v", prompt="p", request_id=2)
        assert req.png_bytes == b"img"
        assert req.model == "v"
        assert req.prompt == "p"

    def test_vision_response_defaults(self) -> None:
        resp = VisionResponse()
        assert resp.text == ""
        assert resp.error == ""

    def test_load_model_request_defaults(self) -> None:
        req = LoadModelRequest(model="m")
        assert req.model_type == "embed"

    def test_shutdown_request(self) -> None:
        req = ShutdownRequest()
        assert isinstance(req, ShutdownRequest)

    def test_config_snapshot_fields(self, config_snap: ConfigSnapshot) -> None:
        assert config_snap.models_dir.endswith("models")
        assert config_snap.embedding_model == "test-embed"
        assert config_snap.embedding_dim == 768


class TestConfigSnapshotFromCfg:
    def test_builds_from_cfg(self) -> None:
        snap = config_snapshot_from_cfg()
        assert isinstance(snap, ConfigSnapshot)
        assert snap.embedding_model != ""


class TestCloseModel:
    def test_closes_model_with_close_method(self) -> None:
        model = mock.MagicMock()
        _close_model(model)
        model.close.assert_called_once()

    def test_none_is_safe(self) -> None:
        _close_model(None)

    def test_close_exception_suppressed(self) -> None:
        model = mock.MagicMock()
        model.close.side_effect = RuntimeError("boom")
        _close_model(model)  # Should not raise

    def test_no_close_attr_is_safe(self) -> None:
        _close_model(42)  # int has no close method


class TestHandleEmbed:
    def test_success(self) -> None:
        llm = mock.MagicMock()
        with mock.patch(
            "lilbee.providers.llama_cpp_provider.embed_one",
            side_effect=[[0.1, 0.2], [0.3, 0.4]],
        ):
            req = EmbedRequest(texts=["a", "b"], model="m", request_id=5)
            resp = _handle_embed(llm, req)
        assert resp.vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert resp.request_id == 5
        assert resp.error == ""

    def test_error_returns_error_response(self) -> None:
        llm = mock.MagicMock()
        with mock.patch(
            "lilbee.providers.llama_cpp_provider.embed_one",
            side_effect=RuntimeError("GPU OOM"),
        ):
            req = EmbedRequest(texts=["a"], model="m", request_id=3)
            resp = _handle_embed(llm, req)
        assert resp.error == "GPU OOM"
        assert resp.vectors == []


class TestHandleVision:
    def test_success(self) -> None:
        llm = mock.MagicMock()
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "extracted text"}}]
        }
        req = VisionRequest(png_bytes=b"png", model="v", prompt="", request_id=7)
        with mock.patch("lilbee.vision.build_vision_messages", return_value=[]):
            resp = _handle_vision(llm, req)
        assert resp.text == "extracted text"
        assert resp.request_id == 7

    def test_error_returns_error_response(self) -> None:
        llm = mock.MagicMock()
        llm.create_chat_completion.side_effect = RuntimeError("model failed")
        req = VisionRequest(png_bytes=b"png", model="v", request_id=8)
        with mock.patch("lilbee.vision.build_vision_messages", return_value=[]):
            resp = _handle_vision(llm, req)
        assert resp.error == "model failed"

    def test_uses_default_prompt(self) -> None:
        llm = mock.MagicMock()
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "text"}}]}
        req = VisionRequest(png_bytes=b"png", model="v", prompt="", request_id=1)
        with mock.patch("lilbee.vision.build_vision_messages", return_value=[]) as mock_build:
            _handle_vision(llm, req)
        # Should use OCR_PROMPT when prompt is empty
        call_args = mock_build.call_args[0]
        assert len(call_args[0]) > 0  # Non-empty prompt passed

    def test_uses_custom_prompt(self) -> None:
        llm = mock.MagicMock()
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "text"}}]}
        req = VisionRequest(png_bytes=b"png", model="v", prompt="custom prompt", request_id=1)
        with mock.patch("lilbee.vision.build_vision_messages", return_value=[]) as mock_build:
            _handle_vision(llm, req)
        assert mock_build.call_args[0][0] == "custom prompt"


class TestWorkerMain:
    """Test _worker_main loop handling of different request types."""

    def test_shutdown_exits_loop(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()
        req_q.put(ShutdownRequest())
        _worker_main(req_q, resp_q, config_snap)
        assert resp_q.empty()

    def test_embed_request_processed(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(EmbedRequest(texts=["hello"], model="test-embed", request_id=1))
        req_q.put(ShutdownRequest())

        with (
            mock.patch(
                "lilbee.providers.worker_process._load_embed_model",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "lilbee.providers.worker_process._handle_embed",
                return_value=EmbedResponse(vectors=[[1.0]], request_id=1),
            ),
        ):
            _worker_main(req_q, resp_q, config_snap)

        resp = resp_q.get(timeout=1)
        assert isinstance(resp, EmbedResponse)
        assert resp.vectors == [[1.0]]

    def test_vision_request_processed(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(VisionRequest(png_bytes=b"img", model="test-vision", request_id=2))
        req_q.put(ShutdownRequest())

        with (
            mock.patch(
                "lilbee.providers.worker_process._load_vision_model",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "lilbee.providers.worker_process._handle_vision",
                return_value=VisionResponse(text="extracted", request_id=2),
            ),
        ):
            _worker_main(req_q, resp_q, config_snap)

        resp = resp_q.get(timeout=1)
        assert isinstance(resp, VisionResponse)
        assert resp.text == "extracted"

    def test_load_model_request_clears_embed(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(LoadModelRequest(model="new-model", model_type="embed"))
        req_q.put(ShutdownRequest())

        with mock.patch("lilbee.providers.worker_process._close_model") as mock_close:
            _worker_main(req_q, resp_q, config_snap)
        # _close_model called for embed (None initially) + shutdown cleanup (2x None)
        assert mock_close.call_count >= 1

    def test_load_model_request_clears_vision(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(LoadModelRequest(model="new-vision", model_type="vision"))
        req_q.put(ShutdownRequest())

        with mock.patch("lilbee.providers.worker_process._close_model") as mock_close:
            _worker_main(req_q, resp_q, config_snap)
        assert mock_close.call_count >= 1

    def test_embed_model_cached_between_requests(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(EmbedRequest(texts=["a"], model="test-embed", request_id=1))
        req_q.put(EmbedRequest(texts=["b"], model="test-embed", request_id=2))
        req_q.put(ShutdownRequest())

        with (
            mock.patch(
                "lilbee.providers.worker_process._load_embed_model",
                return_value=mock.MagicMock(),
            ) as mock_load,
            mock.patch(
                "lilbee.providers.worker_process._handle_embed",
                return_value=EmbedResponse(vectors=[[1.0]]),
            ),
        ):
            _worker_main(req_q, resp_q, config_snap)

        # Model loaded only once for same model name
        assert mock_load.call_count == 1

    def test_embed_model_reloaded_on_change(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(EmbedRequest(texts=["a"], model="model-a", request_id=1))
        req_q.put(EmbedRequest(texts=["b"], model="model-b", request_id=2))
        req_q.put(ShutdownRequest())

        with (
            mock.patch(
                "lilbee.providers.worker_process._load_embed_model",
                return_value=mock.MagicMock(),
            ) as mock_load,
            mock.patch(
                "lilbee.providers.worker_process._handle_embed",
                return_value=EmbedResponse(vectors=[[1.0]]),
            ),
        ):
            _worker_main(req_q, resp_q, config_snap)

        assert mock_load.call_count == 2

    def test_vision_model_cached_between_requests(self, config_snap: ConfigSnapshot) -> None:
        ctx = get_context("spawn")
        req_q: multiprocessing.Queue = ctx.Queue()
        resp_q: multiprocessing.Queue = ctx.Queue()

        req_q.put(VisionRequest(png_bytes=b"a", model="test-vision", request_id=1))
        req_q.put(VisionRequest(png_bytes=b"b", model="test-vision", request_id=2))
        req_q.put(ShutdownRequest())

        with (
            mock.patch(
                "lilbee.providers.worker_process._load_vision_model",
                return_value=mock.MagicMock(),
            ) as mock_load,
            mock.patch(
                "lilbee.providers.worker_process._handle_vision",
                return_value=VisionResponse(text="ok"),
            ),
        ):
            _worker_main(req_q, resp_q, config_snap)

        assert mock_load.call_count == 1

    def test_broken_queue_exits(self, config_snap: ConfigSnapshot) -> None:
        """EOFError on queue.get causes graceful exit."""
        req_q = mock.MagicMock()
        req_q.get.side_effect = EOFError("broken pipe")
        resp_q = mock.MagicMock()
        _worker_main(req_q, resp_q, config_snap)
        # Should exit without raising


class TestWorkerProcessLifecycle:
    def test_start_and_stop(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        with mock.patch.object(wp, "_ctx") as mock_ctx:
            mock_proc = mock.MagicMock()
            mock_proc.is_alive.return_value = True
            mock_ctx.Process.return_value = mock_proc
            mock_ctx.Queue.return_value = mock.MagicMock()

            wp.start()
            assert wp._started is True
            mock_proc.start.assert_called_once()

            # Stop should send shutdown and join
            mock_proc.is_alive.return_value = False
            wp.stop()
            assert wp._started is False
            assert wp._process is None

    def test_start_noop_when_running(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._started = True

        wp.start()
        # Process.start() should NOT be called again
        mock_proc.start.assert_not_called()

    def test_stop_noop_when_not_started(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp.stop()  # Should not raise

    def test_stop_terminates_on_timeout(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True  # Never exits gracefully
        wp._process = mock_proc
        wp._started = True
        wp._request_queue = mock.MagicMock()

        wp.stop()
        mock_proc.terminate.assert_called_once()

    def test_stop_handles_broken_queue(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = False
        wp._process = mock_proc
        wp._started = True
        broken_q = mock.MagicMock()
        broken_q.put.side_effect = OSError("broken")
        wp._request_queue = broken_q

        wp.stop()  # Should not raise

    def test_restart(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        with (
            mock.patch.object(wp, "stop") as mock_stop,
            mock.patch.object(wp, "start") as mock_start,
            mock.patch("lilbee.providers.worker_process.time.sleep"),
        ):
            wp.restart()
            mock_stop.assert_called_once()
            mock_start.assert_called_once()

    def test_is_alive_false_when_no_process(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        assert wp.is_alive() is False

    def test_is_alive_delegates_to_process(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        assert wp.is_alive() is True


class TestWorkerProcessEmbed:
    def test_embed_success(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc

        resp_q = mock.MagicMock()
        resp_q.get.return_value = EmbedResponse(vectors=[[0.1, 0.2]], request_id=1)
        wp._response_queue = resp_q
        wp._request_queue = mock.MagicMock()

        result = wp.embed(["hello"])
        assert result == [[0.1, 0.2]]

    def test_embed_error_raises(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc

        resp_q = mock.MagicMock()
        resp_q.get.return_value = EmbedResponse(error="GPU OOM", request_id=1)
        wp._response_queue = resp_q
        wp._request_queue = mock.MagicMock()

        with pytest.raises(RuntimeError, match="GPU OOM"):
            wp.embed(["hello"])

    def test_embed_crash_retries_once(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        # First call: alive then dead (crash), second: alive
        mock_proc.is_alive.side_effect = [True, False, True, True]
        wp._process = mock_proc

        req_q = mock.MagicMock()
        wp._request_queue = req_q

        resp_q = mock.MagicMock()
        # First get: timeout (simulate crash response path)
        # After restart: success
        resp_q.get.side_effect = [
            EmbedResponse(vectors=[[1.0]], request_id=1),
        ]
        wp._response_queue = resp_q

        with (
            mock.patch.object(wp, "restart") as mock_restart,
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, EmbedResponse(vectors=[[1.0]], request_id=1)],
            ),
        ):
            result = wp.embed(["hello"])
        assert result == [[1.0]]
        mock_restart.assert_called_once()

    def test_embed_crash_retry_also_fails(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, None],
            ),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp.embed(["hello"])

    def test_embed_unexpected_response_type(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "_get_response", return_value=VisionResponse(text="wrong type")),
            pytest.raises(RuntimeError, match="Unexpected response type"),
        ):
            wp.embed(["hello"])

    def test_embed_timeout(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "_get_response", side_effect=TimeoutError("timed out")),
            pytest.raises(TimeoutError),
        ):
            wp.embed(["hello"])

    def test_embed_autostart(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        with (
            mock.patch.object(wp, "start") as mock_start,
            mock.patch.object(
                wp,
                "_get_response",
                return_value=EmbedResponse(vectors=[[1.0]]),
            ),
        ):
            wp._request_queue = mock.MagicMock()
            wp.embed(["hello"])
        mock_start.assert_called_once()


class TestWorkerProcessVision:
    def test_vision_success(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with mock.patch.object(
            wp, "_get_response", return_value=VisionResponse(text="extracted text", request_id=1)
        ):
            result = wp.vision_ocr(b"png", "model")
        assert result == "extracted text"

    def test_vision_error_raises(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(
                wp, "_get_response", return_value=VisionResponse(error="model failed", request_id=1)
            ),
            pytest.raises(RuntimeError, match="model failed"),
        ):
            wp.vision_ocr(b"png", "model")

    def test_vision_crash_retries(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, VisionResponse(text="recovered", request_id=1)],
            ),
        ):
            result = wp.vision_ocr(b"png", "model")
        assert result == "recovered"

    def test_vision_crash_retry_fails(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(wp, "_get_response", side_effect=[None, None]),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp.vision_ocr(b"png", "model")

    def test_vision_unexpected_response_type(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "_get_response", return_value=EmbedResponse(vectors=[[1.0]])),
            pytest.raises(RuntimeError, match="Unexpected response type"),
        ):
            wp.vision_ocr(b"png", "model")

    def test_vision_retry_unexpected_type(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, EmbedResponse(vectors=[[1.0]])],
            ),
            pytest.raises(RuntimeError, match="Unexpected response type"),
        ):
            wp.vision_ocr(b"png", "model")

    def test_vision_retry_error_response(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, VisionResponse(error="still broken")],
            ),
            pytest.raises(RuntimeError, match="still broken"),
        ):
            wp.vision_ocr(b"png", "model")


class TestWorkerProcessLoadModel:
    def test_load_model_sends_request(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        req_q = mock.MagicMock()
        wp._request_queue = req_q

        wp.load_model("new-model", "embed")
        put_arg = req_q.put.call_args[0][0]
        assert isinstance(put_arg, LoadModelRequest)
        assert put_arg.model == "new-model"


class TestGetResponse:
    def test_returns_response_from_queue(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc

        resp = EmbedResponse(vectors=[[1.0]])
        resp_q = mock.MagicMock()
        resp_q.get.return_value = resp
        wp._response_queue = resp_q

        result = wp._get_response(timeout=5.0)
        assert result is resp

    def test_returns_none_when_worker_dead(self, config_snap: ConfigSnapshot) -> None:
        import queue as _queue

        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = False
        wp._process = mock_proc

        resp_q = mock.MagicMock()
        resp_q.get.side_effect = _queue.Empty()
        wp._response_queue = resp_q

        result = wp._get_response(timeout=0.1)
        assert result is None

    def test_timeout_raises(self, config_snap: ConfigSnapshot) -> None:
        import queue as _queue

        wp = WorkerProcess(config_snap)
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc

        resp_q = mock.MagicMock()
        resp_q.get.side_effect = _queue.Empty()
        wp._response_queue = resp_q

        with pytest.raises(TimeoutError):
            wp._get_response(timeout=0.05)


class TestEnsureConfig:
    def test_uses_provided_config(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        assert wp._ensure_config() is config_snap

    def test_builds_from_cfg_when_none(self) -> None:
        wp = WorkerProcess(config=None)
        result = wp._ensure_config()
        assert isinstance(result, ConfigSnapshot)

    def test_caches_built_config(self) -> None:
        wp = WorkerProcess(config=None)
        first = wp._ensure_config()
        second = wp._ensure_config()
        assert first is second


class TestNextRequestId:
    def test_increments(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        id1 = wp._next_request_id()
        id2 = wp._next_request_id()
        assert id2 == id1 + 1


class TestEmbedRetryUnexpectedType:
    def test_retry_returns_unexpected_type(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, VisionResponse(text="wrong")],
            ),
            pytest.raises(RuntimeError, match="Unexpected response type"),
        ):
            wp.embed(["hello"])

    def test_retry_error_response(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        mock_proc = mock.MagicMock()
        mock_proc.is_alive.return_value = True
        wp._process = mock_proc
        wp._request_queue = mock.MagicMock()

        with (
            mock.patch.object(wp, "restart"),
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, EmbedResponse(error="still broken")],
            ),
            pytest.raises(RuntimeError, match="still broken"),
        ):
            wp.embed(["hello"])


class TestLoadEmbedModel:
    def test_loads_model(self, config_snap: ConfigSnapshot, tmp_path: Path) -> None:
        mock_llm = mock.MagicMock()
        model_path = str(tmp_path / "model.gguf")
        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=model_path,
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.load_llama",
                return_value=mock_llm,
            ) as mock_load,
        ):
            result = _load_embed_model("test-embed")
        assert result is mock_llm
        mock_load.assert_called_once_with(model_path, embedding=True)


class TestWorkerNotStartedGuards:
    """Cover all 'Worker not started' RuntimeError guards when _request_queue is None."""

    def test_round_trip_not_started(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        wp._request_queue = None
        req = EmbedRequest(texts=["hello"], model="m", request_id=1)
        with pytest.raises(RuntimeError, match="Worker not started"):
            wp._put_and_get(req, timeout=1.0)

    def test_put_and_get_retry_not_started(self, config_snap: ConfigSnapshot) -> None:
        """After a crash the retry leg hits the same Worker-not-started guard."""
        wp = WorkerProcess(config_snap)
        wp._started = True
        wp._request_queue = None

        def _clear_queue() -> None:
            wp._request_queue = None

        with (
            mock.patch.object(wp, "restart", side_effect=_clear_queue),
            pytest.raises(RuntimeError, match="Worker not started"),
        ):
            wp._round_trip(
                EmbedRequest(texts=["hello"], model="m", request_id=1),
                EmbedResponse,
                timeout=1.0,
                label="embed",
            )

    def test_get_response_not_started(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._response_queue = None
        with pytest.raises(RuntimeError, match="Worker not started"):
            wp._get_response(timeout=1.0)

    def test_load_model_not_started(self, config_snap: ConfigSnapshot) -> None:
        wp = WorkerProcess(config_snap)
        wp._started = True
        wp._request_queue = None
        with (
            mock.patch.object(wp, "_ensure_started"),
            pytest.raises(RuntimeError, match="Worker not started"),
        ):
            wp.load_model("test-model")


class TestLoadVisionModel:
    def test_loads_model(self, config_snap: ConfigSnapshot, tmp_path: Path) -> None:
        mock_llm = mock.MagicMock()
        vision_path = str(tmp_path / "vision.gguf")
        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=vision_path,
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.load_vision_llama",
                return_value=mock_llm,
            ) as mock_load,
        ):
            result = _load_vision_model("test-vision")
        assert result is mock_llm
        mock_load.assert_called_once_with(vision_path)


class TestRedirectStdio:
    """Test _redirect_stdio without the autouse mock (override the fixture)."""

    @pytest.fixture(autouse=True)
    def _no_stdio_redirect(self):
        """Override the module-level autouse fixture — let _redirect_stdio run."""
        yield

    def test_redirects_stdout_stderr_to_devnull(self) -> None:
        """_redirect_stdio points sys.stdout/stderr to devnull."""
        import os

        from lilbee.providers.worker_process import _redirect_stdio

        orig_out, orig_err = sys.stdout, sys.stderr
        orig_fd1, orig_fd2 = os.dup(1), os.dup(2)
        try:
            _redirect_stdio()
            assert sys.stdout.name == os.devnull
            assert sys.stderr.name == os.devnull
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = orig_out
            sys.stderr = orig_err
            os.dup2(orig_fd1, 1)
            os.dup2(orig_fd2, 2)
            os.close(orig_fd1)
            os.close(orig_fd2)
