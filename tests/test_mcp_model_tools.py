"""Tests for MCP model management tools."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.catalog import DownloadProgress
from lilbee.cli.model import (
    ListModelsResult,
    PullResult,
    PullStatus,
    RemoveResult,
    ShowModelResult,
)
from lilbee.mcp import (
    _log_progress_failure,
    model_list,
    model_pull,
    model_rm,
    model_show,
)
from lilbee.model_manager import ModelNotFoundError, ModelSource


class TestMcpList:
    def test_native_source_forwarded(self):
        expected = ListModelsResult(models=[], total=0)
        with patch("lilbee.cli.model.list_models_data", return_value=expected) as fn:
            result = model_list(source="native", task="chat")
        assert result == expected.model_dump()
        fn.assert_called_once_with(source=ModelSource.NATIVE, task="chat")

    def test_empty_strings_mean_all(self):
        expected = ListModelsResult(models=[], total=0)
        with patch("lilbee.cli.model.list_models_data", return_value=expected) as fn:
            model_list()
        fn.assert_called_once_with(source=None, task=None)

    def test_invalid_source_returns_explicit_error(self):
        with patch("lilbee.cli.model.list_models_data") as fn:
            result = model_list(source="bogus")
        assert result == {"error": "invalid source 'bogus'; expected one of: native, litellm"}
        fn.assert_not_called()


class TestMcpShow:
    def test_delegates_and_serializes(self):
        expected = ShowModelResult(model="qwen3:0.6b", installed=True, source="native")
        with patch("lilbee.cli.model.show_model_data", return_value=expected) as fn:
            result = model_show("qwen3:0.6b")
        fn.assert_called_once_with("qwen3:0.6b")
        assert result == expected.model_dump()

    def test_not_found_returns_error_dict(self):
        with patch(
            "lilbee.cli.model.show_model_data",
            side_effect=ModelNotFoundError("model not found: ghost"),
        ):
            result = model_show("ghost")
        assert result == {"error": "model not found: ghost"}


class TestMcpRemove:
    def test_default_source_is_none(self):
        expected = RemoveResult(model="qwen3:0.6b", deleted=True, freed_gb=5.0)
        with patch("lilbee.cli.model.remove_model_data", return_value=expected) as fn:
            result = model_rm("qwen3:0.6b")
        fn.assert_called_once_with("qwen3:0.6b", source=None)
        assert result == expected.model_dump()

    def test_native_source(self):
        expected = RemoveResult(model="qwen3:0.6b", deleted=True)
        with patch("lilbee.cli.model.remove_model_data", return_value=expected) as fn:
            model_rm("qwen3:0.6b", source="native")
        fn.assert_called_once_with("qwen3:0.6b", source=ModelSource.NATIVE)

    def test_invalid_source_explicit_error(self):
        with patch("lilbee.cli.model.remove_model_data") as fn:
            result = model_rm("qwen3:0.6b", source="bogus")
        assert result == {"error": "invalid source 'bogus'; expected one of: native, litellm"}
        fn.assert_not_called()


class TestMcpPull:
    @staticmethod
    def _fake_ctx() -> MagicMock:
        ctx = MagicMock()
        ctx.report_progress = AsyncMock()
        return ctx

    @pytest.mark.asyncio
    async def test_streams_progress_and_returns_final_dict(self):
        ctx = self._fake_ctx()
        final = PullResult(model="qwen3:0.6b", source="native", status=PullStatus.OK)

        def fake_pull(model, source, *, on_update):
            on_update(DownloadProgress(percent=10, detail="10 MB", is_cache_hit=False))
            on_update(DownloadProgress(percent=50, detail="50 MB", is_cache_hit=False))
            return final

        with patch("lilbee.cli.model.pull_model_data", side_effect=fake_pull):
            result = await model_pull("qwen3:0.6b", source="native", ctx=ctx)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert result == final.model_dump()
        assert ctx.report_progress.await_count == 2
        first = ctx.report_progress.await_args_list[0]
        assert first.kwargs == {
            "progress": 10.0,
            "total": 100.0,
            "message": "10 MB",
        }

    @pytest.mark.asyncio
    async def test_pull_without_ctx_does_not_report_progress(self):
        final = PullResult(model="qwen3:0.6b", source="native", status=PullStatus.OK)

        def fake_pull(model, source, *, on_update):
            on_update(DownloadProgress(percent=30, detail="", is_cache_hit=False))
            return final

        with patch("lilbee.cli.model.pull_model_data", side_effect=fake_pull):
            result = await model_pull("qwen3:0.6b")
        assert result == final.model_dump()

    @pytest.mark.asyncio
    async def test_pull_litellm_source(self):
        captured: list[ModelSource] = []
        final = PullResult(model="llama3:latest", source="litellm", status=PullStatus.OK)

        def fake_pull(model, source, *, on_update):
            captured.append(source)
            return final

        with patch("lilbee.cli.model.pull_model_data", side_effect=fake_pull):
            await model_pull("llama3:latest", source="litellm")
        assert captured == [ModelSource.LITELLM]

    @pytest.mark.asyncio
    async def test_pull_runtime_error_returned_as_dict(self):
        with patch(
            "lilbee.cli.model.pull_model_data",
            side_effect=RuntimeError("no network"),
        ):
            result = await model_pull("qwen3:0.6b")
        assert result == {"error": "no network"}

    @pytest.mark.asyncio
    async def test_pull_permission_error_returned_as_dict(self):
        with patch(
            "lilbee.cli.model.pull_model_data",
            side_effect=PermissionError("gated"),
        ):
            result = await model_pull("qwen3:0.6b")
        assert result == {"error": "gated"}

    @pytest.mark.asyncio
    async def test_pull_invalid_source(self):
        result = await model_pull("qwen3:0.6b", source="bogus")
        assert result == {"error": "invalid source 'bogus'; expected one of: native, litellm"}


class TestLogProgressFailure:
    def test_success_is_silent(self, caplog):
        fut: Future[None] = Future()
        fut.set_result(None)
        with caplog.at_level(logging.WARNING, logger="lilbee.mcp"):
            _log_progress_failure(fut)
        assert "report_progress failed" not in caplog.text

    def test_exception_is_logged_at_warning(self, caplog):
        fut: Future[None] = Future()
        fut.set_exception(RuntimeError("notify failed"))
        with caplog.at_level(logging.WARNING, logger="lilbee.mcp"):
            _log_progress_failure(fut)
        assert "report_progress failed" in caplog.text
        assert "notify failed" in caplog.text
