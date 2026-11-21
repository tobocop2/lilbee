"""The repo id error names the expected form, an example, and the rejected value."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog.refs import REPO_ID_HINT, invalid_repo_id_message
from lilbee.catalog.types import ModelFormat, ModelSource
from lilbee.cli import app
from lilbee.mcp_server import model_pull
from lilbee.modelhub.model_manager import ModelManager, ModelNotFoundError
from lilbee.modelhub.registry import ModelRegistry, _validate_hf_repo, parse_hf_ref
from lilbee.server.handlers.models import models_pull

_BAD_REF = "standalone"
_EXAMPLE = "Qwen/Qwen3-0.6B-GGUF"
_runner = CliRunner()


@pytest.fixture
def real_manager(tmp_path: Path) -> Iterator[ModelManager]:
    """A real manager over an empty models dir; the engine and arch checks are stubbed."""
    manager = ModelManager(tmp_path, MagicMock())
    with (
        patch.object(ModelManager, "require_engine", return_value=MagicMock()),
        patch.object(ModelManager, "enforce_arch_compat"),
        patch("lilbee.app.models.get_services", return_value=MagicMock(model_manager=manager)),
        patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=manager),
        ),
    ):
        yield manager


class TestMessageText:
    def test_names_value_form_and_example(self) -> None:
        text = invalid_repo_id_message("bad repo")
        assert "'bad repo'" in text
        assert "'<org>/<repo>'" in text
        assert f"'{_EXAMPLE}'" in text

    def test_hint_is_part_of_the_message(self) -> None:
        assert invalid_repo_id_message("x").endswith(REPO_ID_HINT)


class TestRegistryValidator:
    @pytest.mark.parametrize("value", ["standalone", "a/b/c", "../etc/passwd", ""])
    def test_rejected_repo_carries_the_message(self, value: str) -> None:
        with pytest.raises(ValueError) as caught:
            _validate_hf_repo(value)
        assert str(caught.value) == invalid_repo_id_message(value)

    def test_ref_with_a_bad_repo_segment_carries_the_message(self) -> None:
        with pytest.raises(ValueError) as caught:
            parse_hf_ref("bad repo/name/file.gguf")
        assert str(caught.value) == invalid_repo_id_message("bad repo/name")

    def test_resolving_a_bad_non_gguf_repo_carries_the_message(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as caught:
            ModelRegistry(tmp_path).resolve("bad repo/name#mlx")
        assert str(caught.value) == invalid_repo_id_message("bad repo/name")


class TestPullSurfaces:
    """Every pull surface shows the form and the example for a ref that is no repo id."""

    def test_manager_pull_refusal(self, real_manager: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError) as caught:
            real_manager.pull(_BAD_REF, ModelSource.NATIVE)
        assert f"Model '{_BAD_REF}' not recognized. {REPO_ID_HINT}" in str(caught.value)

    def test_manager_pull_refusal_text(self, real_manager: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError) as caught:
            real_manager.pull(_BAD_REF, ModelSource.NATIVE)
        assert str(caught.value) == (
            "Model 'standalone' not recognized. "
            "Write a repo id as '<org>/<repo>', for example 'Qwen/Qwen3-0.6B-GGUF'. "
            "Or give a featured model name."
        )

    def test_cli_json(self, real_manager: ModelManager) -> None:
        result = _runner.invoke(app, ["--json", "model", "pull", _BAD_REF])
        assert result.exit_code == 1, result.output
        error = json.loads(result.output.strip().splitlines()[-1])["error"]
        assert REPO_ID_HINT in error
        assert _BAD_REF in error

    def test_mcp(self, real_manager: ModelManager) -> None:
        result = asyncio.run(model_pull(_BAD_REF, source="native", ctx=None))
        assert REPO_ID_HINT in result["error"]
        assert _BAD_REF in result["error"]

    def test_rest_stream(self, real_manager: ModelManager) -> None:
        async def collect() -> str:
            return "".join([event async for event in models_pull(_BAD_REF, source="native")])

        body = asyncio.run(collect())
        assert "event: error" in body
        assert REPO_ID_HINT in body


_BAD_FORMAT_REF = "org/repo#mxl"
_FORMAT_REFUSAL = (
    f"Model ref {_BAD_FORMAT_REF!r} names format 'mxl'. "
    f"Use one of: {', '.join(fmt.value for fmt in ModelFormat)}."
)


_MLX_SHAPE = f"must be '<org>/<repo>#{ModelFormat.MLX}'."
_SHAPE_REFUSALS = {
    "a/b/c#mlx": f"Model ref 'a/b/c#mlx' {_MLX_SHAPE}",
    "standalone#mlx": f"Model ref 'standalone#mlx' {_MLX_SHAPE}",
    "foo#xdna": f"Model ref 'foo#xdna' must be '<name>:<tag>#{ModelFormat.XDNA}'.",
}


class TestBadFormatShapeSurfaces:
    """A valid format on a wrongly shaped repo shows the parser's shape text, not the hint."""

    @pytest.mark.parametrize("ref", _SHAPE_REFUSALS)
    def test_manager_pull_refusal(self, real_manager: ModelManager, ref: str) -> None:
        with pytest.raises(ModelNotFoundError) as caught:
            real_manager.pull(ref, ModelSource.NATIVE)
        assert str(caught.value) == _SHAPE_REFUSALS[ref]

    @pytest.mark.parametrize("ref", _SHAPE_REFUSALS)
    def test_cli_json(self, real_manager: ModelManager, ref: str) -> None:
        result = _runner.invoke(app, ["--json", "model", "pull", ref])
        assert result.exit_code == 1, result.output
        assert json.loads(result.output.strip().splitlines()[-1])["error"] == _SHAPE_REFUSALS[ref]

    @pytest.mark.parametrize("ref", _SHAPE_REFUSALS)
    def test_mcp(self, real_manager: ModelManager, ref: str) -> None:
        result = asyncio.run(model_pull(ref, source="native", ctx=None))
        assert result["error"] == _SHAPE_REFUSALS[ref]

    @pytest.mark.parametrize("ref", _SHAPE_REFUSALS)
    def test_rest_stream(self, real_manager: ModelManager, ref: str) -> None:
        async def collect() -> str:
            return "".join([event async for event in models_pull(ref, source="native")])

        body = asyncio.run(collect())
        assert "event: error" in body
        assert _SHAPE_REFUSALS[ref] in body


class TestBadFormatQualifierSurfaces:
    """A pull of a ref with an unknown format shows the bad qualifier and the valid formats."""

    def test_valid_formats_are_listed(self) -> None:
        assert len(list(ModelFormat)) > 1
        assert "gguf" in _FORMAT_REFUSAL

    def test_manager_pull_refusal(self, real_manager: ModelManager) -> None:
        with pytest.raises(ModelNotFoundError) as caught:
            real_manager.pull(_BAD_FORMAT_REF, ModelSource.NATIVE)
        assert str(caught.value) == _FORMAT_REFUSAL

    def test_cli_json(self, real_manager: ModelManager) -> None:
        result = _runner.invoke(app, ["--json", "model", "pull", _BAD_FORMAT_REF])
        assert result.exit_code == 1, result.output
        error = json.loads(result.output.strip().splitlines()[-1])["error"]
        assert error == _FORMAT_REFUSAL

    def test_mcp(self, real_manager: ModelManager) -> None:
        result = asyncio.run(model_pull(_BAD_FORMAT_REF, source="native", ctx=None))
        assert result["error"] == _FORMAT_REFUSAL

    def test_rest_stream(self, real_manager: ModelManager) -> None:
        async def collect() -> str:
            events = models_pull(_BAD_FORMAT_REF, source="native")
            return "".join([event async for event in events])

        body = asyncio.run(collect())
        assert "event: error" in body
        assert "names format 'mxl'" in body
        assert "Use one of: " in body
