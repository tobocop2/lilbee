"""The refusal of a model with no engine chosen, read from the CLI's stdout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from tests.conformance.fake_adapter import BundledFakeAdapter

runner = CliRunner()
REF = "org/embedder#safetensors"
HINT = "Choose tei with lilbee engine prefer. It applies to every model without its own choice."


class _SafetensorsEngine(BundledFakeAdapter):
    """An installed engine that loads safetensors and is the default for no format."""

    name = "tei"
    formats = frozenset({"safetensors"})


@pytest.fixture(autouse=True)
def _installed_safetensors_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    monkeypatch.setitem(choice.ADAPTERS, EngineName.TEI, _SafetensorsEngine())  # type: ignore[arg-type]
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(cfg, "preferred_engine", EngineSetting.AUTO)
    monkeypatch.setattr(cfg, "engine_choices", None)
    yield


def _stdout(args: list[str]) -> tuple[int, str]:
    result = runner.invoke(app, args)
    return result.exit_code, " ".join(result.stdout.split())


def test_a_pull_with_no_engine_chosen_names_prefer_only() -> None:
    code, out = _stdout(["model", "pull", REF])
    assert code == 1
    assert f"No engine is chosen by default for the safetensors format of {REF}. {HINT}" in out
    assert "engine use" not in out


def test_use_embedder_with_no_engine_chosen_prints_the_refusal_on_stdout() -> None:
    code, out = _stdout(["use-embedder", REF])
    assert code == 1
    assert (
        f"Error: No engine is chosen by default for the safetensors format of {REF}. {HINT}" in out
    )
    assert "engine use" not in out


def test_use_embedder_json_carries_the_refusal() -> None:
    result = runner.invoke(app, ["--json", "use-embedder", REF])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"].endswith(HINT)


def test_the_environment_variable_does_not_choose_the_engine_for_a_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``preferred_engine`` is written only by the engine commands, so the variable is no source."""
    monkeypatch.setenv("LILBEE_PREFERRED_ENGINE", "tei")
    code, out = _stdout(["model", "pull", REF])
    assert code == 1
    assert "No engine is chosen by default" in out
