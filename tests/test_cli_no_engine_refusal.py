"""A pull of a model with no engine chosen downloads it and warns, read from the CLI's stdout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.catalog.hf_client import HfClient
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.modelhub.model_manager import ModelManager
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.provider import FleetProvider
from tests.conformance.fake_adapter import BundledFakeAdapter
from tests.conftest import make_mock_services

runner = CliRunner()
REF = "org/embedder#safetensors"
HINT = "Choose tei with lilbee engine prefer. It applies to every model without its own choice."
NOTICE = f"No engine is chosen by default for the safetensors format of {REF}. {HINT}"
# The adopt shows the settings notice of the embedding role, once, instead of the pull notice.
ROLE_NOTICE = (
    f"For embed: No engine is chosen by default for the safetensors format of {REF}. "
    "Choose tei with lilbee engine use for this model, or with lilbee engine prefer "
    "for every model without its own choice."
)


class _SafetensorsEngine(BundledFakeAdapter):
    """An installed engine that loads safetensors and is the default for no format."""

    name = "tei"
    formats = frozenset({"safetensors"})


@pytest.fixture(autouse=True)
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[str]]:
    """An installed but unchosen safetensors engine; every download that starts is recorded."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.TEI, _SafetensorsEngine())  # type: ignore[arg-type]
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.models_dir = tmp_path / "models"
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    # The CLI callback resets cfg, so the eager start is cut at the provider instead.
    monkeypatch.setattr(FleetProvider, "warm_up_pool", lambda self: None)
    started: list[str] = []

    def _download(entry: object, engine: object, **_kw: object) -> Path:
        started.append(REF)
        return tmp_path / "models" / "model.safetensors"

    monkeypatch.setattr("lilbee.catalog.download_model", _download)
    store = MagicMock()
    store.index_mismatch.return_value = None
    svc_mod.set_services(
        make_mock_services(
            store=store, model_manager=ModelManager(cfg.models_dir, cfg, hf_client=HfClient(cfg))
        )
    )
    yield started
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _stdout(args: list[str]) -> tuple[int, str]:
    result = runner.invoke(app, args)
    return result.exit_code, " ".join(result.stdout.split())


def test_a_pull_with_no_engine_chosen_downloads_and_warns_naming_prefer_only(
    downloads: list[str],
) -> None:
    code, out = _stdout(["model", "pull", REF])
    assert (code, downloads) == (0, [REF])
    assert f"Warning: {NOTICE}" in out
    assert "engine use" not in out


def test_use_embedder_with_no_engine_chosen_downloads_and_warns_on_stdout(
    downloads: list[str],
) -> None:
    code, out = _stdout(["use-embedder", REF])
    assert (code, downloads) == (0, [REF])
    assert f"Now embedding with {REF}. Warning: {ROLE_NOTICE}" in out
    assert out.count("No engine is chosen") == 1


def test_use_embedder_json_carries_the_warning(downloads: list[str]) -> None:
    result = runner.invoke(app, ["--json", "use-embedder", REF])
    assert (result.exit_code, downloads) == (0, [REF])
    assert json.loads(result.stdout)["warning"] == ROLE_NOTICE


def test_the_environment_variable_does_not_choose_the_engine_for_a_pull(
    monkeypatch: pytest.MonkeyPatch, downloads: list[str]
) -> None:
    """``preferred_engine`` is written only by the engine commands, so the variable is no source."""
    monkeypatch.setenv("LILBEE_PREFERRED_ENGINE", "tei")
    code, out = _stdout(["model", "pull", REF])
    assert (code, downloads) == (0, [REF])
    assert f"Warning: {NOTICE}" in out
