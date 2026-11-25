"""`lilbee placement show` names the engine backend, and says what to do when it is unknown."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.app.placement import PlacementView
from lilbee.cli import placement as cli_placement
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.roles import EngineBackend


def _show(monkeypatch: pytest.MonkeyPatch, backend: EngineBackend, *, json_mode: bool = False) -> str:
    view = PlacementView(
        gpus=(), roles=(), unplaceable=(), manual=False, spec_json=None, engine_backend=backend
    )
    monkeypatch.setattr(cli_placement, "get_placement", lambda: view)
    monkeypatch.setattr(cfg, "json_mode", json_mode)
    result = CliRunner().invoke(placement_app, ["show"])
    assert result.exit_code == 0
    return " ".join(result.output.split())


def test_a_cpu_host_names_its_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "Engine backend: cpu" in _show(monkeypatch, EngineBackend.CPU)


def test_an_unanswered_probe_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _show(monkeypatch, EngineBackend.UNKNOWN)
    assert "Engine backend: unknown" in out
    assert "lilbee engine install" in out


def test_json_output_keeps_its_shape_without_the_text_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _show(monkeypatch, EngineBackend.CPU, json_mode=True)
    assert json.loads(out)["engine_backend"] == "cpu"
    assert "Engine backend:" not in out
