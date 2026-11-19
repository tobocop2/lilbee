"""A bare Enter at each destructive confirm prompt answers no and runs nothing destructive."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg

runner = CliRunner()

# A case arms its stand-ins and returns the command line plus the call that must not happen.
Arm = Callable[[pytest.MonkeyPatch], tuple[list[str], MagicMock]]


def _no_overrides(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    for module in modules:
        monkeypatch.setattr(f"{module}.apply_overrides", MagicMock())
    monkeypatch.setattr(cfg, "json_mode", False)


def _model_rm(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.model")
    removal = MagicMock()
    monkeypatch.setattr("lilbee.cli.model.remove_model_data", removal)
    return ["model", "rm", "some-model"], removal


def _sessions_delete(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.sessions")
    store = MagicMock()
    monkeypatch.setattr("lilbee.cli.sessions._store", lambda: store)
    monkeypatch.setattr("lilbee.cli.sessions._resolve_id", lambda session_id: "abcdef123456")
    return ["sessions", "delete", "abcdef12"], store.delete


def _preset_remove(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.preset")
    removal = MagicMock()
    monkeypatch.setattr("lilbee.cli.preset.remove_preset", removal)
    return ["preset", "remove", "omlx"], removal


def _wiki_wipe(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.commands.wiki")
    monkeypatch.setattr("lilbee.cli.commands.wiki.get_services", MagicMock())
    wipe = MagicMock()
    monkeypatch.setattr("lilbee.wiki.wipe.wipe_wiki", wipe)
    return ["wiki", "wipe"], wipe


def _login(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "existing-token")
    monkeypatch.setattr("huggingface_hub.login", MagicMock())
    opened = MagicMock()
    monkeypatch.setattr("webbrowser.open", opened)
    return ["login"], opened


def _reset(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.commands.meta")
    reset = MagicMock()
    monkeypatch.setattr("lilbee.cli.commands.meta.perform_reset", reset)
    return ["reset"], reset


CASES: dict[str, Arm] = {
    "model rm": _model_rm,
    "sessions delete": _sessions_delete,
    "preset remove": _preset_remove,
    "wiki wipe": _wiki_wipe,
    "login again": _login,
    "reset": _reset,
}


def test_the_census_covers_every_prompt_site() -> None:
    assert len(CASES) == 6


@pytest.mark.parametrize("answer", ["\n", "n\n"], ids=["bare-enter", "explicit-n"])
@pytest.mark.parametrize("arm", CASES.values(), ids=CASES.keys())
def test_a_declined_prompt_runs_nothing_destructive(
    arm: Arm, answer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, destructive = arm(monkeypatch)
    result = runner.invoke(app, argv, input=answer)
    destructive.assert_not_called()
    assert "[y/N]" in result.output
