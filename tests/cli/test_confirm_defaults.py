"""A bare Enter at each destructive confirm prompt answers no and runs nothing destructive."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

import lilbee
from lilbee.app.engine_service import EmbedConfirmRequiredError
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


def _remove_glob(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    _no_overrides(monkeypatch, "lilbee.cli.commands.ingest_sync")
    documents = ["docs/a.pdf", "docs/b.pdf"]
    monkeypatch.setattr(
        "lilbee.cli.commands.ingest_sync.iter_removable_names", lambda: iter(documents)
    )
    monkeypatch.setattr(
        "lilbee.cli.commands.ingest_sync.removable_among", lambda names: set(documents)
    )
    monkeypatch.setattr(
        "lilbee.cli.commands.ingest_sync.expand_remove_targets", lambda names, known: documents
    )
    removal = MagicMock()
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync._remove_or_exit", removal)
    return ["remove", "docs"], removal


def _engine_prefer(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], MagicMock]:
    monkeypatch.setattr(cfg, "json_mode", False)
    monkeypatch.setattr("lilbee.cli.engine._interactive", lambda: True)
    write = MagicMock(return_value="openvino")

    def prefer(name: str, confirmed: bool = False) -> str:
        if not confirmed:
            raise EmbedConfirmRequiredError("library", 3)
        return write(name)

    monkeypatch.setattr("lilbee.app.engine_service.prefer", prefer)
    return ["engine", "prefer", "openvino"], write


# Every confirm call site, keyed by file under src/lilbee and the function that holds the call.
CASES: dict[str, Arm] = {
    "cli/model.py::_confirm_remove_or_exit": _model_rm,
    "cli/sessions.py::delete_cmd": _sessions_delete,
    "cli/preset.py::remove_cmd": _preset_remove,
    "cli/commands/wiki.py::wiki_wipe": _wiki_wipe,
    "cli/commands/setup.py::login": _login,
    "cli/commands/meta.py::reset": _reset,
    "cli/commands/ingest_sync.py::remove": _remove_glob,
    "cli/engine.py::_confirming": _engine_prefer,
}

EXEMPT: dict[str, str] = {
    "cli/launchers/setup_gate.py::confirm_first_run_setup": (
        "the setup question defaults to yes by design and runs no destructive action"
    ),
}


def _confirm_sites() -> set[str]:
    """Find every call named confirm under src/lilbee, as file::enclosing function."""
    root = Path(lilbee.__file__).parent
    sites: set[str] = set()

    def visit(node: ast.AST, label: str, holder: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            holder = node.name
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "confirm") or (
                isinstance(func, ast.Name) and func.id == "confirm"
            ):
                sites.add(f"{label}::{holder}")
        for child in ast.iter_child_nodes(node):
            visit(child, label, holder)

    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visit(tree, path.relative_to(root).as_posix(), "")
    return sites


def test_every_confirm_call_is_a_case_or_a_named_exemption() -> None:
    sites = _confirm_sites()
    assert sites
    assert not CASES.keys() & EXEMPT.keys()
    assert sites == CASES.keys() | EXEMPT.keys()


@pytest.mark.parametrize("answer", ["\n", "n\n"], ids=["bare-enter", "explicit-n"])
@pytest.mark.parametrize("arm", CASES.values(), ids=CASES.keys())
def test_a_declined_prompt_runs_nothing_destructive(
    arm: Arm, answer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, destructive = arm(monkeypatch)
    result = runner.invoke(app, argv, input=answer)
    destructive.assert_not_called()
    assert "[y/N]" in result.output
