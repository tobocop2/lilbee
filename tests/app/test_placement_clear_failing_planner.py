"""A placement clear that cannot read the plan still says what the running fleet keeps."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lilbee import mcp_server
from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole
from lilbee.server import handlers

_SPEC = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
_CHAT = "tiny/chat-repo/chat-Q4_K_M.gguf"
_EMBED = "tiny/embed-repo/embed-Q8_0.gguf"


class _Provider:
    """A fleet provider that records re-plans and serves the roles in ``served``."""

    def __init__(self) -> None:
        self.replans = 0
        self.served = {WorkerRole.CHAT, WorkerRole.EMBED}

    def replan_fleet(self, *, wait: bool = False) -> None:
        self.replans += 1

    def role_ready(self, role: WorkerRole) -> bool:
        return role in self.served


def _planner_fails(spec, reading, **kwargs):
    raise OSError("the engine binary cannot be read")


@pytest.fixture
def running_fleet(monkeypatch, tmp_path):
    """A data root with a saved placement and a live fleet whose planner cannot read a plan."""
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "chat_model", _CHAT)
    monkeypatch.setattr(cfg, "embedding_model", _EMBED)
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    provider = _Provider()
    monkeypatch.setattr(app_placement, "peek_services", lambda: SimpleNamespace(provider=provider))
    monkeypatch.setattr(app_placement, "_resolve", _planner_fails)
    settings.update_values(tmp_path, {"placement": _SPEC.to_json()})
    settings.sync_from_resolver(cfg, ["placement"])
    assert cfg.placement == _SPEC.to_json()
    return provider


def _notices(skipped):
    return {skip.role: skip for skip in skipped}


def test_the_view_names_the_models_the_running_fleet_keeps(running_fleet, tmp_path):
    view = app_placement.set_placement(None)
    assert "placement" not in settings.load(tmp_path)
    skipped = _notices(view.skipped_no_engine)
    assert set(skipped) == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert skipped[WorkerRole.CHAT].model == _CHAT
    assert skipped[WorkerRole.EMBED].model == _EMBED
    assert all("still running" in skip.notice for skip in skipped.values())
    assert all("Restart lilbee" in skip.notice for skip in skipped.values())
    assert running_fleet.replans == 0


def test_the_notice_says_why_the_clear_is_not_applied(running_fleet):
    view = app_placement.set_placement(None)
    chat = _notices(view.skipped_no_engine)[WorkerRole.CHAT]
    assert chat.notice == (
        f"The chat model {_CHAT} is still running: the placement could not be read, "
        "so the clear is saved but not applied. Restart lilbee to apply it."
    )


def test_a_clear_with_no_fleet_running_reports_nothing_running(running_fleet, monkeypatch):
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    view = app_placement.set_placement(None)
    assert "placement" not in settings.load(cfg.data_root)
    assert view.skipped_no_engine == ()


def test_a_role_with_no_model_is_not_reported_running(running_fleet):
    view = app_placement.set_placement(None)
    assert WorkerRole.RERANK not in _notices(view.skipped_no_engine)
    assert WorkerRole.VISION not in _notices(view.skipped_no_engine)


def test_a_role_configured_but_not_served_is_not_reported_running(running_fleet, monkeypatch):
    monkeypatch.setattr(cfg, "reranker_model", "tiny/rerank-repo/rerank-Q8_0.gguf")
    view = app_placement.set_placement(None)
    assert set(_notices(view.skipped_no_engine)) == {WorkerRole.CHAT, WorkerRole.EMBED}


def test_an_engine_notice_and_a_running_notice_on_one_role_keep_the_engine_notice(
    running_fleet, monkeypatch
):
    book_notice = "No engine is installed for the chat model."
    monkeypatch.setattr(
        app_placement,
        "_planning_engines",
        lambda: BookReading(adapters={}, notices={WorkerRole.CHAT: book_notice}),
    )
    skipped = _notices(app_placement.set_placement(None).skipped_no_engine)
    assert skipped[WorkerRole.CHAT].notice == book_notice
    assert "still running" in skipped[WorkerRole.EMBED].notice


def test_a_role_on_a_remote_model_is_not_reported_running(running_fleet, monkeypatch):
    monkeypatch.setattr(cfg, "chat_model", "openai/gpt-4o")
    view = app_placement.set_placement(None)
    assert set(_notices(view.skipped_no_engine)) == {WorkerRole.EMBED}


def test_the_rest_answer_names_the_running_models(running_fleet):
    response = asyncio.run(handlers.placement_clear())
    assert {skip.model for skip in response.skipped_no_engine} == {_CHAT, _EMBED}


def test_the_mcp_answer_names_the_running_models(running_fleet):
    answer = mcp_server.clear_placement_tool()
    assert {skip["model"] for skip in answer["skipped_no_engine"]} == {_CHAT, _EMBED}


def test_the_cli_answer_names_the_running_models(running_fleet):
    result = CliRunner().invoke(placement_app, ["clear"])
    assert result.exit_code == 0, result.output
    assert _CHAT in result.output
    assert "still running" in result.output


def test_the_cli_json_answer_names_the_running_models(running_fleet):
    cfg.json_mode = True
    try:
        result = CliRunner().invoke(placement_app, ["clear"])
    finally:
        cfg.json_mode = False
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert {skip["model"] for skip in body["skipped_no_engine"]} == {_CHAT, _EMBED}
