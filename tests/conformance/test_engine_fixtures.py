"""Replays each recorded engine run through the real provider, with no server."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.factory import sdk_library
from tests.conformance.preset_check import (
    ENGINE_FIXTURES,
    Exchange,
    configured,
    load_fixture,
    replaying,
    run_steps,
    steps_for,
)

# The replay drives the real provider, so it runs where the extra is installed:
# the integration job names this directory's tests in `make test-integration`.
pytest.importorskip(sdk_library().module)

_FILES = sorted(ENGINE_FIXTURES.glob("**/*.json"))


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_replay_gives_the_recorded_results(path: Path, tmp_path: Path) -> None:
    setup, recorded, exchanges, doc = load_fixture(path)
    assert doc.get("strict") is True, "an engine recording sets strict"
    assert exchanges, "the fixture holds no exchange"
    assert set(recorded) == set(steps_for(setup))
    with replaying(exchanges) as url, configured(url, setup, tmp_path):
        replayed = run_steps(setup)
    assert replayed == recorded
    assert all(outcome.ok for outcome in recorded.values())


def _chat_bodies(exchanges: list[Exchange]) -> Iterator[tuple[Exchange, dict[str, Any]]]:
    for exchange in exchanges:
        if exchange.status == 200 and exchange.path.endswith("/chat/completions"):
            body = json.loads(exchange.body)
            if body.get("choices"):
                yield exchange, body


def _replayed_after(
    path: Path, tmp_path: Path, change: Callable[[Exchange, dict[str, Any]], bool]
) -> set[str]:
    """Replay *path* after *change* edited the stored replies; return the steps that went red."""
    setup, recorded, exchanges, _doc = load_fixture(path)
    changed = 0
    for exchange, body in _chat_bodies(exchanges):
        if change(exchange, body):
            exchange.body = json.dumps(body)
            changed += 1
    assert changed, "the change found no reply to edit"
    with replaying(exchanges) as url, configured(url, setup, tmp_path):
        replayed = run_steps(setup)
    return {step for step in recorded if not replayed[step].ok}


def _rename_tool(_exchange: Exchange, body: dict[str, Any]) -> bool:
    calls = body["choices"][0]["message"].get("tool_calls")
    if calls:
        calls[0]["function"]["name"] = "get_forecast"
    return bool(calls)


def _leak_think_tag(exchange: Exchange, body: dict[str, Any]) -> bool:
    message = body["choices"][0]["message"]
    if "17 plus 25" in json.dumps(exchange.request) and (
        message.get("reasoning_content") or message.get("reasoning")
    ):
        message["content"] = "<think>leak</think>" + str(message.get("content") or "")
        return True
    return False


_CHAT_FILES = [
    path for path in _FILES if "chat" in json.loads(path.read_text(encoding="utf-8"))["roles"]
]


@pytest.mark.parametrize("path", _CHAT_FILES, ids=lambda path: path.stem)
def test_a_changed_tool_name_turns_the_tool_step_red(path: Path, tmp_path: Path) -> None:
    assert _CHAT_FILES
    assert _replayed_after(path, tmp_path, _rename_tool) == {"tool"}


@pytest.mark.parametrize("path", _CHAT_FILES, ids=lambda path: path.stem)
def test_a_leaked_think_tag_turns_the_reasoning_step_red(path: Path, tmp_path: Path) -> None:
    assert _replayed_after(path, tmp_path, _leak_think_tag) == {"reasoning"}


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.stem)
def test_another_engine_name_turns_the_named_step_red(path: Path, tmp_path: Path) -> None:
    setup, recorded, exchanges, _doc = load_fixture(path)
    other = replace(setup, server="another-engine")
    with replaying(exchanges) as url, configured(url, other, tmp_path):
        replayed = run_steps(other)
    assert replayed["named"] != recorded["named"]
    assert "engine=another-engine" in replayed["named"].detail
