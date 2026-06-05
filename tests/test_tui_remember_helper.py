"""Tests for the pure ``/remember`` orchestration helper.

The TUI worker body is a single call to :func:`remember_from_input`, so the
parse / gate / store logic is covered here without a running Textual app.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.memory import MEMORY_DISABLED_HINT
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.chat_helpers import RememberOutcome, remember_from_input
from lilbee.core.config import cfg
from lilbee.data.store import MemoryKind
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def mock_svc():
    store = MagicMock()
    store.add_memory.return_value = "id123"
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embedding_available.return_value = True
    services = make_mock_services(store=store, embedder=embedder)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture(autouse=True)
def _enabled():
    snapshot = cfg.memory_enabled
    cfg.memory_enabled = True
    yield
    cfg.memory_enabled = snapshot


def test_disabled_returns_hint():
    cfg.memory_enabled = False
    outcome = remember_from_input("uses rust")
    assert outcome == RememberOutcome(MEMORY_DISABLED_HINT, "warning")


def test_fact_is_stored(mock_svc):
    outcome = remember_from_input("uses rust")
    assert outcome.message == msg.CMD_REMEMBER_SUCCESS.format(kind="fact")
    assert outcome.severity == "information"
    record = mock_svc.store.add_memory.call_args.args[0]
    assert record.kind is MemoryKind.FACT
    assert record.text == "uses rust"


def test_preference_prefix_stored_as_preference(mock_svc):
    outcome = remember_from_input("pref: keep answers terse")
    assert outcome.message == msg.CMD_REMEMBER_SUCCESS.format(kind="preference")
    record = mock_svc.store.add_memory.call_args.args[0]
    assert record.kind is MemoryKind.PREFERENCE
    assert record.text == "keep answers terse"


def test_preference_prefix_is_case_insensitive(mock_svc):
    remember_from_input("PREF: shout less")
    record = mock_svc.store.add_memory.call_args.args[0]
    assert record.kind is MemoryKind.PREFERENCE
    assert record.text == "shout less"


def test_empty_text_returns_usage(mock_svc):
    outcome = remember_from_input("   ")
    assert outcome == RememberOutcome(msg.CMD_REMEMBER_USAGE, "warning")
    mock_svc.store.add_memory.assert_not_called()


def test_preference_with_no_body_returns_usage(mock_svc):
    outcome = remember_from_input("pref:   ")
    assert outcome == RememberOutcome(msg.CMD_REMEMBER_USAGE, "warning")
    mock_svc.store.add_memory.assert_not_called()


def test_missing_embedder_returns_hint(mock_svc):
    mock_svc.embedder.embedding_available.return_value = False
    outcome = remember_from_input("uses rust")
    assert outcome == RememberOutcome(msg.CMD_REMEMBER_NO_EMBED, "warning")
    mock_svc.store.add_memory.assert_not_called()
