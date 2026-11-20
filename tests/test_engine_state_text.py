"""The words for the state and the download size of an engine, and the install retry notice."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption
from lilbee.cli.tui import engine_install_flow, engine_mark
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.engine_picker import state_text


def test_a_bundled_engine_reads_bundled() -> None:
    assert state_text(EngineOption("llama.cpp", bundled=True)) == msg.ENGINE_STATE_BUNDLED


def test_an_installed_engine_reads_installed() -> None:
    assert state_text(EngineOption("tei")) == msg.ENGINE_STATE_INSTALLED


def test_a_missing_engine_of_unknown_size_reads_not_installed() -> None:
    option = EngineOption("openvino", installed=False)
    assert state_text(option) == msg.ENGINE_STATE_NOT_INSTALLED


def test_a_missing_engine_shows_its_size() -> None:
    option = EngineOption("openvino", installed=False, download_bytes=180_000_000)
    assert state_text(option) == msg.ENGINE_STATE_DOWNLOAD.format(size="180 MB")


@pytest.mark.parametrize(
    ("size", "text"),
    [(1, "1 MB"), (180_000_000, "180 MB"), (1_200_000_000, "1.2 GB")],
)
def test_a_download_size_reads_as_one_short_figure(size: int, text: str) -> None:
    assert engine_mark.format_size(size) == text


def test_the_download_question_names_the_size_when_it_is_known() -> None:
    option = EngineOption("openvino", installed=False, download_bytes=2_000_000_000)
    assert "2.0 GB" in engine_install_flow.confirm_text(option)


def test_the_download_question_has_no_size_when_it_is_not_known() -> None:
    option = EngineOption("openvino", installed=False)
    text = engine_install_flow.confirm_text(option)
    assert text == msg.ENGINE_DOWNLOAD_MESSAGE_NO_SIZE.format(engine="openvino")


def test_a_retry_of_the_install_task_shows_as_a_warning_toast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = MagicMock()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        engine_install_flow, "call_from_thread", lambda _app, fn, *a, **k: calls.append((fn, a, k))
    )

    def fake_install(name: str, *, on_update: Any, on_retry: Any, cancel: Any) -> None:
        on_retry("try 2 of 3")

    monkeypatch.setattr(engine_service, "install", fake_install)
    engine_install_flow.install_then(app, "openvino", lambda: None)
    target = app.task_bar.start_task.call_args.args[2]
    target(MagicMock())
    assert calls == [(app.notify, ("try 2 of 3",), {"severity": "warning"})]
