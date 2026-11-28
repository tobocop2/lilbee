"""The TUI /sync and /add commands take --ocr and --ocr-timeout and enter the OCR override scope."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lilbee.cli.tui.command_registry import COMMANDS
from lilbee.cli.tui.screens.chat import _split_ocr_flags
from lilbee.cli.tui.widgets.slash_command_catalog import CATALOG_GROUPS
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.data.ingest import SyncResult
from tests._async_wait import wait_until
from tests.test_tui_screens import ChatTestApp

DEFAULT_TIMEOUT = 300.0

# (ocr mode, ocr timeout, force_rebuild) as the extractor reads them at the sync call
Seen = tuple[str, float, bool]


def _recording_sync(rec: list[Seen]) -> Callable[..., Any]:
    """Stand in for the ingest sync; record the OCR settings the extractor would read."""

    async def fake_sync(**kwargs: Any) -> SyncResult:
        from lilbee.data.extract.document import _effective_ocr_mode, _effective_ocr_timeout

        rec.append(
            (
                str(_effective_ocr_mode()),
                _effective_ocr_timeout(),
                bool(kwargs.get("force_rebuild")),
            )
        )
        return SyncResult()

    return fake_sync


def _pin_cfg() -> None:
    cfg.ocr = OcrMode.AUTO
    cfg.vision_model = ""
    cfg.ocr_timeout = DEFAULT_TIMEOUT


async def _drive(
    tmp_path: Path,
    text: str,
    *,
    names: tuple[str, ...] = ("scan.pdf",),
    duplicate: bool = False,
    pause_budget: int = 60,
    then: str | None = None,
) -> tuple[list[Seen], list[str]]:
    """Type *text* into the chat screen; return (sync records, toast texts) once the work ends."""
    from lilbee.app.ingest import RegisterResult

    _pin_cfg()
    for name in names:
        (tmp_path / name).write_bytes(b"fake")
    rec: list[Seen] = []
    toasts: list[str] = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen

        def accept(dialog: Any, callback: Callable[[Any], None] | None = None) -> None:
            if callback is not None:
                callback(True)

        with ExitStack() as stack:
            stack.enter_context(patch("lilbee.data.ingest.sync", new=_recording_sync(rec)))
            stack.enter_context(
                patch(
                    "lilbee.app.ingest.register_sources",
                    return_value=RegisterResult(registered=list(names)),
                )
            )
            stack.enter_context(
                patch.object(screen, "notify", side_effect=lambda m, **k: toasts.append(str(m)))
            )
            stack.enter_context(patch.object(screen._task_bar, "start_detect_pending"))
            if duplicate:
                stack.enter_context(
                    patch("lilbee.app.ingest.source_label_taken", return_value=True)
                )
                stack.enter_context(patch.object(app, "push_screen", side_effect=accept))
            screen._handle_slash(text)
            await wait_until(pilot, lambda: bool(rec), max_pauses=pause_budget, timeout=3.0)
            if then is not None:
                await wait_until(
                    pilot,
                    lambda: not screen._sync_work_live(),
                    max_pauses=pause_budget,
                    timeout=3.0,
                )
                screen._handle_slash(then)
                await wait_until(pilot, lambda: len(rec) >= 2, max_pauses=pause_budget, timeout=3.0)
            await pilot.pause()
    return rec, toasts


def _named(toasts: list[str], word: str) -> bool:
    return any(word in text for text in toasts)


async def test_chat_slash_sync_runs_one_sync(tmp_path: Path) -> None:
    rec, _ = await _drive(tmp_path, "/sync")
    assert rec == [("auto", DEFAULT_TIMEOUT, False)]


@pytest.mark.parametrize("mode", ["all", "off"])
async def test_chat_slash_sync_ocr_mode_reaches_the_sync(tmp_path: Path, mode: str) -> None:
    rec, _ = await _drive(tmp_path, f"/sync --ocr {mode}")
    assert rec == [(mode, DEFAULT_TIMEOUT, False)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/sync --ocr-timeout 90", ("auto", 90.0, False)),
        ("/sync --ocr-timeout 0", ("auto", 0.0, False)),
        ("/sync --ocr all --ocr-timeout 12", ("all", 12.0, False)),
    ],
)
async def test_chat_slash_sync_ocr_timeout_reaches_the_sync(
    tmp_path: Path, text: str, expected: Seen
) -> None:
    rec, _ = await _drive(tmp_path, text)
    assert rec == [expected]


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("/sync --ocr bogus", "bogus"),
        ("/sync --ocr-timeout -5", "-5"),
        ("/sync --ocr-timeout soon", "soon"),
        ("/sync --nope", "--nope"),
        ("/sync --ocr", "--ocr"),
        ("/sync stray words", "stray words"),
    ],
)
async def test_chat_slash_sync_refuses_bad_option_by_name(
    tmp_path: Path, text: str, named: str
) -> None:
    rec, toasts = await _drive(tmp_path, text, pause_budget=15)
    assert rec == []
    assert _named(toasts, named)


async def test_chat_slash_sync_option_does_not_persist(tmp_path: Path) -> None:
    rec, _ = await _drive(tmp_path, "/sync --ocr all", then="/sync")
    assert rec == [("all", DEFAULT_TIMEOUT, False), ("auto", DEFAULT_TIMEOUT, False)]
    assert cfg.ocr == OcrMode.AUTO


async def test_chat_slash_add_option_does_not_persist(tmp_path: Path) -> None:
    rec, _ = await _drive(tmp_path, f"/add {tmp_path / 'scan.pdf'}", then="/sync --ocr all")
    assert rec == [("auto", DEFAULT_TIMEOUT, False), ("all", DEFAULT_TIMEOUT, False)]


async def test_chat_slash_sync_refused_while_sync_runs() -> None:
    from lilbee.cli.tui.task_queue import TaskType

    _pin_cfg()
    rec: list[Seen] = []
    toasts: list[str] = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        app.task_bar.add_task("Sync", TaskType.SYNC)
        with (
            patch("lilbee.data.ingest.sync", new=_recording_sync(rec)),
            patch.object(screen, "notify", side_effect=lambda m, **k: toasts.append(str(m))),
        ):
            screen._handle_slash("/sync --ocr all")
            await pilot.pause()
    assert rec == []
    assert _named(toasts, "Sync in progress")


async def test_chat_slash_add_without_option_reads_the_setting(tmp_path: Path) -> None:
    rec, _ = await _drive(tmp_path, f"/add {tmp_path / 'scan.pdf'}")
    assert rec == [("auto", DEFAULT_TIMEOUT, False)]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ("--ocr all", ("all", DEFAULT_TIMEOUT, False)),
        ("--ocr off --ocr-timeout 33", ("off", 33.0, False)),
    ],
)
async def test_chat_slash_add_ocr_option_reaches_the_sync(
    tmp_path: Path, flags: str, expected: Seen
) -> None:
    rec, _ = await _drive(tmp_path, f"/add {flags} {tmp_path / 'scan.pdf'}")
    assert rec == [expected]


async def test_chat_slash_add_ocr_option_keeps_a_path_with_spaces(tmp_path: Path) -> None:
    name = "my scan (v2).pdf"
    rec, toasts = await _drive(tmp_path, f"/add --ocr all {tmp_path / name}", names=(name,))
    assert rec == [("all", DEFAULT_TIMEOUT, False)]
    assert not _named(toasts, "Not found")


async def test_chat_slash_add_ocr_option_with_several_paths(tmp_path: Path) -> None:
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    rec, _ = await _drive(
        tmp_path, f'/add --ocr all "{first}" "{second}"', names=("a.pdf", "b.pdf")
    )
    assert rec == [("all", DEFAULT_TIMEOUT, False)]


async def test_chat_slash_add_ocr_option_survives_the_overwrite_confirmation(
    tmp_path: Path,
) -> None:
    rec, _ = await _drive(tmp_path, f"/add --ocr all {tmp_path / 'scan.pdf'}", duplicate=True)
    assert rec == [("all", DEFAULT_TIMEOUT, False)]


async def test_chat_slash_add_refuses_bad_ocr_option_and_registers_nothing(tmp_path: Path) -> None:
    rec, toasts = await _drive(
        tmp_path, f"/add --ocr bogus {tmp_path / 'scan.pdf'}", pause_budget=15
    )
    assert rec == []
    assert _named(toasts, "bogus")
    assert not _named(toasts, "Not found")


async def test_chat_slash_add_option_without_a_path(tmp_path: Path) -> None:
    rec, toasts = await _drive(tmp_path, "/add --ocr all", pause_budget=15)
    assert rec == []
    assert _named(toasts, "Usage: /add")
    assert not _named(toasts, "Not found")


async def test_chat_slash_add_refuses_ocr_option_with_a_url(tmp_path: Path) -> None:
    rec, toasts = await _drive(tmp_path, "/add --ocr all https://example.com/a", pause_budget=10)
    assert rec == []
    assert _named(toasts, "--ocr")
    assert not _named(toasts, "Not found")


async def test_chat_slash_rebuild_keeps_the_setting() -> None:
    _pin_cfg()
    rec: list[Seen] = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen

        def accept(dialog: Any, callback: Callable[[Any], None] | None = None) -> None:
            if callback is not None:
                callback(True)

        with (
            patch("lilbee.data.ingest.sync", new=_recording_sync(rec)),
            patch.object(app, "push_screen", side_effect=accept),
            patch.object(screen._task_bar, "start_detect_pending"),
        ):
            screen._handle_slash("/rebuild")
            await wait_until(pilot, lambda: bool(rec), max_pauses=60, timeout=3.0)
    assert rec == [("auto", DEFAULT_TIMEOUT, True)]


async def test_run_sync_without_options_reads_the_setting() -> None:
    _pin_cfg()
    rec: list[Seen] = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        with (
            patch("lilbee.data.ingest.sync", new=_recording_sync(rec)),
            patch.object(screen._task_bar, "start_detect_pending"),
        ):
            screen.run_sync()
            await wait_until(pilot, lambda: bool(rec), max_pauses=60, timeout=3.0)
    assert rec == [("auto", DEFAULT_TIMEOUT, False)]


def test_sync_is_registered_and_listed() -> None:
    assert any(command.name == "/sync" for command in COMMANDS)
    group = next(g for g in CATALOG_GROUPS if g.title == "KNOWLEDGE")
    assert "/sync" in group.members


def test_add_and_sync_hints_name_the_ocr_options() -> None:
    by_name = {command.name: command for command in COMMANDS}
    for name in ("/add", "/sync"):
        assert "--ocr " in by_name[name].args_hint
        assert "--ocr-timeout" in by_name[name].args_hint


def test_split_ocr_flags_leaves_a_path_with_dashes_alone() -> None:
    assert _split_ocr_flags("--ocr all /tmp/a--b c.pdf") == (OcrMode.ALL, None, "/tmp/a--b c.pdf")


def test_split_ocr_flags_accepts_the_options_in_either_order() -> None:
    expected = (OcrMode.OFF, 9.0, "/tmp/a.pdf")
    assert _split_ocr_flags("--ocr off --ocr-timeout 9 /tmp/a.pdf") == expected
    assert _split_ocr_flags("--ocr-timeout 9 --ocr off /tmp/a.pdf") == expected


def test_split_ocr_flags_takes_the_last_of_a_repeated_option() -> None:
    assert _split_ocr_flags("--ocr all --ocr off x.pdf") == (OcrMode.OFF, None, "x.pdf")


@pytest.mark.parametrize("flag", ["--ocrx", "--ocr=all", "--ocr-timeouts"])
def test_split_ocr_flags_refuses_an_option_that_only_starts_like_ours(flag: str) -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        _split_ocr_flags(f"{flag} all x.pdf")


def test_catalog_row_puts_the_help_below_a_long_args_hint() -> None:
    from lilbee.cli.tui.command_registry import get_command
    from lilbee.cli.tui.widgets.slash_command_catalog import _render_row

    lines = _render_row(get_command("/add"), 62).plain.splitlines()
    assert len(lines) > 1
    assert "<path>" in lines[0]
    assert "Add file" not in lines[0]
    assert all(len(line) <= 62 for line in lines)
    assert "".join(line.strip() for line in lines[1:]).startswith("Add file")
