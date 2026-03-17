"""Chat REPL loop, model initialization, and toolbar."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from rich.console import Console

from lilbee.cli import theme
from lilbee.cli.chat.complete import make_completer
from lilbee.cli.chat.slash import QuitChat, dispatch_slash
from lilbee.cli.chat.stream import stream_response
from lilbee.cli.chat.sync import SyncStatus, run_sync_background, shutdown_executor

if TYPE_CHECKING:
    from lilbee.query import ChatMessage


def sync_toolbar(status: SyncStatus) -> list[tuple[str, str]] | str:
    """Return prompt_toolkit bottom toolbar content for sync status."""
    if status.text:
        return [("class:bottom-toolbar", status.text)]
    return ""


def _init_models(con: Console) -> None:
    """Validate chat and embedding models, with a spinner on first run."""
    from lilbee.embedder import validate_model
    from lilbee.models import ensure_chat_model

    with con.status("Initializing..."):
        ensure_chat_model()
        validate_model()


def chat_loop(con: Console, *, auto_sync_bg: bool = False) -> None:
    """Interactive REPL with slash-command support."""
    _init_models(con)
    con.print(f"[{theme.LABEL}]lilbee chat[/{theme.LABEL}] — type /help for commands\n")
    history: list[ChatMessage] = []
    sync_status = SyncStatus()

    # When prompt_toolkit is active its StdoutProxy intercepts sys.stdout,
    # so Rich markup written via the global Console renders as raw ANSI.
    # chat_con bypasses StdoutProxy by writing to sys.__stdout__ directly.
    # Outside a TTY (tests / pipes) there's no StdoutProxy, so con is fine.
    chat_con: Console = con

    _prompt_fn: Callable[[], str] | None = None
    _patch_ctx: AbstractContextManager[object] | None = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout
            from prompt_toolkit.styles import Style

            session: PromptSession[str] = PromptSession(
                completer=make_completer(),
                bottom_toolbar=lambda: sync_toolbar(sync_status),
                refresh_interval=0.5,
                style=Style.from_dict(
                    {
                        "bottom-toolbar": f"fg:{theme.TOOLBAR_FG} bg:{theme.TOOLBAR_BG}",
                    }
                ),
            )
            _prompt_fn = lambda: session.prompt("> ")  # noqa: E731
            _patch_ctx = patch_stdout()
            chat_con = Console(file=sys.__stdout__)
        except ImportError:
            pass

    if _patch_ctx is not None:
        _patch_ctx.__enter__()

    if auto_sync_bg:
        run_sync_background(con, chat_mode=True, sync_status=sync_status)

    try:
        while True:
            try:
                if _prompt_fn is not None:
                    question = _prompt_fn()
                else:
                    question = con.input(f"[{theme.PROMPT}]> [/{theme.PROMPT}]")
            except (EOFError, KeyboardInterrupt):
                break
            if not question.strip():
                continue
            try:
                if dispatch_slash(question, chat_con, sync_status=sync_status):
                    continue
            except QuitChat:
                break
            stream_response(question, history, con, chat_mode=True, chat_console=chat_con)
    finally:
        shutdown_executor()
        if _patch_ctx is not None:
            _patch_ctx.__exit__(None, None, None)
