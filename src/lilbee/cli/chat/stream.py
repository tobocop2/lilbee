"""LLM response streaming for chat mode."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console

from lilbee.cli import theme

if TYPE_CHECKING:
    from lilbee.query import ChatMessage


def stream_response(
    question: str,
    history: list[ChatMessage],
    con: Console,
    *,
    chat_mode: bool = False,
    chat_console: Console | None = None,
) -> None:
    """Stream an LLM answer and append the exchange to *history*.

    When *chat_console* is provided (a Console wired to ``sys.__stdout__``),
    it is used for the thinking spinner instead of creating a one-off Console.
    """
    from lilbee.query import ask_stream

    stream = ask_stream(question, history=history)
    response_parts: list[str] = []
    cancelled = False

    # In chat mode, use chat_console (bypasses StdoutProxy) for all output.
    out_con = chat_console if chat_mode and chat_console else con

    try:
        if chat_mode:
            spinner_con = chat_console or Console(file=sys.__stdout__)
            ctx = spinner_con.status("Thinking...")
        else:
            ctx = con.status("Thinking...")

        with ctx:
            first_token = next(stream, None)

        if first_token is not None:
            out_con.print(first_token, end="")
            response_parts.append(first_token)

        for token in stream:
            out_con.print(token, end="")
            response_parts.append(token)
    except KeyboardInterrupt:
        cancelled = True
        stream.close()
        out_con.print(f"\n[{theme.MUTED}](stopped)[/{theme.MUTED}]")
    except RuntimeError as exc:
        out_con.print(f"\n[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        return

    if not cancelled:
        out_con.print("\n")
    full = "".join(response_parts)
    if full:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full})
