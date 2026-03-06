#!/usr/bin/env python3
"""Record lilbee demo GIF using asciinema + pexpect + agg.

Drives a real ``lilbee`` interactive session through pexpect so that
prompt_toolkit renders correctly, records via asciinema, converts to GIF with agg.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pexpect

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
COLS = 120
ROWS = 36
TYPING_DELAY = 0.03  # seconds between keystrokes (visual effect)
SLASH_CMD_PAUSE = 2.0  # pause after slash commands
POST_ANSWER_PAUSE = 1.0  # pause after each LLM answer
CAST_FILE = "demo.cast"
GIF_FILE = "demo.gif"

# Questions to ask in the demo
QUESTIONS = [
    "What size engine does my Crown Victoria have?",
    "What is the headlight part number for this car?",
]

# Shell prompt pattern (matches typical bash/zsh prompts)
SHELL_PROMPT = r"[\$#%] "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _type_slowly(child: pexpect.spawn, text: str) -> None:  # type: ignore[type-arg]
    """Send *text* one character at a time with a small delay."""
    for ch in text:
        child.send(ch)
        time.sleep(TYPING_DELAY)
    time.sleep(0.1)
    child.sendline("")


def _wait_for_shell(child: pexpect.spawn, timeout: int = 15) -> None:  # type: ignore[type-arg]
    """Wait until we see a shell prompt."""
    child.expect(SHELL_PROMPT, timeout=timeout)


def _drain_buffer(child: pexpect.spawn) -> None:  # type: ignore[type-arg]
    """Consume any pending output so the buffer is clean for the next expect."""
    try:
        while True:
            child.read_nonblocking(size=4096, timeout=0.2)
    except (pexpect.TIMEOUT, pexpect.EOF):
        pass


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_demo(cast_file: str = CAST_FILE) -> None:
    """Drive lilbee through pexpect under asciinema recording."""
    env = os.environ.copy()
    env["COLUMNS"] = str(COLS)
    env["LINES"] = str(ROWS)
    env["TERM"] = "xterm-256color"

    child = pexpect.spawn(
        "asciinema",
        [
            "rec",
            "--overwrite",
            "--output-format",
            "asciicast-v2",
            "--window-size",
            f"{COLS}x{ROWS}",
            cast_file,
        ],
        encoding="utf-8",
        timeout=120,
        env=env,
    )
    child.setwinsize(ROWS, COLS)

    # Wait for the shell prompt inside asciinema
    _wait_for_shell(child)
    time.sleep(0.3)

    # Clean prompt and suppress Rich CPR warning
    child.sendline("export PS1='$ ' COLUMNS=120")
    _wait_for_shell(child)
    child.sendline("clear")
    time.sleep(0.5)
    _drain_buffer(child)

    # --- Enter interactive mode (default command) ---
    _type_slowly(child, "lilbee")
    child.expect(r"lilbee chat", timeout=30)
    time.sleep(1.5)
    _drain_buffer(child)

    # --- /help ---
    child.sendline("/help")
    time.sleep(SLASH_CMD_PAUSE)

    # --- /status ---
    child.sendline("/status")
    time.sleep(SLASH_CMD_PAUSE)

    # --- Questions ---
    for question in QUESTIONS:
        _type_slowly(child, question)
        child.expect(r"Sources:", timeout=120)
        time.sleep(POST_ANSWER_PAUSE)
        _drain_buffer(child)

    # --- Exit ---
    child.sendline("/quit")
    time.sleep(1.0)
    _drain_buffer(child)

    # Exit the asciinema shell
    child.sendline("exit")
    time.sleep(0.5)
    child.sendeof()
    child.expect(pexpect.EOF, timeout=15)

    print(f"Recording saved to {cast_file}")


# ---------------------------------------------------------------------------
# GIF conversion
# ---------------------------------------------------------------------------


def convert_to_gif(cast_file: str = CAST_FILE, gif_file: str = GIF_FILE) -> None:
    """Convert asciinema cast to GIF using agg."""
    subprocess.run(
        [
            "agg",
            "--theme",
            "monokai",
            "--font-size",
            "14",
            "--speed",
            "2",
            "--idle-time-limit",
            "2",
            cast_file,
            gif_file,
        ],
        check=True,
    )
    print(f"GIF saved to {gif_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cast = CAST_FILE
    gif = GIF_FILE
    if "--cast-only" in sys.argv:
        record_demo(cast)
    elif "--gif-only" in sys.argv:
        convert_to_gif(cast, gif)
    else:
        record_demo(cast)
        convert_to_gif(cast, gif)
