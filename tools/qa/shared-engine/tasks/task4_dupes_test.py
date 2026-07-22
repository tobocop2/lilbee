import inspect

import task4_dupes_impl as impl
from task4_dupes_impl import render_admin_row, render_user_row


def test_user_row():
    assert render_user_row("  ada lovelace ", 36) == "| Ada Lovelace (36)              |"


def test_admin_row():
    assert render_admin_row("grace", 45) == "| Grace (45) [admin]             |"


def test_blank_name_falls_back():
    assert render_user_row("   ", 1) == "| Unknown (1)                    |"


def test_duplication_removed():
    # The refactor must extract the shared logic: at least three functions
    # (the two renderers plus a shared helper) after deduplication.
    functions = [n for n, f in inspect.getmembers(impl, inspect.isfunction)]
    assert len(functions) >= 3, "extract the duplicated logic into a shared helper"
