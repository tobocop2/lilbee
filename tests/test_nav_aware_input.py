"""Tests for NavAwareInput: [ and ] bubble up, other printables stay captured."""

from __future__ import annotations

import ast
from pathlib import Path

from lilbee.cli.tui.widgets.nav_aware_input import NavAwareInput


def test_check_consume_key_bubbles_nav_bindings():
    """NavAwareInput must not claim [ or ] so they reach screen bindings."""
    widget = NavAwareInput()
    assert widget.check_consume_key("left_square_bracket", "[") is False
    assert widget.check_consume_key("right_square_bracket", "]") is False


def test_check_consume_key_still_captures_printable_chars():
    """Every other printable character must still be captured as text."""
    widget = NavAwareInput()
    for key, char in [
        ("a", "a"),
        ("A", "A"),
        ("space", " "),
        ("slash", "/"),
        ("full_stop", "."),
    ]:
        assert widget.check_consume_key(key, char) is True, (
            f"Expected NavAwareInput to consume {key!r}"
        )


def test_check_consume_key_ignores_non_printable_keys():
    """Non-printable keys (no character) must not be captured."""
    widget = NavAwareInput()
    assert widget.check_consume_key("tab", None) is False
    assert widget.check_consume_key("escape", None) is False


def _is_bare_input_call(node: ast.AST) -> bool:
    """True if ``node`` is a call whose callee resolves to the name ``Input``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Input"
    if isinstance(func, ast.Attribute):
        return func.attr == "Input"
    return False


def test_tui_screens_do_not_reintroduce_bare_input():
    """TUI screens must use NavAwareInput, not the base Input class."""
    screens_dir = Path(__file__).parent.parent / "src" / "lilbee" / "cli" / "tui" / "screens"
    offenders: list[str] = []
    for path in sorted(screens_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _is_bare_input_call(node):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "TUI screens must instantiate NavAwareInput, not Input. "
        "Bare `Input(...)` calls found at: " + ", ".join(offenders)
    )
