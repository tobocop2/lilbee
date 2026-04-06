"""Autocomplete dropdown overlay for the chat input."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.cli.tui.app import DARK_THEMES
from lilbee.cli.tui.command_registry import command_names

log = logging.getLogger(__name__)

_SLASH_COMMANDS = command_names()
_MAX_VISIBLE = 8  # max dropdown items shown at once


def get_completions(text: str) -> list[str]:
    """Return completion options for the current input text."""
    if not text.startswith("/"):
        return []

    if " " not in text:
        return [c for c in _SLASH_COMMANDS if c.startswith(text) and c != text]

    cmd, _, partial = text.partition(" ")
    cmd = cmd.lower()
    return _get_arg_completions(cmd, partial)


def _get_arg_completions(cmd: str, partial: str) -> list[str]:
    """Get argument completions for a specific command."""
    sources = _ARG_SOURCES.get(cmd)
    if sources is None:
        return []
    if cmd == "/add":
        return _path_options(partial)
    options = sources()
    if partial:
        return [o for o in options if o.lower().startswith(partial.lower())]
    return options


def _model_options() -> list[str]:
    try:
        from lilbee.models import list_installed_models

        return list_installed_models()
    except Exception:
        log.debug("Failed to list models for autocomplete", exc_info=True)
        return []


def _vision_options() -> list[str]:
    names = ["off"]
    try:
        from lilbee.models import VISION_CATALOG

        names.extend(m.name for m in VISION_CATALOG)
    except Exception:
        log.debug("Failed to load vision catalog for autocomplete", exc_info=True)
    return names


def _setting_options() -> list[str]:
    return list(SETTINGS_MAP.keys())


def _document_options() -> list[str]:
    try:
        from lilbee.services import get_services

        return [s.get("filename", s.get("source", "")) for s in get_services().store.get_sources()]
    except Exception:
        log.debug("Failed to list documents for autocomplete", exc_info=True)
        return []


def _theme_options() -> list[str]:
    return list(DARK_THEMES)


def _path_options(partial: str = "") -> list[str]:
    """Return filesystem completions for a partial path.

    Handles relative paths, absolute paths, and ~ expansion.
    Directories get a trailing / so the user knows to keep typing.
    """
    try:
        expanded = Path(partial).expanduser() if partial else Path(".")
        if partial and not expanded.is_dir():
            parent = expanded.parent
            prefix = expanded.name.lower()
        else:
            parent = expanded
            prefix = ""

        if not parent.is_dir():
            return []

        results: list[str] = []
        for p in sorted(parent.iterdir()):
            if p.name.startswith("."):
                continue
            if prefix and not p.name.lower().startswith(prefix):
                continue
            display = str(p) if partial and Path(partial) != Path(".") else p.name
            if p.is_dir():
                display = display.rstrip("/") + "/"
            results.append(display)
            if len(results) >= 20:
                break
        return results
    except Exception:
        log.debug("Failed to list paths for autocomplete", exc_info=True)
        return []


_ARG_SOURCES: dict[str, Callable[[], list[str]]] = {
    "/model": _model_options,
    "/vision": _vision_options,
    "/set": _setting_options,
    "/delete": _document_options,
    "/remove": _model_options,
    "/theme": _theme_options,
    "/add": _path_options,
}


class CompletionOverlay(Vertical):
    """Dropdown overlay showing completion options above the input."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss_overlay", show=False),
    ]

    DEFAULT_CSS = """
    CompletionOverlay {
        dock: bottom;
        height: auto;
        max-height: 10;
        layer: overlay;
        offset-y: -3;
        background: $surface;
        border: tall $primary;
        display: none;
    }
    CompletionOverlay OptionList {
        height: auto;
        max-height: 8;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._options: list[str] = []
        self._index = 0

    def compose(self) -> ComposeResult:
        yield OptionList(id="completion-list")

    def show_completions(self, options: list[str]) -> None:
        """Populate and show the overlay."""
        self._options = options[:_MAX_VISIBLE]
        self._index = 0
        ol = self.query_one("#completion-list", OptionList)
        ol.clear_options()
        for opt in self._options:
            ol.add_option(Option(opt))
        if self._options:
            ol.highlighted = 0
            self.display = True
        else:
            self.display = False

    def cycle_next(self) -> str | None:
        """Cycle to next option and return it."""
        if not self._options:
            return None
        self._index = (self._index + 1) % len(self._options)
        ol = self.query_one("#completion-list", OptionList)
        ol.highlighted = self._index
        return self._options[self._index]

    def cycle_prev(self) -> str | None:
        """Cycle to previous option and return it."""
        if not self._options:
            return None
        self._index = (self._index - 1) % len(self._options)
        ol = self.query_one("#completion-list", OptionList)
        ol.highlighted = self._index
        return self._options[self._index]

    def get_current(self) -> str | None:
        """Get the currently highlighted option."""
        if not self._options or self._index >= len(self._options):
            return None
        return self._options[self._index]

    def hide(self) -> None:
        """Hide the overlay."""
        self.display = False
        self._options = []

    @property
    def is_visible(self) -> bool:
        return bool(self.display) and bool(self._options)

    def action_dismiss_overlay(self) -> None:
        self.hide()
