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

from lilbee.app.services import get_services
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui.command_registry import completion_names

log = logging.getLogger(__name__)

_SLASH_COMMANDS = completion_names()
_MAX_VISIBLE = 8  # max dropdown items shown at once
# Hard cap on path completions surfaced for /add so a deep directory doesn't
# stall the dropdown rebuild.
_MAX_PATH_COMPLETIONS = 20

_CSS_FILE = Path(__file__).parent / "autocomplete.tcss"


# Cached document list for ``/delete`` and ``/reset`` Tab completion.
# Invalidated by ``invalidate_document_cache`` on document mutations so
# Tab returns the live set. Order is stable across reads because the
# dropdown renders in fetch order.
_doc_cache: list[str] | None = None


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
    """Get argument completions for a specific command.

    Drops the option that exactly equals what the user has typed so a
    fully-typed argument collapses the dropdown and lets Enter submit,
    mirroring the command-discovery rule for slash commands.
    """
    sources = _ARG_SOURCES.get(cmd)
    if sources is None:
        return []
    options = _path_options(partial) if cmd == "/add" else sources()
    if partial:
        low = partial.lower()
        options = [o for o in options if o.lower().startswith(low)]
    return [o for o in options if o.lower() != partial.lower()]


def _model_options() -> list[str]:
    try:
        from lilbee.modelhub.models import list_installed_models

        return list_installed_models()
    except Exception:
        log.debug("Failed to list models for autocomplete", exc_info=True)
        return []


def _setting_options() -> list[str]:
    return list(SETTINGS_MAP.keys())


def _document_options() -> list[str]:
    global _doc_cache
    if _doc_cache is not None:
        return _doc_cache
    try:
        _doc_cache = [
            s.get("filename", s.get("source", "")) for s in get_services().store.get_sources()
        ]
    except Exception:
        log.debug("Failed to list documents for autocomplete", exc_info=True)
        _doc_cache = []
    return _doc_cache


def invalidate_document_cache() -> None:
    """Drop the cached document list; the next Tab refetches from the store."""
    global _doc_cache
    _doc_cache = None


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
            if len(results) >= _MAX_PATH_COMPLETIONS:
                break
        return results
    except Exception:
        log.debug("Failed to list paths for autocomplete", exc_info=True)
        return []


_ARG_SOURCES: dict[str, Callable[[], list[str]]] = {
    "/model": _model_options,
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

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

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
