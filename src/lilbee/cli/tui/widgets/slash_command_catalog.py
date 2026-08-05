"""Modal listing every slash command, grouped and filterable; reads ``COMMANDS``."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import COMMANDS, SlashCommand
from lilbee.cli.tui.widgets.clamped_option_list import ClampedOptionList


@dataclass(frozen=True)
class CatalogGroup:
    """A named group of slash commands, ordered for display."""

    title: str
    members: tuple[str, ...]


# Visual layout constants for ``_render_row``: align the command name +
# args column at this width, with at least this much gutter before the
# help text starts. Picked to fit the longest /set <key> <value> entry.
# Help text wraps at the row width with a hanging indent so wrapped
# lines stay in the description column; the fallback width matches the
# option area of the default 70-col modal before layout has run.
_ROW_NAME_COLUMN_WIDTH = 28
_ROW_HELP_GUTTER_MIN = 2
_ROW_HELP_MIN_WIDTH = 16
_ROW_FALLBACK_WIDTH = 64
# Horizontal padding the .option-list--option rule adds around each row.
_ROW_OPTION_PADDING = 2


CATALOG_GROUPS: tuple[CatalogGroup, ...] = (
    CatalogGroup(
        "CHAT & SESSION",
        ("/sessions", "/clear", "/cancel", "/quit", "/help", "/status"),
    ),
    CatalogGroup(
        "MODELS",
        ("/model", "/models", "/setup"),
    ),
    CatalogGroup(
        "KNOWLEDGE",
        ("/add", "/crawl", "/wiki", "/delete", "/rebuild", "/export", "/import"),
    ),
    CatalogGroup(
        "MEMORY",
        ("/remember", "/memories"),
    ),
    CatalogGroup(
        "SETTINGS & SYSTEM",
        ("/settings", "/set", "/theme", "/reset", "/remove", "/login", "/version"),
    ),
)


def _by_name() -> dict[str, SlashCommand]:
    return {cmd.name: cmd for cmd in COMMANDS}


def _matches(cmd: SlashCommand, query: str) -> bool:
    if not query:
        return True
    needle = query.lower().lstrip("/")
    if needle in cmd.name.lower():
        return True
    if any(needle in alias.lower() for alias in cmd.aliases):
        return True
    return needle in cmd.help_text.lower()


class SlashCommandCatalog(ModalScreen[str | None]):
    """Modal browser for every slash command; dismisses with the picked name or ``None``."""

    CSS_PATH = "slash_command_catalog.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("enter", "select", "Run", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="catalog-root"):
            yield Static(msg.SLASH_CATALOG_TITLE, id="catalog-title")
            yield Input(placeholder=msg.SLASH_CATALOG_FILTER_PLACEHOLDER, id="catalog-filter")
            yield ClampedOptionList(id="catalog-list")
            yield Static(msg.SLASH_CATALOG_FOOTER_HINT, id="catalog-hint")

    def on_mount(self) -> None:
        self._rebuild("")
        self.query_one("#catalog-filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "catalog-filter":
            return
        self._rebuild(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "catalog-filter":
            return
        self._select_first_match()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id and option_id.startswith("/"):
            self.dismiss(option_id)

    def action_select(self) -> None:
        ol = self.query_one("#catalog-list", OptionList)
        index = ol.highlighted
        if index is None:
            self._select_first_match()
            return
        try:
            opt = ol.get_option_at_index(index)
        except IndexError:
            return
        if opt.id and opt.id.startswith("/"):
            self.dismiss(opt.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _select_first_match(self) -> None:
        """Dismiss with the first runnable command in the current filtered list."""
        ol = self.query_one("#catalog-list", OptionList)
        for i in range(ol.option_count):
            opt = ol.get_option_at_index(i)
            if opt.id and opt.id.startswith("/"):
                self.dismiss(opt.id)
                return

    def on_resize(self) -> None:
        """Re-render rows so help wrapping tracks the option list's width."""
        self._rebuild(self.query_one("#catalog-filter", Input).value)

    def _rebuild(self, query: str) -> None:
        ol = self.query_one("#catalog-list", OptionList)
        ol.clear_options()
        groups = _filter_groups(query)
        if not groups:
            ol.add_option(Option(msg.SLASH_CATALOG_NO_MATCH, id=None, disabled=True))
            return
        first_runnable = _populate_options(ol, groups, _row_width(ol))
        if first_runnable is not None:
            ol.highlighted = first_runnable


def _row_width(ol: OptionList) -> int:
    """Usable text width of one option row, before layout the fallback width."""
    width = ol.scrollable_content_region.width - _ROW_OPTION_PADDING
    return width if width > 0 else _ROW_FALLBACK_WIDTH


def _filter_groups(query: str) -> list[tuple[str, list[SlashCommand]]]:
    """Each ``CatalogGroup`` paired with its filtered (non-empty) command list."""
    registry = _by_name()
    out: list[tuple[str, list[SlashCommand]]] = []
    for group in CATALOG_GROUPS:
        matching = [
            cmd
            for name in group.members
            if (cmd := registry.get(name)) is not None and _matches(cmd, query)
        ]
        if matching:
            out.append((group.title, matching))
    return out


def _populate_options(
    ol: OptionList, groups: list[tuple[str, list[SlashCommand]]], width: int
) -> int | None:
    """Add header + command rows for each group, return the first runnable row index."""
    first_runnable: int | None = None
    for title, commands in groups:
        ol.add_option(Option(_render_header(title), id=None, disabled=True))
        for cmd in commands:
            if first_runnable is None:
                first_runnable = ol.option_count
            ol.add_option(Option(_render_row(cmd, width), id=cmd.name))
    return first_runnable


def _render_header(title: str) -> Content:
    return Content.styled(title, "bold $primary")


def _render_row(cmd: SlashCommand, width: int) -> Content:
    """One row at *width* cols: name + args column, help with a hanging indent."""
    lead = f"  {cmd.name}"
    args = f" {cmd.args_hint}" if cmd.args_hint else ""
    help_col = max(_ROW_NAME_COLUMN_WIDTH, len(lead) + len(args) + _ROW_HELP_GUTTER_MIN)
    help_width = max(_ROW_HELP_MIN_WIDTH, width - help_col)
    wrapped = textwrap.wrap(cmd.help_text, help_width) or [""]
    first_pad = " " * (help_col - len(lead) - len(args))
    help_block = first_pad + ("\n" + " " * help_col).join(wrapped)
    name_part = Content.styled(lead, "$success bold")
    args_part = Content.styled(args, "$text-muted") if args else Content("")
    help_part = Content.styled(help_block, "$text-muted")
    return Content.assemble(name_part, args_part, help_part)
