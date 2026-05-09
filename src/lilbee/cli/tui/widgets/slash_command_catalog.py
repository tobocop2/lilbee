"""Modal listing every slash command, grouped and filterable.

Opened by ``/help`` or by clicking the help-hint chip in the chat footer.
Reads commands from :mod:`lilbee.cli.tui.command_registry` so the list stays
in sync with the registry without duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from lilbee.cli.tui.command_registry import COMMANDS, SlashCommand


@dataclass(frozen=True)
class CatalogGroup:
    """A named group of slash commands, ordered for display."""

    title: str
    members: tuple[str, ...]


CATALOG_GROUPS: tuple[CatalogGroup, ...] = (
    CatalogGroup(
        "CHAT & SESSION",
        ("/clear", "/cancel", "/quit", "/help", "/status"),
    ),
    CatalogGroup(
        "MODELS",
        ("/model", "/models", "/setup"),
    ),
    CatalogGroup(
        "KNOWLEDGE",
        ("/add", "/crawl", "/wiki", "/delete"),
    ),
    CatalogGroup(
        "SETTINGS & SYSTEM",
        ("/settings", "/set", "/theme", "/reset", "/remove", "/login", "/version"),
    ),
)

CATALOG_TITLE = "Slash Commands"
CATALOG_FILTER_PLACEHOLDER = "Filter commands..."
CATALOG_FOOTER_HINT = "↑↓ select   Enter run   Esc close"
CATALOG_NO_MATCH = "No commands match"


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
    """Modal browser for every slash command in the registry.

    Dismisses with the chosen command name (e.g. ``"/wiki"``) or ``None`` if
    cancelled. The chat screen inserts ``"<name> "`` into the chat input so
    the existing autocomplete can take over for arguments.
    """

    CSS_PATH = "slash_command_catalog.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("enter", "select", "Run", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="catalog-root"):
            yield Static(CATALOG_TITLE, id="catalog-title")
            yield Input(placeholder=CATALOG_FILTER_PLACEHOLDER, id="catalog-filter")
            yield OptionList(id="catalog-list")
            yield Static(CATALOG_FOOTER_HINT, id="catalog-hint")

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

    def _rebuild(self, query: str) -> None:
        ol = self.query_one("#catalog-list", OptionList)
        ol.clear_options()
        registry = _by_name()
        any_added = False
        first_runnable_index: int | None = None

        for group in CATALOG_GROUPS:
            visible: list[SlashCommand] = []
            for name in group.members:
                cmd = registry.get(name)
                if cmd is None:
                    continue
                if _matches(cmd, query):
                    visible.append(cmd)
            if not visible:
                continue
            ol.add_option(Option(_render_header(group.title), id=None, disabled=True))
            for cmd in visible:
                if first_runnable_index is None:
                    first_runnable_index = ol.option_count
                ol.add_option(Option(_render_row(cmd), id=cmd.name))
            any_added = True

        if not any_added:
            ol.add_option(Option(CATALOG_NO_MATCH, id=None, disabled=True))

        if first_runnable_index is not None:
            ol.highlighted = first_runnable_index


def _render_header(title: str) -> Content:
    return Content.styled(title, "bold $primary")


def _render_row(cmd: SlashCommand) -> Content:
    name_part = Content.styled(f"  {cmd.name}", "$success bold")
    args_part = Content.styled(f" {cmd.args_hint}", "$text-muted") if cmd.args_hint else Content("")
    pad_target = 28
    visible_len = len(f"  {cmd.name}") + (len(f" {cmd.args_hint}") if cmd.args_hint else 0)
    pad = " " * max(2, pad_target - visible_len)
    help_part = Content.styled(f"{pad}{cmd.help_text}", "$text-muted")
    return Content.assemble(name_part, args_part, help_part)
