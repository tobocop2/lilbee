"""Virtualized model list backed by Textual's OptionList.

OptionList renders only on-screen rows, so frontier-tab populations of
hundreds of cloud models stay smooth. ``set_rows`` rebuilds the list
from a flat sequence of ``ModelListSection``; ``Selected`` is posted
when the user activates a non-heading row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

from textual import on
from textual.content import Content
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS
from lilbee.modelhub.models import FEATURED_STAR

_CSS_FILE = Path(__file__).parent / "model_list.tcss"


class ModelListSection(NamedTuple):
    """One contiguous block of rows under an optional heading."""

    heading: str | None
    rows: list[CatalogRow]


class ModelList(OptionList):
    """OptionList specialized for catalog rows, posting Selected on activate."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    @dataclass
    class Selected(Message):
        """Posted when a non-heading row is activated."""

        row: CatalogRow

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._row_by_option_id: dict[str, CatalogRow] = {}

    def set_rows(self, sections: list[ModelListSection]) -> None:
        """Replace the contents with options derived from *sections*."""
        self._row_by_option_id.clear()
        self.clear_options()
        options: list[Option] = []
        idx = 0
        for section_n, section in enumerate(sections):
            if section.heading:
                options.append(_heading_option(section.heading, section_n))
            for row in section.rows:
                option_id = f"row-{idx}"
                self._row_by_option_id[option_id] = row
                options.append(Option(_render_row(row), id=option_id))
                idx += 1
        if options:
            self.add_options(options)

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is None:
            return
        row = self._row_by_option_id.get(event.option.id)
        if row is None:
            return
        event.stop()
        self.post_message(self.Selected(row))


def _heading_option(heading: str, n: int) -> Option:
    return Option(
        Content.styled(heading, "bold $accent"),
        id=f"heading-{n}",
        disabled=True,
    )


def _render_row(row: CatalogRow) -> Content:
    if isinstance(row, FrontierCatalogRow):
        return _render_frontier(row)
    return _render_local(row)


def _render_frontier(row: FrontierCatalogRow) -> Content:
    parts: list[Content] = [
        Content.styled(row.name, "bold"),
        Content("  "),
        pill(row.provider, "$accent", "$text"),
        Content(" "),
        _key_status_pill(row.key_status),
    ]
    return Content.assemble(*parts)


def _render_local(row: LocalCatalogRow) -> Content:
    bg = TASK_COLORS.get(row.task, "$primary")
    parts: list[Content] = []
    if row.featured:
        parts.append(Content.styled(f"{FEATURED_STAR} ", "$warning"))
    parts.append(Content.styled(row.name, "bold"))
    parts.append(Content("  "))
    parts.append(pill(row.task, bg, "$text"))
    if row.backend:
        parts.append(Content(" "))
        parts.append(pill(row.backend, "$accent", "$text"))
    specs = _format_specs(row)
    if specs:
        parts.append(Content("  "))
        parts.append(Content.styled(specs, "$text-muted"))
    if row.installed:
        parts.append(Content("  "))
        parts.append(pill("installed", "$success", "$text"))
    return Content.assemble(*parts)


def _key_status_pill(status: KeyStatus) -> Content:
    if status == KeyStatus.READY:
        return pill("ready", "$success", "$text")
    return pill("needs key", "$warning", "$text")


def _format_specs(row: LocalCatalogRow) -> str:
    parts = [p for p in (row.params, row.quant, row.size) if p and p != "--"]
    return f" {MIDDLE_DOT} ".join(parts)
