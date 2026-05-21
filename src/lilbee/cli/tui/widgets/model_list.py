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

from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS
from lilbee.modelhub.models import FEATURED_STAR
from lilbee.runtime.hardware import FitChip, FitLevel

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
        options = self._build_options(sections, start_idx=0)
        if options:
            self.add_options(options)

    def append_rows(self, rows: list[CatalogRow]) -> None:
        """Append rows under the existing sections without rebuilding."""
        if not rows:
            return
        start = len(self._row_by_option_id)
        section = ModelListSection(heading=None, rows=rows)
        options = self._build_options([section], start_idx=start)
        if options:
            self.add_options(options)

    @property
    def row_count(self) -> int:
        """Number of selectable rows currently mounted (excludes section headings)."""
        return len(self._row_by_option_id)

    def row_at(self, option_id: str) -> CatalogRow | None:
        """Return the CatalogRow for the given option id, or None when unknown."""
        return self._row_by_option_id.get(option_id)

    def highlighted_row(self) -> CatalogRow | None:
        """Return the CatalogRow under the highlight cursor, or None."""
        idx = self.highlighted
        if idx is None:
            return None
        try:
            opt = self.get_option_at_index(idx)
        except IndexError:
            return None
        if opt.id is None:
            return None
        return self.row_at(opt.id)

    def _build_options(self, sections: list[ModelListSection], *, start_idx: int) -> list[Option]:
        options: list[Option] = []
        idx = start_idx
        for section_n, section in enumerate(sections):
            if section.heading:
                options.append(_heading_option(section.heading, start_idx + section_n))
            for row in section.rows:
                option_id = f"row-{idx}"
                self._row_by_option_id[option_id] = row
                options.append(Option(_render_row(row), id=option_id))
                idx += 1
        return options

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
    if row.kind == CatalogRowKind.FRONTIER:
        return _render_frontier(row)
    return _render_local(row)


def _render_frontier(row: FrontierCatalogRow) -> Content:
    # Two-line row mirroring the local layout: name + key-status tag on top,
    # provider strip below. Plain text styling, no colored pills.
    line1: list[Content] = [Content("  "), Content.styled(row.name, "bold")]
    if row.key_status == KeyStatus.READY:
        line1.append(Content.styled("    ready", "$success italic"))
    else:
        line1.append(Content.styled("    needs key", "$warning italic"))
    line2: list[Content] = [Content("   "), Content.styled(row.provider, "dim $text-muted")]
    return Content.assemble(*line1, Content("\n"), *line2, Content("\n"))


def _render_local(row: LocalCatalogRow) -> Content:
    line1 = _render_local_headline(row)
    line2 = _render_local_meta(row)
    return Content.assemble(*line1, Content("\n"), *line2, Content("\n"))


def _render_local_headline(row: LocalCatalogRow) -> list[Content]:
    parts: list[Content] = [
        Content.styled(f"{FEATURED_STAR} ", "$warning") if row.featured else Content("  "),
        Content.styled(row.name, "bold"),
    ]
    if row.installed:
        parts.append(Content.styled("    installed", "$success italic"))
    parts.extend(_fit_tag(row.fit))
    parts.extend(_compat_tag(row.compat))
    return parts


def _fit_tag(fit: FitChip | None) -> list[Content]:
    """List-style fit indicator (italic colored text), matching the grid card chip set."""
    if fit is None:
        return []
    if fit.level is FitLevel.FITS:
        return [Content.styled("    fits", "$success italic")]
    if fit.level is FitLevel.TIGHT:
        return [Content.styled("    tight", "$warning italic")]
    return [Content.styled("    won't run", "$error italic")]


def _compat_tag(compat: ModelCompat) -> list[Content]:
    """List-style compat indicator. Empty for SUPPORTED to keep the row visually quiet."""
    from lilbee.cli.tui import messages as msg

    if compat is ModelCompat.SUPPORTED:
        return []
    if compat is ModelCompat.UNSUPPORTED:
        return [Content.styled(f"    {msg.COMPAT_PILL_UNSUPPORTED}", "$warning italic")]
    return [Content.styled(f"    {msg.COMPAT_PILL_UNKNOWN}", "$text-muted italic")]


def _render_local_meta(row: LocalCatalogRow) -> list[Content]:
    parts: list[Content] = [Content("   ")]
    if row.task:
        task_color = TASK_COLORS.get(row.task, "$text-muted")
        parts.append(Content.styled(row.task, f"{task_color} italic"))
        parts.append(Content.styled(f" {MIDDLE_DOT} ", "dim $text-muted"))
    rest = [s for s in _local_meta_strip(row) if s]
    if rest:
        parts.append(Content.styled(f" {MIDDLE_DOT} ".join(rest), "dim $text-muted"))
    return parts


def _local_meta_strip(row: LocalCatalogRow) -> list[str]:
    rest: list[str] = []
    if row.backend and row.backend != "native":
        rest.append(row.backend)
    specs = _format_specs(row)
    if specs:
        rest.append(specs)
    if row.downloads and row.downloads != "--":
        rest.append(f"↓ {row.downloads}")
    return rest


def _format_specs(row: LocalCatalogRow) -> str:
    parts = [p for p in (row.params, row.quant, row.size) if p and p != "--"]
    return f" {MIDDLE_DOT} ".join(parts)
