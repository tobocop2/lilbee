"""Tests for ★ Picks pinned section in per-task grid grouping."""

from __future__ import annotations

from lilbee.cli.tui.screens.catalog import (
    _PICKS_SECTION_HEADING,
    _group_task_rows_with_picks,
)
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.modelhub.models import ModelTask


def _row(name: str, *, featured: bool = False, installed: bool = False) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=featured,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


def test_picks_section_heading_uses_star_glyph() -> None:
    assert _PICKS_SECTION_HEADING.startswith("★")


def test_featured_rows_go_into_picks_section() -> None:
    rows = [_row("Llama", featured=True), _row("Mistral", featured=True), _row("DeepSeek")]
    sections = _group_task_rows_with_picks(rows, "Chat")
    by_heading = {s.heading: s.rows for s in sections}
    assert {r.name for r in by_heading[_PICKS_SECTION_HEADING]} == {"Llama", "Mistral"}


def test_installed_rows_go_into_installed_section() -> None:
    rows = [_row("Llama", installed=True), _row("Mistral")]
    sections = _group_task_rows_with_picks(rows, "Chat")
    by_heading = {s.heading: s.rows for s in sections}
    assert [r.name for r in by_heading["Installed"]] == ["Llama"]
    assert [r.name for r in by_heading["Chat"]] == ["Mistral"]


def test_picks_appear_above_installed_above_others() -> None:
    rows = [
        _row("Llama", featured=True),
        _row("Mistral", installed=True),
        _row("DeepSeek"),
    ]
    sections = _group_task_rows_with_picks(rows, "Chat")
    headings = [s.heading for s in sections]
    assert (
        headings.index(_PICKS_SECTION_HEADING)
        < headings.index("Installed")
        < headings.index("Chat")
    )


def test_featured_takes_precedence_over_installed() -> None:
    """A row that is both featured AND installed lives in Picks, not Installed.

    Picks is the curation layer; if the curator featured a model, the
    user should see it under Picks regardless of disk state.
    """
    rows = [_row("Llama", featured=True, installed=True)]
    sections = _group_task_rows_with_picks(rows, "Chat")
    by_heading = {s.heading: s.rows for s in sections}
    assert [r.name for r in by_heading[_PICKS_SECTION_HEADING]] == ["Llama"]
    assert by_heading["Installed"] == []


def test_empty_input_yields_three_empty_sections() -> None:
    """All three sections still emit so callers can rely on stable section count;
    the catalog screen filters out empty sections at render time."""
    sections = _group_task_rows_with_picks([], "Chat")
    assert [s.heading for s in sections] == [_PICKS_SECTION_HEADING, "Installed", "Chat"]
    assert all(s.rows == [] for s in sections)


def test_task_label_passes_through_into_other_section_heading() -> None:
    rows = [_row("nomic-embed")]
    sections = _group_task_rows_with_picks(rows, "Embedding")
    headings = [s.heading for s in sections]
    assert "Embedding" in headings
    assert "Chat" not in headings
