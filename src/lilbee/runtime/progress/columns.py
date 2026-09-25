"""Rich progress columns that show task text literally, never as markup."""

from __future__ import annotations

from rich.progress import TextColumn


def literal_text_column(text_format: str, *, style: str = "none") -> TextColumn:
    """A text column whose rendered task fields are never parsed as markup."""
    return TextColumn(text_format, style=style, markup=False)
