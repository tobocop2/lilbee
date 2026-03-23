"""Sync status bar widget."""

from __future__ import annotations

from textual.widgets import Static


class SyncBar(Static):
    """One-line status bar showing background sync progress."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", **kwargs)  # type: ignore[arg-type]

    def set_status(self, text: str) -> None:
        """Update the sync status text."""
        self.update(text)
