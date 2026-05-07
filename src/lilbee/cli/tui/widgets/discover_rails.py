"""Discover-tab rails: For You / Your Collection / Fresh.

Each rail is a small ModelGrid bound to a curated row slice the catalog
screen passes in via ``set_rails``. The widget owns layout (heading +
ModelGrid stack) and nothing else; row construction stays in the
catalog screen so the rails inherit every cache, fit-stamp, and routing
behavior the per-task tabs already have.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.model_grid import ModelGrid

_CSS_FILE = Path(__file__).parent / "discover_rails.tcss"


class DiscoverRails(Vertical):
    """Stack of three named rails. Each rail is a small ModelGrid."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8") if _CSS_FILE.exists() else ""

    _RAIL_FOR_YOU = "For You"
    _RAIL_COLLECTION = "Your Collection"
    _RAIL_FRESH = "Fresh on the Hub"

    def compose(self) -> ComposeResult:
        for rail_id, label in (
            ("for-you", self._RAIL_FOR_YOU),
            ("collection", self._RAIL_COLLECTION),
            ("fresh", self._RAIL_FRESH),
        ):
            yield Static(label, classes="discover-rail-heading", id=f"heading-{rail_id}")
            yield ModelGrid(id=f"discover-grid-{rail_id}", classes="discover-rail-grid")

    def set_rails(
        self,
        *,
        for_you: list[LocalCatalogRow],
        collection: list[LocalCatalogRow],
        fresh: list[LocalCatalogRow],
    ) -> None:
        """Push three row slices into their respective rail grids.

        Empty lists render an empty grid (zero height); we don't omit
        the heading because a steady three-rail layout reads better than
        a layout that shifts when one rail has data and another doesn't.
        """
        self._set_rail("for-you", for_you)
        self._set_rail("collection", collection)
        self._set_rail("fresh", fresh)

    def _set_rail(self, rail_id: str, rows: list[LocalCatalogRow]) -> None:
        try:
            grid = self.query_one(f"#discover-grid-{rail_id}", ModelGrid)
        except Exception:
            return
        grid.set_rows(rows)
