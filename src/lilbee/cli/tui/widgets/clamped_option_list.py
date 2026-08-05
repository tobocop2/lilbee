"""OptionList whose height measurement survives a zero-width squeeze."""

from __future__ import annotations

from textual.geometry import Size
from textual.widgets import OptionList


class ClampedOptionList(OptionList):
    """Clamp the measurement width so option wrapping never sees zero.

    Textual 8.2.8 ``OptionList.get_content_height`` feeds ``width minus the
    option padding`` straight into visual wrapping; when a drawer plus a tiny
    terminal squeeze the list to zero columns, rich's cell chopper raises
    ``ValueError: range() arg 3 must not be zero`` and the app dies. Keep the
    subtraction positive until that is fixed upstream.
    """

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        padding_width = self.get_component_styles("option-list--option").padding.width
        return super().get_content_height(container, viewport, max(width, padding_width + 1))
