"""Footer that skips a recompose when the shown bindings have not changed."""

from __future__ import annotations

from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer

_BindingSignature = tuple[tuple[str, str, bool], ...]


def _bindings_signature(screen: Screen) -> _BindingSignature:
    """The (key, action, enabled) triples a Footer render of *screen* would show."""
    return tuple(
        sorted(
            (key, binding.action, enabled)
            for key, (_namespace, binding, enabled, _tooltip) in screen.active_bindings.items()
        )
    )


class StableFooter(Footer):
    """Footer that recomposes only when its rendered key row would change.

    Textual's own ``Footer.bindings_changed`` recomposes on every screen
    ``refresh_bindings()`` call, which fires on every focus change (once for
    the blur, once for the focus). Each recompose tears down and remounts
    every ``FooterKey``, which costs a full CSS stylesheet apply per new
    widget. A screen with many focusable fields where most transitions
    leave the same bindings enabled (moving between two Input rows, for
    example) pays that cost on every Tab press for no visible change.
    """

    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        show_command_palette: bool = True,
        compact: bool = False,
    ) -> None:
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            show_command_palette=show_command_palette,
            compact=compact,
        )
        self._last_bindings_signature: _BindingSignature | None = None

    def bindings_changed(self, screen: Screen) -> None:
        self._bindings_ready = True
        if not screen.app.app_focus:
            return
        if not (self.is_attached and screen is self.screen):
            return
        signature = _bindings_signature(screen)
        if signature == self._last_bindings_signature:
            return
        self._last_bindings_signature = signature
        self.call_after_refresh(self.recompose)
