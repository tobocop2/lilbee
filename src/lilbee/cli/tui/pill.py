"""Pill badge: a colored inline label with a single space of padding each side.

The padding is part of the ``{background}`` fill, not a separate styled cap:
half-block caps over a ``transparent`` background take their color from whatever
is behind the pill, which on a near-black terminal reads as a black sliver. A
flat fill renders the same on any terminal and theme.
"""

from textual.content import Content

DOT_SEP = " · "  # middle-dot separator for inline dividers


def pill(text: Content | str, background: str, foreground: str) -> Content:
    """Format *text* as a colored pill badge: the label on a ``{background}`` fill.

    Args:
        text: Pill contents.
        background: Background color (Textual color string, e.g. ``"$primary"``).
        foreground: Foreground color (Textual color string, e.g. ``"$text"``).

    Returns:
        Styled ``Content`` with one space of padding on each side.
    """
    content = Content(text) if isinstance(text, str) else text
    style = f"{foreground} on {background}"
    return Content.assemble((" ", style), content.stylize(style), (" ", style))
