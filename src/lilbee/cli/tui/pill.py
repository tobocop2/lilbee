"""Pill badge: a colored inline label with one space of ``{background}``-fill padding each side.

Adapted from toad (https://github.com/batrachianai/toad); the padding is part
of the solid fill rather than toad's reversed half-block caps, which take their
color from the cell behind the pill and read as a black sliver on a near-black
terminal.
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
