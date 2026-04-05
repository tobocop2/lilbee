"""Pill badge — colored inline label using half-block characters.

Ported from toad (https://github.com/batrachianai/toad).
"""

from textual.content import Content

PILL_LEFT = "\u258c"  # ▌ left half block
PILL_RIGHT = "\u2590"  # ▐ right half block


def pill(text: Content | str, background: str, foreground: str) -> Content:
    """Format text as a pill badge with rounded half-block ends.

    Args:
        text: Pill contents.
        background: Background color (Textual color string, e.g. "$primary").
        foreground: Foreground color (Textual color string, e.g. "$text").

    Returns:
        Styled Content with half-block ends.
    """
    content = Content(text) if isinstance(text, str) else text
    main_style = f"{foreground} on {background}"
    end_style = f"{background} on transparent r"
    return Content.assemble(
        (PILL_LEFT, end_style),
        content.stylize(main_style),
        (PILL_RIGHT, end_style),
    )
