"""Pill badge — colored inline label using half-block characters.

Ported from toad (https://github.com/batrachianai/toad).
"""

from textual.content import Content


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
        ("\u258c", end_style),
        content.stylize(main_style),
        ("\u2590", end_style),
    )
