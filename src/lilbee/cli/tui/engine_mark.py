"""The color mark that shows which engine runs a model.

Every theme that ctrl+t cycles gets one CSS variable for each engine, named
``engine-<id>`` (for example ``$engine-llama-cpp``). The color of an engine keeps one
hue in every theme. Its lightness moves until the mark reads against the background and
the surface of the theme, so no theme needs a hand-picked table that can drift.

Only the ``■`` mark takes the color. The name keeps the text color, and an engine color
is never a pill fill and never a fit or warning color: the hues sit between blue and
magenta, where no built-in theme puts its success, warning or error color.
"""

from __future__ import annotations

import colorsys

from textual.color import Color
from textual.content import Content

from lilbee.providers.fleet.engines.base import EngineName

MARK = "■"
UNKNOWN_ENGINE_COLOR = "$text-muted"

# One hue in degrees for each engine. Success sits at 60 to 190 in the built-in themes,
# warning at 15 to 45 and error at 335 to 10, so 195 to 325 stays clear of all three.
ENGINE_HUES: dict[EngineName, int] = {
    EngineName.LLAMA: 205,
    EngineName.OPENVINO: 225,
    EngineName.IK_LLAMA: 245,
    EngineName.TEI: 265,
    EngineName.MISTRAL: 285,
    EngineName.LEMONADE: 305,
    EngineName.MLX: 325,
    EngineName.OVMS: 200,
}

# Saturation alternates along the hues, so two close hues still differ.
_SATURATIONS = (0.95, 0.5)
_LIGHTNESS_START = 0.6
_LIGHTNESS_STEP = 0.01
_LIGHTNESS_MAX = 0.95
# WCAG asks 3:1 for a graphic mark. The mark sits next to text, so ask for the text grade.
MIN_CONTRAST = 4.5
# The sRGB threshold of the WCAG luminance formula.
_LINEAR_CUTOFF = 0.03928


def variable_name(engine: str) -> str:
    """The CSS variable name of *engine*, without the ``$``, such as ``engine-llama-cpp``."""
    return "engine-" + engine.replace(".", "-").replace("_", "-")


def _luminance(color: Color) -> float:
    """The WCAG relative luminance of *color*."""

    def channel(value: int) -> float:
        scaled = value / 255
        return scaled / 12.92 if scaled <= _LINEAR_CUTOFF else ((scaled + 0.055) / 1.055) ** 2.4

    r, g, b = color.rgb
    return 0.2126 * channel(int(r)) + 0.7152 * channel(int(g)) + 0.0722 * channel(int(b))


def contrast(first: Color, second: Color) -> float:
    """The WCAG contrast ratio of two colors, from 1 to 21."""
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def engine_color(hue: int, saturation: float, *surfaces: Color) -> Color:
    """The color of one hue that reads against every surface, as light as needed."""
    lightness = _LIGHTNESS_START
    while True:
        red, green, blue = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
        color = Color(round(red * 255), round(green * 255), round(blue * 255))
        if lightness >= _LIGHTNESS_MAX or all(
            contrast(color, surface) >= MIN_CONTRAST for surface in surfaces
        ):
            return color
        lightness += _LIGHTNESS_STEP


def engine_variables(theme_variables: dict[str, str]) -> dict[str, str]:
    """One ``engine-<id>`` variable for each engine, drawn for the theme in *theme_variables*."""
    surfaces = tuple(Color.parse(theme_variables[key]) for key in ("background", "surface"))
    return {
        variable_name(engine): engine_color(
            hue, _SATURATIONS[place % len(_SATURATIONS)], *surfaces
        ).hex
        for place, (engine, hue) in enumerate(ENGINE_HUES.items())
    }


def engine_mark(engine: str) -> Content:
    """The mark and the name of *engine*: the mark takes the engine color, the name does not."""
    known = engine in set(EngineName)
    color = f"${variable_name(engine)}" if known else UNKNOWN_ENGINE_COLOR
    return Content.assemble((MARK, color), " ", engine)


_MB = 1_000_000
_GB = 1_000_000_000


def format_size(size: int) -> str:
    """A download size as one short figure, such as ``180 MB`` or ``1.2 GB``."""
    if size >= _GB:
        return f"{size / _GB:.1f} GB"
    return f"{max(round(size / _MB), 1)} MB"
