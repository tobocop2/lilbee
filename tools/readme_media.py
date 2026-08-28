"""Turn the README's GitHub video players back into GIFs.

The README shows each demo as a GitHub video player: a bare
``https://github.com/user-attachments/assets/<uuid>`` URL on its own line.
GitHub is the only renderer that turns that into a player. PyPI prints it as a
plain link, so the demos would go from moving pictures to a wall of URLs.

Every video is preceded by a ``<!-- demo: <name> | <caption> -->`` comment, so
the same reel's GIF on gh-pages is always recoverable, caption included. The
PyPI long description is built through here (see hatch_build.py); GitHub gets
the videos, PyPI keeps the GIFs, and the README stays the single source.

hatch-fancy-pypi-readme does the same substitution from pyproject.toml and was
the obvious buy. It is one `re.sub` either way, and putting the pattern in TOML
would either drop the missing-comment check below or make the test parse the
pattern back out of pyproject. A build dependency is not worth that.
"""

from __future__ import annotations

import re

GIF_BASE = "https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/"

VIDEO_BLOCK = re.compile(
    r"<!-- demo: (?P<demo>[a-z0-9_-]+) \| (?P<caption>[^\n]*?) -->\n"
    r"\n"
    r"https://github\.com/user-attachments/assets/[0-9a-f-]+"
)
_ASSET_URL = "user-attachments/assets"


def to_gifs(markdown: str) -> str:
    """Replace every GitHub video block with the GIF of the same reel."""
    result = VIDEO_BLOCK.sub(
        lambda m: f"![{m['caption']}]({GIF_BASE}{m['demo']}.gif)",
        markdown,
    )
    stray = [line for line in result.splitlines() if _ASSET_URL in line]
    if stray:
        raise ValueError(
            "README video without a '<!-- demo: <name> | <caption> -->' comment "
            f"above it, so it has no GIF to fall back to on PyPI: {stray}"
        )
    return result
