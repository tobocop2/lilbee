"""Build the PyPI long description from README.md.

The README's demos are GitHub video players, which only GitHub renders. This
hook rewrites them to the gh-pages GIFs of the same reels so the PyPI page
keeps its moving pictures. tools/readme_media.py has the details.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface

# hatchling loads this file directly rather than importing it as part of the
# project, so the repository root is not on sys.path and `tools` is invisible.
sys.path.insert(0, str(Path(__file__).parent))

from tools.readme_media import to_gifs


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict[str, Any]) -> None:
        readme = Path(self.root) / "README.md"
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": to_gifs(readme.read_text(encoding="utf-8")),
        }
