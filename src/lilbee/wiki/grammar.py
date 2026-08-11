"""Source of truth for the wiki's markdown grammar.

Every structural delimiter and pattern the wiki contract depends on lives
here. Modules that author or parse wiki pages import from this module
instead of re-declaring the patterns. If the grammar ever changes, this
is the only file that needs to move.
"""

from __future__ import annotations

import re

CITATION_BLOCK_SEP = "---"
CITATION_BLOCK_COMMENT = "<!-- citations (auto-generated from _citations table -- do not edit) -->"
CODE_FENCE_PREFIX = "```"

CITE_RE = re.compile(r"\[\^(src\d+)\]")
FOOTNOTE_RE = re.compile(r"^\[\^(src\d+)\]:\s*(.+)$", re.MULTILINE)
# A prompt evidence label ([Chunk 3], [Chunks 1, 2]) echoed into prose, with
# the inline ">" a model glues onto it. A blockquote's own ">" is separated
# from the label by a space, so it stays.
CHUNK_MARKER_RE = re.compile(r"\s*>?\[Chunks?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\]", re.IGNORECASE)
INFERENCE_RE = re.compile(r"\[\*inference\*\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
CODE_FENCE_RE = re.compile(r"^(```|~~~)")
H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
