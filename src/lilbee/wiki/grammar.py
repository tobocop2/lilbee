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
INFERENCE_RE = re.compile(r"\[\*inference\*\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
CODE_FENCE_RE = re.compile(r"^(```|~~~)")
H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
