"""The ``Content-Disposition`` header that names a downloaded file."""

from __future__ import annotations

import re
from urllib.parse import quote

CONTENT_DISPOSITION = "Content-Disposition"
# Printable ASCII except the quote and backslash a quoted filename cannot hold bare.
_FILENAME_UNSAFE_RE = re.compile(r"[^ !#-\[\]-~]")
_FILENAME_PLACEHOLDER = "_"


def attachment_disposition(filename: str) -> str:
    """An RFC 6266 attachment header: an ASCII *filename* fallback plus the exact UTF-8 name."""
    fallback = _FILENAME_UNSAFE_RE.sub(_FILENAME_PLACEHOLDER, filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"
