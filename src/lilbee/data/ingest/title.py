"""Document title and source-level metadata derivation for ingest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

from lilbee.data.store import SourceMeta

# Filename-stem separators flattened to spaces when no extracted title exists.
_STEM_SEPARATOR_RE = re.compile(r"[_\-\s]+")


def derive_title(source_name: str, metadata_title: str | None = None) -> str:
    """Human-readable document title: the extracted title, else the cleaned filename stem.

    The stem cleanup flattens underscore/hyphen separators to spaces so BM25
    tokenizes ``survey_214.pdf`` into the same terms a query would use.
    """
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()
    return _STEM_SEPARATOR_RE.sub(" ", PurePath(source_name).stem).strip()


def source_meta_from_extraction(metadata: Mapping[str, Any], source_name: str) -> SourceMeta:
    """Fold kreuzberg extraction metadata into a :class:`SourceMeta`.

    The title falls back to the filename stem; authors and creation date stay
    empty (persisted NULL) when the extractor reports none.
    """
    authors = metadata.get("authors") or []
    return SourceMeta(
        title=derive_title(source_name, metadata.get("title")),
        authors=", ".join(a for a in authors if a),
        created_at=str(metadata.get("created_at") or ""),
    )
