"""Document title and source-level metadata derivation for ingest."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import TYPE_CHECKING

from lilbee.data.store import SourceMeta

if TYPE_CHECKING:
    from xberg import Metadata

# Filename-stem separators flattened to spaces when no extracted title exists.
_STEM_SEPARATOR_RE = re.compile(r"[_\-\s]+")


def derive_title(source_name: str, metadata_title: str | None = None) -> str:
    """Human-readable document title: the extracted title, else the cleaned filename stem.

    The stem cleanup flattens underscore/hyphen separators to spaces so BM25
    tokenizes ``survey_214.pdf`` into the same terms a query would use.
    """
    if isinstance(metadata_title, str) and metadata_title.strip():
        return metadata_title.strip()
    return _STEM_SEPARATOR_RE.sub(" ", PurePath(source_name).stem).strip()


def source_meta_from_extraction(metadata: Metadata, source_name: str) -> SourceMeta:
    """Fold xberg extraction metadata into a :class:`SourceMeta`.

    The title falls back to the filename stem; authors and creation date stay
    empty (persisted NULL) when the extractor reports none.

    ``Metadata`` annotates ``authors`` as ``list[str] | None``, but the binding
    does not enforce it: ``Metadata(authors="John Doe")`` keeps a bare string,
    which is the shape a raw PDF ``/Author`` field arrives in. Joining that
    directly yields "J, o, h, n", so the entries are still coerced here.
    """
    raw_authors = metadata.authors
    if isinstance(raw_authors, str):
        authors: list[str] = [raw_authors]
    elif isinstance(raw_authors, (list, tuple)):
        authors = [str(author) for author in raw_authors if author]
    else:
        authors = []
    return SourceMeta(
        title=derive_title(source_name, metadata.title),
        authors=", ".join(author for author in authors if author),
        created_at=str(metadata.created_at or ""),
    )
