"""Document title and source-level metadata derivation for ingest."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Protocol

from lilbee.data.store import SourceMeta

# Filename-stem separators flattened to spaces when no extracted title exists.
_STEM_SEPARATOR_RE = re.compile(r"[_\-\s]+")


class ExtractionMetadata(Protocol):
    """The extraction-metadata fields lilbee folds into a source row.

    Structural, so it is satisfied by both xberg's public ``Metadata`` and the
    native type its extractor actually hands back. The fields are ``object``
    because xberg annotates but does not enforce them: ``authors`` is declared
    ``list[str] | None`` yet arrives as a bare ``str`` for a PDF ``/Author``, and
    a non-string ``title`` is accepted. Callers narrow before using them.
    """

    @property
    def title(self) -> object: ...

    @property
    def authors(self) -> object: ...

    @property
    def created_at(self) -> object: ...


def derive_title(source_name: str, metadata_title: object = None) -> str:
    """Human-readable document title: the extracted title, else the cleaned filename stem.

    The stem cleanup flattens underscore/hyphen separators to spaces so BM25
    tokenizes ``survey_214.pdf`` into the same terms a query would use.
    """
    if isinstance(metadata_title, str) and metadata_title.strip():
        return metadata_title.strip()
    return _STEM_SEPARATOR_RE.sub(" ", PurePath(source_name).stem).strip()


def source_meta_from_extraction(
    metadata: ExtractionMetadata | None, source_name: str
) -> SourceMeta:
    """Fold xberg extraction metadata into a :class:`SourceMeta`.

    The title falls back to the filename stem; authors and creation date stay
    empty (persisted NULL) when the extractor reports none. xberg annotates these
    fields but does not enforce them: a PDF ``/Author`` arrives as a bare ``str``
    where ``list[str]`` is declared, so a string is treated as one author rather
    than split into its characters, and non-string entries are coerced.
    """
    raw_authors = metadata.authors if metadata is not None else None
    if isinstance(raw_authors, str):
        authors: list[str] = [raw_authors]
    elif isinstance(raw_authors, (list, tuple)):
        authors = [str(a) for a in raw_authors if a]
    else:
        authors = []
    return SourceMeta(
        title=derive_title(source_name, metadata.title if metadata is not None else None),
        authors=", ".join(a for a in authors if a),
        created_at=str((metadata.created_at if metadata is not None else None) or ""),
    )
