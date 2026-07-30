"""Type definitions for the chunk-level mutual-kNN clusterer."""

from __future__ import annotations

from dataclasses import dataclass, field

# Minimum token length for TF-IDF labeling. Shorter tokens are mostly
# articles, prepositions, and single letters: noise that inflates term
# counts without adding topic signal. Three characters keeps useful
# acronyms (api, xml, sql).
_MIN_TF_TOKEN_LEN = 3


def _tokenize_for_tf(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for TF-IDF scoring.

    Deliberately has NO stopword list: common words like "the" or "and"
    get an IDF near zero (they appear in almost every chunk) so TF-IDF
    filters them automatically. A hand-curated English stoplist would
    add maintenance burden and break on non-English corpora for no
    additional quality.

    Lives beside :class:`ClusterChunk` because the record derives its
    tokens from it; importing it from the helpers module would be circular.
    """
    result: list[str] = []
    for raw in text.lower().split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if len(word) >= _MIN_TF_TOKEN_LEN:
            result.append(word)
    return result


@dataclass(slots=True)
class ClusterChunk:
    """Lightweight view of one chunk row used by the clusterer.

    Named for the clusterer rather than ``ChunkRecord`` so it cannot be
    confused with :class:`lilbee.data.types.ChunkRecord`, an
    unrelated TypedDict with a different field set used across ingest.

    ``tokens`` is a derived cache of ``_tokenize_for_tf(text)``. It is
    computed here rather than at each construction site: TF-IDF labeling
    silently degrades to fallback labels when a record carries no tokens,
    so the invariant is structural instead of caller discipline. ``slots``
    keeps the per-record footprint down, since one record is materialized
    per corpus chunk.
    """

    source: str
    chunk_index: int
    text: str
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = _tokenize_for_tf(self.text)
