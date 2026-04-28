"""spaCy-backed NLP helpers for the concept graph."""

from __future__ import annotations

import logging
from typing import Any

from lilbee.wiki.shared import is_valid_label

log = logging.getLogger(__name__)


def concepts_available() -> bool:
    """Check if concept graph dependencies (spacy, graspologic) are installed."""
    try:
        import graspologic_native  # noqa: F401
        import spacy  # noqa: F401

        return True
    except ImportError:
        return False


def _ensure_spacy_model() -> Any:
    """Load the spaCy NER model; raise ImportError with an install hint if missing."""
    import spacy

    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise ImportError(
            f"spaCy model {model_name!r} not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


def load_spacy_pipeline() -> Any:
    """Public wrapper around the shared spaCy NER + noun-chunk pipeline.

    Raises ``ImportError`` if spaCy or the ``en_core_web_sm`` model cannot
    be installed.
    """
    return _ensure_spacy_model()


def _filter_noun_chunks(doc: Any, max_concepts: int) -> list[str]:
    """Extract deduplicated, filtered noun chunks from a spaCy doc.

    Applies the same :func:`is_valid_label` gate the wiki entity
    extractor uses, so structural-noise concepts (markdown table
    delimiters, page-number-prefixed tokens, sub-three-char fragments)
    never enter the co-occurrence graph and therefore never become a
    synthesis-page cluster label.

    The gate runs on the lowercased form here while the NER extractor
    gates on the original-cased surface; the two decisions match
    because ``is_valid_label`` is case-agnostic today. Any future
    case-sensitive rule must land in both call sites together.
    """
    seen: set[str] = set()
    concepts: list[str] = []
    for chunk in doc.noun_chunks:
        concept = chunk.text.lower().strip()
        if not is_valid_label(concept):
            continue
        if concept in seen:
            continue
        seen.add(concept)
        concepts.append(concept)
        if len(concepts) >= max_concepts:
            break
    return concepts
