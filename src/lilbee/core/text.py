"""Pure label/slug helpers shared across wiki and retrieval layers.

These functions know nothing about the wiki page layout; they exist
here so retrieval/concepts and wiki can both call them without
introducing an upward layer dependency.
"""

from __future__ import annotations

import re

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]")

# Characters that signal markdown-structural noise in a concept label.
# Single source of truth for both ``is_valid_label`` (membership check)
# and ``clean_label_for_display`` (regex strip).
_STRUCTURAL_CHARS = frozenset("|#>")
_DISPLAY_STRUCTURAL_RE = re.compile(f"[{re.escape(''.join(_STRUCTURAL_CHARS))}]+")
_DISPLAY_WHITESPACE_RE = re.compile(r"\s+")

LABEL_SANITY_MIN_LEN = 3
LABEL_SANITY_MIN_ALNUM_RATIO = 0.5


def make_slug(label: str) -> str:
    """Turn a concept label into a filesystem-safe slug.

    Lowercases, maps whitespace to single hyphens and slashes to double
    hyphens (path encoding), strips anything outside ``[a-z0-9-]``, and
    trims leading and trailing hyphens. Returns ``""`` when no sluggable
    characters remain; callers must treat an empty slug as "skip this
    entity" so the generator never writes a file called ``.md``.

    Internal hyphen runs from the ``/`` path encoding are preserved;
    only leading and trailing hyphens (e.g. ``--body`` from a stripped
    ``| | Body``) are removed.
    """
    slug = label.lower().replace(" ", "-").replace("/", "--")
    slug = _SLUG_CLEAN_RE.sub("", slug)
    return slug.strip("-")


def is_valid_label(label: str) -> bool:
    """Reject structural-noise labels before aggregation.

    Catches the noise patterns observed in QA (bb-8b7s):

    - empty or sub-three-char fragments,
    - markdown table delimiters (``| | designer``),
    - page-number-prefixed tokens (``158 vehicle``),
    - paren-prefixed numerics (``(7.0 l)``: would otherwise slug to
      ``70-l`` after punctuation cleanup),
    - hyphen-prefixed fragments (``-answers``: trailing text from
      markdown bracket-link extraction).

    Requires the first non-whitespace character to be a Unicode letter
    so any non-alpha prefix (digit, bracket, hyphen, punctuation) is
    rejected up front. Legitimate labels like ``E-mail`` or ``iPhone``
    pass. Still permissive on three-char fragments like ``cro`` /
    ``fus``; A3's entity-type filter and ``wiki_entity_min_mentions``
    catch those downstream.
    """
    stripped = label.strip()
    if len(stripped) < LABEL_SANITY_MIN_LEN:
        return False
    if not stripped[0].isalpha():
        return False
    if any(ch in _STRUCTURAL_CHARS for ch in stripped):
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    return alnum / len(stripped) >= LABEL_SANITY_MIN_ALNUM_RATIO


def clean_label_for_display(label: str) -> str:
    """Return a prompt-safe version of *label* for the ``{topic}`` slot.

    Defense-in-depth behind :func:`is_valid_label`: a concept or entity
    label that reached this function already passed the sanity gate
    and should not contain ``|#>`` in practice. The structural-char
    strip here guards against a future code path that bypasses the
    gate (synthesis cluster labels sourced from ``concept_nodes``,
    user-supplied topics, tests). The always-useful work is whitespace
    normalization: spaCy surface forms can carry internal runs of
    whitespace that would reach the H1 verbatim.

    Preserves the original capitalization so proper nouns
    (``Chevrolet Caprice``, ``iPhone``) survive intact; the model
    title-cases lowercase common nouns on its own.
    """
    clean = _DISPLAY_STRUCTURAL_RE.sub("", label)
    return _DISPLAY_WHITESPACE_RE.sub(" ", clean).strip()
