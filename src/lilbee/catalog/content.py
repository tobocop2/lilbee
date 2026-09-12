"""Content flag for catalog rows, derived from HuggingFace repo tags."""

from collections.abc import Iterable

# Tag stems marking a safety-stripped (abliterated/uncensored) model.
# Surveyed 2026-09-12 over 200 trending text-generation GGUF repos: uncensored
# (38) and abliterated (28) dominate; decensored, nsfw, and heretic cover the
# rest of the tag vocabulary, each a singleton-class tag in the sample.
STRIPPED_TAG_STEMS: tuple[str, ...] = ("uncensor", "abliter", "decensor", "nsfw", "heretic")


def is_safety_stripped(tags: Iterable[str]) -> bool:
    """True when any tag names safety-stripped content (case-insensitive)."""
    return any(stem in tag.lower() for tag in tags for stem in STRIPPED_TAG_STEMS)
