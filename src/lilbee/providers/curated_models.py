"""Curated chat-model lists per cloud provider.

litellm's ``models_by_provider`` is exhaustive: it carries every alias,
preview snapshot, and deprecated id the provider has ever published.
Surfacing that raw list in a flat dropdown makes the picker unusable
once a user has any provider key configured (Gemini alone publishes
50+ entries).

This module is the single source of truth for "which models does
lilbee surface by default in the picker." The catalog screen still
shows everything; the picker / dropdown defaults to this curated set
and exposes a "Show all" affordance backed by the un-curated litellm
catalog.

Maintenance: each provider entry carries a ``# Last verified: YYYY-MM-DD``
header. Bump it when curation is reviewed against the upstream catalog.
Models removed by the provider stay listed until the next review pass
so users on older lilbee builds keep seeing the entries they configured.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratedModel:
    """A blessed model id with display label.

    ``is_recent`` flags the most recent stable models per provider; the
    picker pins these at the top of each provider's section and shows
    a small recency indicator next to them in the catalog.
    """

    id: str
    display_name: str
    is_recent: bool = True


# Last verified: 2026-04-30
_GEMINI: tuple[CuratedModel, ...] = (
    CuratedModel("gemini-2.0-flash", "Gemini 2.0 Flash"),
    CuratedModel("gemini-1.5-pro", "Gemini 1.5 Pro"),
    CuratedModel("gemini-1.5-flash", "Gemini 1.5 Flash"),
    CuratedModel("gemini-1.5-flash-8b", "Gemini 1.5 Flash 8B", is_recent=False),
)

# Last verified: 2026-04-30
_OPENAI: tuple[CuratedModel, ...] = (
    CuratedModel("gpt-4o", "GPT-4o"),
    CuratedModel("gpt-4o-mini", "GPT-4o mini"),
    CuratedModel("gpt-4-turbo", "GPT-4 Turbo", is_recent=False),
    CuratedModel("o1", "o1"),
    CuratedModel("o1-mini", "o1 mini"),
    CuratedModel("gpt-3.5-turbo", "GPT-3.5 Turbo", is_recent=False),
)

# Last verified: 2026-04-30
_ANTHROPIC: tuple[CuratedModel, ...] = (
    CuratedModel("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet"),
    CuratedModel("claude-3-5-haiku-latest", "Claude 3.5 Haiku"),
    CuratedModel("claude-3-opus-latest", "Claude 3 Opus", is_recent=False),
    CuratedModel("claude-3-haiku-20240307", "Claude 3 Haiku", is_recent=False),
)

CURATED_CHAT_MODELS: dict[str, tuple[CuratedModel, ...]] = {
    "gemini": _GEMINI,
    "openai": _OPENAI,
    "anthropic": _ANTHROPIC,
}

# Cap for providers without a curated entry: alphabetical top-N from the
# raw catalog. Picks a small list a new provider can graduate from
# without flooding the dropdown the moment its key lands.
TOP_N_FALLBACK = 8


def curated_ids(provider: str) -> list[str]:
    """Return curated model ids for ``provider`` in display order."""
    return [m.id for m in CURATED_CHAT_MODELS.get(provider, ())]
