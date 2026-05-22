"""``FamilyRegistry`` builds the family-profile lookup and ordered matcher."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

from lilbee.providers.families.profile import FamilyProfile

if TYPE_CHECKING:
    from lilbee.providers.worker.response_parser.families import TemplateFamily


class FamilyRegistry:
    """Owns the family-profile collection and exposes lookup plus detection."""

    def __init__(self, profiles: tuple[FamilyProfile, ...]) -> None:
        self._by_family: dict[TemplateFamily, FamilyProfile] = {p.family: p for p in profiles}
        self._match_order: tuple[FamilyProfile, ...] = tuple(profiles)

    @property
    def all_profiles(self) -> tuple[FamilyProfile, ...]:
        return self._match_order

    def by_family(self, family: TemplateFamily) -> FamilyProfile | None:
        return self._by_family.get(family)

    def detect(
        self, metadata: Mapping[str, object] | None, ref: str | None = None
    ) -> FamilyProfile | None:
        for profile in self._match_order:
            if profile.matches(metadata, ref):
                return profile
        return None


@functools.lru_cache(maxsize=1)
def registry() -> FamilyRegistry:
    """Singleton registry built from the package-level ``ALL_PROFILES`` tuple."""
    # circular: families.__init__ imports detect/registry from this module while
    # this function reads ALL_PROFILES back from families.__init__. Resolved by
    # importing inside the function so the package finishes binding ALL_PROFILES
    # before the first call lands here.
    from lilbee.providers.families import ALL_PROFILES

    return FamilyRegistry(ALL_PROFILES)


def detect(metadata: Mapping[str, object] | None, ref: str | None = None) -> FamilyProfile | None:
    """One-shot helper that defers to the singleton registry."""
    return registry().detect(metadata, ref)
