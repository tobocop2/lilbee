"""Per-family model compatibility profiles.

Each module under this package declares one ``PROFILE = FamilyProfile(...)``
data literal that consolidates everything lilbee needs to know to load,
prompt, and extract from one chat-template family: matching patterns, the
llama-cpp chat-format preset to use (if any), an HF tokenizer download
target (if required), the streaming policy, and the output-format hint
the response parser uses to decide whether bare-JSON fallback applies.

:data:`ALL_PROFILES` is discovered at import time and ordered most-specific
first via :data:`_MATCH_ORDER`. Adding a family is one new module plus one
entry in ``_MATCH_ORDER``.
"""

from __future__ import annotations

import importlib
import pkgutil

from lilbee.providers.families.profile import (
    FamilyProfile,
    OutputFormat,
    StreamingPolicy,
)
from lilbee.providers.families.registry import detect, registry

# Detection order: ref-pattern matches win over name matches win over template
# markers win over architecture fallbacks. Profiles with the narrowest hints
# go first; broader fallbacks last. Modules not listed here are skipped at
# discovery time so an unfinished WIP profile never silently fires.
_MATCH_ORDER: tuple[str, ...] = (
    "qwen3_coder",
    "cohere",
    "gpt_oss",
    "ernie",
    "deepseek_v31",
    "granite",
    "phi4mini",
    "functionary_v3",
    "hermes",
    "smollm",
    "llama3",
    "kimi_k2",
    "olmo3",
    "lfm2",
    "glm47",
    "glm46",
    "gemma4",
    "mistral",
    "qwen3",
    "internlm2",
)


def _discover_profiles() -> tuple[FamilyProfile, ...]:
    """Import every sibling module named in :data:`_MATCH_ORDER` and read its ``PROFILE``."""
    available = {name for _, name, _ in pkgutil.iter_modules(__path__)}
    missing = [name for name in _MATCH_ORDER if name not in available]
    if missing:
        raise RuntimeError(f"Family profiles named in _MATCH_ORDER but not importable: {missing}")
    extras = sorted(available - {*_MATCH_ORDER, "profile", "registry"})
    if extras:
        raise RuntimeError(f"Family modules present but absent from _MATCH_ORDER: {extras}")
    profiles: list[FamilyProfile] = []
    for name in _MATCH_ORDER:
        module = importlib.import_module(f"{__name__}.{name}")
        profile = module.PROFILE
        if not isinstance(profile, FamilyProfile):
            raise TypeError(f"{name}.PROFILE must be a FamilyProfile, got {type(profile).__name__}")
        profiles.append(profile)
    return tuple(profiles)


ALL_PROFILES: tuple[FamilyProfile, ...] = _discover_profiles()

__all__ = [
    "ALL_PROFILES",
    "FamilyProfile",
    "OutputFormat",
    "StreamingPolicy",
    "detect",
    "registry",
]
