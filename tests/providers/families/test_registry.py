"""Coverage for the family-profile registry: presence, ordering, lookup, detect()."""

from __future__ import annotations

import pytest

from lilbee.providers.families import ALL_PROFILES, detect, registry
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily
from lilbee.providers.worker.response_parser.schemas import get_schemas

# Presets we explicitly install at load time via llama-cpp's registry.
# If any profile's chat_format_override names a preset llama-cpp doesn't
# know, lilbee will fail at load with a confusing error.
_VALID_LLAMA_CPP_PRESETS = {
    "chatml",
    "chatml-function-calling",
    "functionary",
    "functionary-v1",
    "functionary-v2",
    "llama-2",
    "llama-3",
    "mistral-instruct",
    "qwen",
    "gemma",
    "openchat",
    "phind",
    "vicuna",
}


def test_all_profiles_loaded() -> None:
    """Every family schema in response_parser/schemas/ has a matching profile."""
    schema_families = {f for f in get_schemas() if f is not TemplateFamily.UNKNOWN}
    profile_families = {p.family for p in ALL_PROFILES}
    missing = schema_families - profile_families
    assert not missing, f"profiles missing for: {sorted(f.value for f in missing)}"


def test_registry_singleton_is_cached() -> None:
    """Repeated registry() calls return the same instance (lru_cache contract)."""
    assert registry() is registry()


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_profile_chat_format_override_is_known_preset(profile: FamilyProfile) -> None:
    """If a profile declares a chat_format override, the preset must exist in llama-cpp."""
    if profile.chat_format_override is None:
        return
    assert profile.chat_format_override in _VALID_LLAMA_CPP_PRESETS, (
        f"{profile.family.value}: unknown chat_format preset "
        f"{profile.chat_format_override!r}; check llama-cpp-python's "
        "LlamaChatCompletionHandlerRegistry."
    )


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_profile_streaming_policy_is_valid(profile: FamilyProfile) -> None:
    """Every profile carries a valid StreamingPolicy."""
    assert isinstance(profile.streaming_policy, StreamingPolicy)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_profile_output_format_is_valid(profile: FamilyProfile) -> None:
    """Every profile carries a valid OutputFormat."""
    assert isinstance(profile.output_format, OutputFormat)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_profile_has_at_least_one_match_signal(profile: FamilyProfile) -> None:
    """Every profile must declare at least one way for detect() to find it."""
    has_signal = bool(
        profile.template_markers
        or profile.name_patterns
        or profile.ref_patterns
        or profile.architectures
    )
    assert has_signal, (
        f"{profile.family.value}: no template_markers / name_patterns / "
        "ref_patterns / architectures declared; detect() will never match it."
    )


def test_detect_returns_profile_for_known_template_markers() -> None:
    """End-to-end: a Qwen3-shaped template resolves to the QWEN3 profile."""
    meta = {"chat_template": "{% for m in messages %}<tool_call>{{ m }}</tool_call>{% endfor %}"}
    profile = detect(meta)
    assert profile is not None
    assert profile.family is TemplateFamily.QWEN3


def test_detect_returns_none_when_nothing_matches() -> None:
    """Empty metadata + no ref must return None, not raise."""
    assert detect(None, ref=None) is None
    assert detect({}, ref=None) is None


def test_by_family_lookup_returns_expected_profile() -> None:
    """The TemplateFamily-keyed lookup table is built correctly."""
    hermes = registry().by_family(TemplateFamily.HERMES)
    assert hermes is not None
    assert hermes.chat_format_override == "chatml-function-calling"
    assert hermes.streaming_policy is StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE
