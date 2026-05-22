"""Coverage for the family-profile registry: presence, ordering, lookup, detect()."""

from __future__ import annotations

import re

import pytest

from lilbee.providers.chat_format import LlamaCppChatFormatPreset
from lilbee.providers.families import ALL_PROFILES, detect, registry
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily
from lilbee.providers.worker.response_parser.schemas import get_schemas


def _upstream_chat_format_presets() -> frozenset[str]:
    """Return the chat-format handler keys the installed llama-cpp-python registers."""
    from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry

    return frozenset(LlamaChatCompletionHandlerRegistry()._chat_handlers.keys())


def test_all_profiles_loaded() -> None:
    """Every family schema in response_parser/schemas/ has a matching profile."""
    schema_families = {f for f in get_schemas() if f is not TemplateFamily.UNKNOWN}
    profile_families = {p.family for p in ALL_PROFILES}
    missing = schema_families - profile_families
    assert not missing, f"profiles missing for: {sorted(f.value for f in missing)}"


def test_registry_singleton_is_cached() -> None:
    """Repeated registry() calls return the same instance (lru_cache contract)."""
    assert registry() is registry()


@pytest.mark.parametrize("preset", list(LlamaCppChatFormatPreset), ids=lambda p: p.value)
def test_chat_format_preset_resolves_in_installed_llama_cpp(
    preset: LlamaCppChatFormatPreset,
) -> None:
    """Every preset in :class:`LlamaCppChatFormatPreset` resolves in upstream."""
    upstream = _upstream_chat_format_presets()
    assert preset.value in upstream, (
        f"Preset {preset.value!r} not registered by the installed llama-cpp-python; "
        "update the enum or pin a build that ships this handler."
    )


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_profile_chat_format_override_uses_typed_enum(profile: FamilyProfile) -> None:
    """A profile's chat_format_override must be a member of :class:`LlamaCppChatFormatPreset`."""
    if profile.chat_format_override is None:
        return
    assert isinstance(profile.chat_format_override, LlamaCppChatFormatPreset)


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


def test_detect_returns_functionary_v3_by_ref_pattern() -> None:
    """``ref_patterns`` matches when neither template nor name fire (functionary-v3 case)."""
    profile = detect(
        {"name": "Llama-3.1-8B-Instruct"},
        ref="meetkai/functionary-small-v3.2-GGUF",
    )
    assert profile is not None
    assert profile.family is TemplateFamily.FUNCTIONARY_V3


def test_by_family_lookup_returns_expected_profile() -> None:
    """The TemplateFamily-keyed lookup table is built correctly."""
    hermes = registry().by_family(TemplateFamily.HERMES)
    assert hermes is not None
    assert hermes.chat_format_override is LlamaCppChatFormatPreset.CHATML_FUNCTION_CALLING
    assert hermes.streaming_policy is StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE


def test_by_family_unknown_returns_none() -> None:
    """``by_family`` returns None for a family that has no registered profile."""
    assert registry().by_family(TemplateFamily.UNKNOWN) is None


def test_name_pattern_match_uses_general_name_field() -> None:
    """``general.name``-style metadata is normalised to ``"name"`` and feeds name_patterns."""
    profile = detect({"name": "Hermes 3 Llama 3.1 8B"}, ref=None)
    assert profile is not None
    assert profile.family is TemplateFamily.HERMES


def test_architecture_fallback_matches_internlm2() -> None:
    """When neither template nor name fire, ``architectures`` is the last-chance hit."""
    profile = detect({"architecture": "internlm2"}, ref=None)
    assert profile is not None
    assert profile.family is TemplateFamily.INTERNLM2


def test_match_order_is_most_specific_first() -> None:
    """Qwen3-Coder sits before plain Qwen3 so coder GGUFs resolve to the coder profile."""
    order = [p.family for p in ALL_PROFILES]
    assert order.index(TemplateFamily.QWEN3_CODER) < order.index(TemplateFamily.QWEN3)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.family.value)
def test_every_compiled_pattern_is_anchored_to_a_search_method(
    profile: FamilyProfile,
) -> None:
    """``name_patterns`` and ``ref_patterns`` must be ``re.Pattern`` (not raw strings)."""
    for pattern in (*profile.name_patterns, *profile.ref_patterns):
        assert isinstance(pattern, re.Pattern)


def test_all_profiles_view_returns_full_match_order() -> None:
    """``FamilyRegistry.all_profiles`` exposes the full match order tuple."""
    assert registry().all_profiles == ALL_PROFILES


def test_discover_profiles_raises_on_missing_module(monkeypatch) -> None:
    """If ``_MATCH_ORDER`` lists a module that doesn't exist, discovery fails fast."""
    import lilbee.providers.families as families_pkg

    monkeypatch.setattr(families_pkg, "_MATCH_ORDER", ("not_a_real_profile",))
    with pytest.raises(RuntimeError, match="not importable"):
        families_pkg._discover_profiles()


def test_discover_profiles_raises_on_extra_module(monkeypatch) -> None:
    """A sibling profile module not listed in ``_MATCH_ORDER`` fails discovery."""
    import lilbee.providers.families as families_pkg

    # Drop one real module from the order; the leftover triggers the extras check.
    short_order = tuple(name for name in families_pkg._MATCH_ORDER if name != "qwen3")
    monkeypatch.setattr(families_pkg, "_MATCH_ORDER", short_order)
    with pytest.raises(RuntimeError, match="absent from _MATCH_ORDER"):
        families_pkg._discover_profiles()


def test_discover_profiles_raises_on_non_profile_value(monkeypatch) -> None:
    """If a profile module's ``PROFILE`` attribute isn't a FamilyProfile, fail loudly."""
    import lilbee.providers.families as families_pkg
    import lilbee.providers.families.qwen3 as qwen3_mod

    monkeypatch.setattr(qwen3_mod, "PROFILE", "not-a-profile")
    with pytest.raises(TypeError, match="must be a FamilyProfile"):
        families_pkg._discover_profiles()
