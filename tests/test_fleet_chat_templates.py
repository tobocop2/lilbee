"""Tests for the fleet chat-template override registry."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.chat_templates import (
    _CHAT_TEMPLATE_OVERRIDES,
    _TEMPLATE_DIR,
    resolve_chat_template,
)


@pytest.mark.parametrize(
    ("model_ref", "filename"),
    [
        ("zai-org/glm-4-9b-chat-GGUF", "glm4.jinja"),
        ("NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "llama3-tools.jinja"),
        ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "llama3-tools.jinja"),
        ("internlm/internlm2_5-7b-chat-gguf", "internlm2.jinja"),
        ("allenai/Olmo-3-7B-Instruct-GGUF", "olmo3.jinja"),
        ("allenai/olmo3-7b-instruct", "olmo3.jinja"),
    ],
)
def test_resolve_returns_override_path_per_family(model_ref, filename):
    path = resolve_chat_template(model_ref)
    assert path == _TEMPLATE_DIR / filename
    assert path.is_file()


def test_resolve_matches_on_a_gguf_path_case_insensitively():
    path = resolve_chat_template("/models/THUDM/GLM-4-9B-Chat/glm-4-9b-Q8_0.gguf")
    assert path == _TEMPLATE_DIR / "glm4.jinja"


def test_resolve_returns_none_for_unlisted_ref():
    assert resolve_chat_template("unsloth/Qwen3-8B-Instruct-GGUF") is None


def test_every_registered_filename_ships_in_the_package():
    for _family, filename in _CHAT_TEMPLATE_OVERRIDES:
        assert (_TEMPLATE_DIR / filename).is_file()


def test_override_templates_declare_tools_for_the_supports_probe():
    """Each shipped override must reference tool tokens so the fleet's tool probe
    (``_TOOL_TEMPLATE_PATTERN``) and llama.cpp recognise it as tool-capable."""
    from lilbee.providers.fleet.provider import _TOOL_TEMPLATE_PATTERN

    seen = {filename for _family, filename in _CHAT_TEMPLATE_OVERRIDES}
    for filename in seen:
        text = (_TEMPLATE_DIR / filename).read_text(encoding="utf-8")
        assert _TOOL_TEMPLATE_PATTERN.search(text) is not None, filename
