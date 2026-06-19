"""Tests for per-architecture chat-template overrides and the chat spec."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.adapters import ROLE_SPECS, chat_spec
from lilbee.providers.fleet.chat_template_overrides import (
    _ARCH_TEMPLATE_OVERRIDES,
    _CHAT_TEMPLATES_DIR,
    chat_template_override,
    template_is_tool_aware,
)
from lilbee.providers.roles import WorkerRole

_TOOL_AWARE = "{% if tools %}{{ tools | tojson }}{% endif %}"
_PLAIN = "{% for m in messages %}{{ m.content }}{% endfor %}"
# A registered override arch + the filename it maps to, picked once so the tests
# stay correct as the registry grows.
_OVERRIDE_ARCH, _OVERRIDE_FILE = next(iter(_ARCH_TEMPLATE_OVERRIDES.items()))


def test_template_is_tool_aware_detects_tool_block() -> None:
    assert template_is_tool_aware(_TOOL_AWARE) is True


def test_template_is_tool_aware_false_for_plain_and_none() -> None:
    assert template_is_tool_aware(_PLAIN) is False
    assert template_is_tool_aware(None) is False


def test_override_applies_to_registered_arch_with_tool_less_template() -> None:
    meta = {"architecture": _OVERRIDE_ARCH, "chat_template": _PLAIN}
    path = chat_template_override(meta)
    assert path is not None
    assert path.name == _OVERRIDE_FILE
    assert path.is_file()


def test_override_skips_registered_arch_when_embedded_template_is_already_tool_aware() -> None:
    # A future quant that ships a good template must not be overridden.
    meta = {"architecture": _OVERRIDE_ARCH, "chat_template": _TOOL_AWARE}
    assert chat_template_override(meta) is None


def test_override_skips_unregistered_arch() -> None:
    # arch 'llama' covers llama3 (tool-aware embedded template) and must stay untouched.
    meta = {"architecture": "llama", "chat_template": _PLAIN}
    assert chat_template_override(meta) is None


def test_override_none_for_empty_meta() -> None:
    assert chat_template_override(None) is None
    assert chat_template_override({}) is None


@pytest.mark.parametrize("filename", sorted(_ARCH_TEMPLATE_OVERRIDES.values()))
def test_vendored_templates_exist_and_are_tool_aware(filename: str) -> None:
    # Guards against a template being deleted or replaced with a tool-less one:
    # the whole point of the override is that it carries a usable tools block.
    path = _CHAT_TEMPLATES_DIR / filename
    assert path.is_file(), f"missing vendored template {filename}"
    assert template_is_tool_aware(path.read_text(encoding="utf-8"))


def test_chat_spec_adds_template_file_for_override_arch() -> None:
    spec = chat_spec({"architecture": _OVERRIDE_ARCH, "chat_template": _PLAIN})
    assert "--chat-template-file" in spec.extra_args
    # --jinja is still required so llama-server renders the override and parses
    # native tool calls from it.
    assert "--jinja" in spec.extra_args
    idx = spec.extra_args.index("--chat-template-file")
    assert spec.extra_args[idx + 1].endswith(_OVERRIDE_FILE)


def test_chat_spec_is_base_for_non_override_model() -> None:
    base = ROLE_SPECS[WorkerRole.CHAT]
    assert chat_spec({"architecture": "llama", "chat_template": _TOOL_AWARE}) == base
    assert chat_spec(None) == base


# The fleet warms each chat server with a single no-tools message. A template
# with an UNGUARDED ``tojson`` (or other reference to an undefined ``tools``) on
# that path crashes llama-server's jinja engine on load -- the exact failure
# that disqualified olmo3. Render every vendored template on that payload, and
# on a tools payload, to lock the guard in (HF renders these with jinja2 too).
_WARMUP_MESSAGES = [{"role": "user", "content": "warm"}]
_TOOLS_PAYLOAD = [
    {
        "type": "function",
        "function": {"name": "search", "description": "search", "parameters": {}},
    }
]


@pytest.mark.parametrize("filename", sorted(_ARCH_TEMPLATE_OVERRIDES.values()))
def test_vendored_templates_render_without_raising(filename: str) -> None:
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    template = env.from_string((_CHAT_TEMPLATES_DIR / filename).read_text(encoding="utf-8"))
    # No-tools warmup must not raise (the olmo3 unguarded-tojson failure class).
    template.render(messages=_WARMUP_MESSAGES, add_generation_prompt=True)
    # The tools path must also render, since that is what exercises the tojson.
    template.render(messages=_WARMUP_MESSAGES, tools=_TOOLS_PAYLOAD, add_generation_prompt=True)
