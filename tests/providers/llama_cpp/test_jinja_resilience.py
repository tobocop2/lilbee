"""Coverage for the llama-cpp-python Jinja2ChatFormatter resilience patch."""

from __future__ import annotations

import jinja2
import pytest
from llama_cpp import llama_chat_format

import lilbee.providers.llama_cpp.jinja_resilience  # noqa: F401  (installs the patch)


def test_patch_is_applied() -> None:
    """The lilbee module flips the sentinel attribute on the class."""
    assert getattr(llama_chat_format.Jinja2ChatFormatter, "__lilbee_jinja_resilience__", False)


def test_unsupported_jinja_tag_no_longer_raises_at_init() -> None:
    """A SmolLM3-style ``{% generation %}`` tag must not blow up Llama load.

    Before the patch this raised TemplateSyntaxError from inside
    ``Llama.__init__`` even when the caller passed a chat_format override,
    making the GGUF entirely unloadable in lilbee.
    """
    template = "{% generation %}{{ messages }}{% endgeneration %}"
    formatter = llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token="</s>",
        bos_token="<s>",
    )
    # Environment is None because compile failed; the chat handler is a
    # no-op the override path will never call.
    assert formatter._environment is None


def test_valid_template_still_compiles() -> None:
    """The patch must not weaken compile for templates that are well-formed."""
    template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    formatter = llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token="</s>",
        bos_token="<s>",
    )
    assert formatter._environment is not None


def test_bad_template_still_raises_on_render() -> None:
    """Deferred failure: if the override path doesn't apply, render still fails."""
    template = "{% generation %}{{ messages }}{% endgeneration %}"
    formatter = llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token="</s>",
        bos_token="<s>",
    )
    with pytest.raises((jinja2.exceptions.TemplateSyntaxError, AttributeError)):
        formatter(messages=[{"role": "user", "content": "hi"}])
