"""Soften llama-cpp-python's eager Jinja compile of embedded chat templates.

``Llama.__init__`` registers a :class:`Jinja2ChatFormatter` for every chat
template found in the GGUF metadata, and that constructor eagerly compiles
the template via :meth:`jinja2.Environment.from_string`. If the template
uses a tag jinja2 can't parse (SmolLM3's ``{% generation %}``, future
HF templates using ``{% endgeneration %}``, etc.) the whole model load
explodes with :class:`jinja2.exceptions.TemplateSyntaxError`, even when
the caller passes an explicit ``chat_format`` preset that would skip the
embedded template entirely.

The fix is a one-line patch: swallow the syntax error during init and
register a no-op handler. If the bad template ever does get called at
prompt-render time (i.e. the caller did NOT override ``chat_format``),
the original exception fires there instead, where it's actionable.

This file applies the patch at import time. Importing it from
``lilbee.providers.llama_cpp`` keeps the patch in scope for every load
path without callers having to remember.

Regression coverage
-------------------
This is a monkey-patch on a third-party library; it can silently no-op if
upstream changes its API. The safety net is two test modules:

- ``tests/providers/llama_cpp/test_jinja_resilience.py`` exercises the
  patched behavior (bad-template input does not raise at construction,
  valid templates still compile, deferred failure fires at render time).
- ``tests/providers/llama_cpp/test_jinja_resilience_upstream_api.py``
  asserts the upstream class still exists, still has the
  ``from_string(template)`` eager-compile call site we patch, and still
  exposes the ``__init__`` signature we patched. If upstream renames the
  class, drops the eager compile, or changes ``__init__`` parameters,
  the upstream-api test fails loudly so the monkey-patch can be
  re-evaluated rather than silently going stale.
"""

from __future__ import annotations

import logging

import jinja2
from llama_cpp import llama_chat_format

log = logging.getLogger(__name__)

_PATCHED_FLAG = "__lilbee_jinja_resilience__"

_original_init = llama_chat_format.Jinja2ChatFormatter.__init__


def _resilient_init(  # type: ignore[no-untyped-def]
    self,
    template,
    eos_token,
    bos_token,
    add_generation_prompt=True,
    stop_token_ids=None,
):
    """Same as upstream init, but a TemplateSyntaxError leaves _environment None."""
    self.template = template
    self.eos_token = eos_token
    self.bos_token = bos_token
    self.add_generation_prompt = add_generation_prompt
    self.stop_token_ids = set(stop_token_ids) if stop_token_ids is not None else None
    try:
        self._environment = llama_chat_format.ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.template)
    except jinja2.exceptions.TemplateSyntaxError as exc:
        # Defer the failure. If no chat_format override applies, the
        # __call__ path below will raise the same error at prompt-render
        # time, where the failure is actionable.
        log.debug("Deferred Jinja compile of embedded template: %s", exc)
        self._environment = None


def install() -> None:
    """Replace Jinja2ChatFormatter.__init__ once, idempotent."""
    if getattr(llama_chat_format.Jinja2ChatFormatter, _PATCHED_FLAG, False):
        return
    llama_chat_format.Jinja2ChatFormatter.__init__ = _resilient_init  # type: ignore[method-assign]
    setattr(llama_chat_format.Jinja2ChatFormatter, _PATCHED_FLAG, True)


install()
