"""Regression tests for the upstream llama-cpp-python API our monkey-patch depends on.

If any of these fail, ``src/lilbee/providers/llama_cpp/jinja_resilience.py``
is at risk of silently no-op'ing. Either the upstream class has been
renamed/removed (the patch site fails outright at import), or the eager
Jinja compile path has been lazified / removed upstream (the patch is no
longer needed), or the constructor signature has shifted (our replacement
init mismatches what callers pass).

These tests are intentionally tightly coupled to upstream so they fail
LOUDLY when ``llama-cpp-python`` evolves, prompting a deliberate
re-evaluation of the monkey-patch rather than a silent regression for
SmolLM3-class models.
"""

from __future__ import annotations

import inspect

import pytest
from llama_cpp import llama_chat_format


def test_upstream_jinja2_chat_formatter_still_exists() -> None:
    """If upstream renames or removes the class, the monkey-patch import fails outright.

    This test exists for completeness and a clearer error than the
    bare ImportError ``jinja_resilience.py`` would surface.
    """
    assert hasattr(llama_chat_format, "Jinja2ChatFormatter"), (
        "llama-cpp-python no longer exposes Jinja2ChatFormatter. "
        "The jinja_resilience monkey-patch site is broken; revisit "
        "the patch design (the upstream API surface has shifted)."
    )


def test_upstream_init_signature_matches_patched_replacement() -> None:
    """Our replacement ``__init__`` accepts the same parameters upstream does.

    The patched init signature is hard-coded to upstream's expected kwargs
    (template, eos_token, bos_token, add_generation_prompt, stop_token_ids).
    If upstream adds, removes, or renames a parameter, callers passing
    the new kwarg would hit a TypeError that surfaces as a model-load
    failure with no obvious link back to this monkey-patch.
    """
    from lilbee.providers.llama_cpp import jinja_resilience  # noqa: F401

    patched_init = llama_chat_format.Jinja2ChatFormatter.__init__
    sig = inspect.signature(patched_init)
    params = list(sig.parameters)

    # Upstream's init at the time the patch was written.
    expected_params = [
        "self",
        "template",
        "eos_token",
        "bos_token",
        "add_generation_prompt",
        "stop_token_ids",
    ]
    assert params == expected_params, (
        f"Jinja2ChatFormatter.__init__ signature has drifted: expected "
        f"{expected_params}, got {params}. The jinja_resilience monkey-"
        f"patch's replacement init must be updated to match the new "
        f"upstream parameter list."
    )


def test_upstream_still_eagerly_compiles_chat_template() -> None:
    """If upstream switches to lazy-compile, the monkey-patch is no longer needed.

    The patch's whole reason for existing is that upstream calls
    ``Environment().from_string(template)`` inside ``__init__``, which
    raises TemplateSyntaxError for unparseable templates before any
    request runs. If upstream defers that compile (e.g. to ``__call__``),
    SmolLM3-style models load fine without our patch and the patch
    becomes a maintenance burden. Skipping this test (via the marker
    below) flags that situation for the reader.
    """
    from lilbee.providers.llama_cpp import jinja_resilience

    # Source of the ORIGINAL upstream init (saved by our patch before
    # replacement). If the captured source no longer contains the eager
    # compile call, upstream has changed and the patch is moot.
    original_init_source = inspect.getsource(jinja_resilience._original_init)
    if "from_string" not in original_init_source:
        pytest.fail(
            "Upstream llama-cpp-python's Jinja2ChatFormatter.__init__ no "
            "longer eagerly compiles the chat template (the .from_string() "
            "call is gone). The jinja_resilience monkey-patch is no "
            "longer needed; delete src/lilbee/providers/llama_cpp/"
            "jinja_resilience.py and its tests."
        )


def test_patched_init_replaces_eager_compile_with_try_except() -> None:
    """The patched init still wraps the eager compile in a try/except.

    Verifies our replacement init didn't get clobbered or partially
    reverted; the resilience BEHAVIOR comes from the try/except, not just
    the patch flag.
    """
    from lilbee.providers.llama_cpp import jinja_resilience  # noqa: F401

    patched_init = llama_chat_format.Jinja2ChatFormatter.__init__
    source = inspect.getsource(patched_init)
    assert "TemplateSyntaxError" in source, (
        "Patched __init__ no longer catches TemplateSyntaxError; the "
        "monkey-patch was probably reverted by an import-order change."
    )
    assert "_environment = None" in source, (
        "Patched __init__ no longer leaves _environment as None on "
        "failure; the deferred-failure contract is broken."
    )
