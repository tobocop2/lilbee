"""Per-architecture chat-template overrides for tool-aware fleet serving.

The fleet serves chat with ``llama-server --jinja``, which renders the GGUF's
*embedded* chat template and parses native tool calls from it. Some published
quants ship a template with no tools block, so the server cannot emit tool
calls and lilbee declines to offer tools at all (``supports_tools`` is False).

For architectures with a known-good upstream template we vendor a copy under
``chat_templates/`` and point llama-server at it with ``--chat-template-file``,
but only when the embedded template is *not* already tool-aware, so a quant
that ships a good template (e.g. arch ``llama`` for Llama 3.1) is never
overridden. The same resolver feeds both seams: the launch argv and the
``supports_tools`` decision, so a model is offered tools exactly when it is
served a tool-aware template.
"""

from __future__ import annotations

import re
from pathlib import Path

# A jinja chat template flags tool support by referencing one of these names as
# an identifier inside a ``{% ... %}`` / ``{{ ... }}`` block (not free-text prose).
_TOOL_TEMPLATE_PATTERN = re.compile(r"\{[%{][^}]*\b(?:tools|tool_calls|functions|function_calls)\b")

_CHAT_TEMPLATES_DIR = Path(__file__).parent / "chat_templates"

# GGUF ``general.architecture`` -> vendored tool-aware template filename. Keyed
# by architecture so it covers every quant of that arch; the embedded-template
# guard in :func:`chat_template_override` keeps it from replacing a template
# that is already tool-aware. Each vendored template is the model's own upstream
# template (tool-aware, with any ``tojson`` guarded so the no-tools warmup call
# does not crash llama-server's jinja engine).
#
# An arch belongs here only once it is verified end-to-end: the override must
# make llama-server emit a *structured* tool call the agent can dispatch. An
# arch whose template offers tools but whose tool-call output llama.cpp does not
# parse (so the model emits the call as prose and the agent never dispatches)
# must NOT be added -- offering tools we cannot parse is worse than declining
# them, because the model then hallucinates instead of failing cleanly.
_ARCH_TEMPLATE_OVERRIDES: dict[str, str] = {
    "lfm2": "lfm2.jinja",
}


def template_is_tool_aware(template: str | None) -> bool:
    """True if a jinja chat template references tools/tool_calls/functions in a tag."""
    return isinstance(template, str) and _TOOL_TEMPLATE_PATTERN.search(template) is not None


def chat_template_override(meta: dict[str, str] | None) -> Path | None:
    """Vendored tool-aware chat template for *meta*'s architecture, or ``None``.

    Returns a path only when the architecture has a registered override and the
    GGUF's embedded template is not already tool-aware, so a good embedded
    template is never replaced. ``None`` when no override applies.
    """
    if not meta:
        return None
    filename = _ARCH_TEMPLATE_OVERRIDES.get(meta.get("architecture", ""))
    if filename is None:
        return None
    if template_is_tool_aware(meta.get("chat_template")):
        return None
    path = _CHAT_TEMPLATES_DIR / filename
    return path if path.is_file() else None
