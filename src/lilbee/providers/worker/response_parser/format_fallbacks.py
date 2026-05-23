"""Bare-JSON tool-call extractor used for :data:`OutputFormat.BARE_JSON` and ``DUAL``.

OpenAI-compatible clients prompt every tool-trained model via the standard
``tools`` parameter, which elicits a bare ``{"name": ..., "arguments": ...}``
JSON object regardless of the family's native wire shape. The extractor
walks the text, uses :class:`json.JSONDecoder` to consume each candidate
object exactly the way ``json.loads`` would, and keeps the ones that match
the tool-call envelope. Using the stdlib decoder rather than a bracket
regex handles nested objects, escaped strings, and unicode the way the
JSON grammar requires.
"""

from __future__ import annotations

import json
import logging

from lilbee.providers.worker.transport import ToolCall

log = logging.getLogger(__name__)

_DECODER = json.JSONDecoder()
_NAME_KEY = "name"
_ARGUMENTS_KEY = "arguments"
# Some tool-trained families (Mistral-Nemo, others) emit the call argument map
# under "parameters" instead of the OpenAI-standard "arguments". Accept either.
_PARAMETERS_KEY = "parameters"


def bare_json_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Yield every ``{"name": ..., "arguments"|"parameters": ...}`` object in *text*."""
    out: list[ToolCall] = []
    for obj in _iter_json_objects(text):
        name = obj.get(_NAME_KEY)
        if not isinstance(name, str) or not name:
            continue
        arguments = obj.get(_ARGUMENTS_KEY)
        if arguments is None:
            arguments = obj.get(_PARAMETERS_KEY)
        out.append(ToolCall(id="", name=name, arguments=_arguments_to_string(arguments)))
    return tuple(out)


def split_content_at_bare_json(text: str) -> str:
    """Prefix of *text* up to the first ``{"name": ..., "arguments": ...}`` object."""
    for offset, _ in _iter_json_objects_with_offsets(text):
        return text[:offset]
    return text


def _iter_json_objects(text: str) -> list[dict[str, object]]:
    return [obj for _, obj in _iter_json_objects_with_offsets(text)]


def _iter_json_objects_with_offsets(text: str) -> list[tuple[int, dict[str, object]]]:
    """Scan *text* for ``{...}`` objects whose top-level keys include ``name``."""
    found: list[tuple[int, dict[str, object]]] = []
    idx = 0
    end = len(text)
    while idx < end:
        brace = text.find("{", idx)
        if brace < 0:
            break
        try:
            value, consumed = _DECODER.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(value, dict) and _NAME_KEY in value:
            found.append((brace, value))
        idx = max(consumed, brace + 1)
    return found


def _arguments_to_string(arguments: object) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, separators=(",", ":"))
    except (TypeError, ValueError):
        log.debug("could not serialise bare-JSON tool-call arguments: %r", arguments)
        return "{}"
