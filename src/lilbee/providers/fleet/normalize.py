"""Reshape an OpenAI tool conversation into strict user/assistant alternation.

Some GGUF chat templates (Mistral-Nemo, Cohere command-r) reject a standard
OpenAI tool exchange: they require plain user/assistant turns to alternate and
raise a Jinja exception on the ``tool`` role or on two same-role turns in a row.
:func:`to_alternating` rewrites the conversation into the shape those templates
accept. The fleet client learns which models need this by probing the live
template once, then applies it proactively before every such request.
"""

from __future__ import annotations

import json
from typing import Any

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool"
# A tool result carried as a user turn is labelled so the model reads it as a
# result rather than a fresh user instruction.
_TOOL_RESULT_PREFIX = "Tool result:"
# An assistant turn whose only payload was tool_calls becomes a short text note
# so the turn is non-empty (the templates reject empty content).
_TOOL_CALL_NOTE_PREFIX = "Calling tool"


def _content_text(content: Any) -> str:
    """Flatten a message ``content`` (string or OpenAI multipart list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _one_tool_call_note(call: Any) -> str:
    """Render one tool call as ``Calling name(args)``; empty when malformed."""
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict):
        return ""
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return ""
    arguments = fn.get("arguments")
    rendered_args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return f"{_TOOL_CALL_NOTE_PREFIX} {name}({rendered_args})"


def _tool_calls_note(tool_calls: Any) -> str:
    """Render an assistant message's ``tool_calls`` as a short text note."""
    if not isinstance(tool_calls, list):
        return ""
    return "\n".join(note for call in tool_calls if (note := _one_tool_call_note(call)))


def _assistant_text(message: dict[str, Any]) -> str:
    """Assistant turn text: its content plus a note for any tool_calls it made."""
    pieces = [_content_text(message.get("content")), _tool_calls_note(message.get("tool_calls"))]
    return "\n".join(piece for piece in pieces if piece)


def _to_turn(message: dict[str, Any]) -> tuple[str, str]:
    """Map one OpenAI message to a ``(user|assistant, text)`` pair.

    An assistant turn keeps its content and gains a note for any tool calls; a
    ``tool`` result becomes a labelled user turn; everything else, including a
    stray non-leading ``system`` message, becomes a plain user turn (so it can't
    re-open a system block or break alternation mid-conversation).
    """
    role = message.get("role")
    if role == _ASSISTANT_ROLE:
        return _ASSISTANT_ROLE, _assistant_text(message)
    if role == _TOOL_ROLE:
        result = _content_text(message.get("content"))
        return _USER_ROLE, f"{_TOOL_RESULT_PREFIX} {result}".strip()
    return _USER_ROLE, _content_text(message.get("content"))


def _append_or_merge(turns: list[dict[str, Any]], turn: dict[str, Any]) -> None:
    """Append a turn, or fold it into the previous turn when the role repeats."""
    if turns and turns[-1]["role"] == turn["role"]:
        turns[-1]["content"] = f"{turns[-1]['content']}\n{turn['content']}"
    else:
        turns.append(turn)


def to_alternating(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite an OpenAI tool conversation into strict user/assistant alternation.

    Keeps any leading system messages verbatim, maps each remaining message to a
    plain user or assistant turn (``tool`` results become labelled user turns;
    assistant tool calls become a text note), then merges consecutive same-role
    turns so the result alternates user/assistant after the system block. Every
    emitted turn has non-empty content, so a strict-alternation template accepts
    it.
    """
    leading_system: list[dict[str, Any]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == _SYSTEM_ROLE:
        leading_system.append(
            {"role": _SYSTEM_ROLE, "content": _content_text(messages[index].get("content"))}
        )
        index += 1

    turns: list[dict[str, Any]] = []
    for message in messages[index:]:
        role, text = _to_turn(message)
        if text:
            _append_or_merge(turns, {"role": role, "content": text})
    return leading_system + turns
