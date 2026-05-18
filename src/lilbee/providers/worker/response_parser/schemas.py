"""Response schemas keyed by detected ``ModelFamily``.

Schemas are declarative JSON dicts consumed by HuggingFace's
``transformers.utils.chat_parsing_utils.recursive_parse``. They describe how
to extract content / thinking / tool_calls from a model's text output. The
Qwen3-Coder and Gemma 4 entries are adapted from
``tests/utils/test_chat_parsing_utils.py`` in the ``transformers`` repository
(Apache-2.0, https://github.com/huggingface/transformers).
"""

from __future__ import annotations

from typing import Any

from lilbee.providers.worker.response_parser.families import ModelFamily

ResponseSchema = dict[str, Any]


# Qwen3 instruct family: emits ``<tool_call>{"name":..., "arguments":...}</tool_call>``
# blocks, optionally preceded by a ``<think>...</think>`` reasoning section.
_QWEN3_SCHEMA: ResponseSchema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {"type": "string", "x-regex": r"<think>\s*(.*?)\s*</think>"},
        "content": {
            "type": "string",
            "x-regex-substitutions": [
                [r"<think>.*?</think>", ""],
                [r"<tool_call>.*?</tool_call>", ""],
            ],
            "x-regex": r"^\s*(.*?)\s*$",
        },
        "tool_calls": {
            "x-regex-iterator": r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            "type": "array",
            "items": {
                "type": "object",
                "x-parser": "json",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
            },
        },
    },
}


# Qwen3-Coder family: uses XML for the tool body. ``<tool_call><function=NAME>
# <parameter=KEY>VALUE</parameter>...</function></tool_call>``. Adapted from
# the upstream HuggingFace test schema (see module docstring).
_QWEN3_CODER_SCHEMA: ResponseSchema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {"type": "string", "x-regex": r"<think>\s*(.*?)\s*</think>"},
        "content": {
            "type": "string",
            "x-regex-substitutions": [
                [r"<think>.*?</think>", ""],
                [r"<tool_call>.*?</tool_call>", ""],
            ],
            "x-regex": r"^\s*(.*?)\s*$",
        },
        "tool_calls": {
            "x-regex-iterator": r"<tool_call>\s*(.*?)\s*</tool_call>",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "x-regex": r"<function=(\w+)>"},
                    "arguments": {
                        "type": "object",
                        "x-regex-key-value": (
                            r"<parameter=(?P<key>\w+)>\n?(?P<value>.*?)\n?</parameter>"
                        ),
                        "additionalProperties": {
                            "x-parser": "json",
                            "x-parser-args": {"allow_non_json": True},
                        },
                    },
                },
            },
        },
    },
}


# Mistral family: emits ``[TOOL_CALLS] [{"name":..., "arguments":...}]`` after
# any text content. The tool_calls section is a single JSON array.
_MISTRAL_SCHEMA: ResponseSchema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": r"^(.*?)(?:\[TOOL_CALLS\]|$)"},
        "tool_calls": {
            "type": "array",
            "x-regex": r"\[TOOL_CALLS\]\s*(\[.*\])",
            "x-parser": "json",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
            },
        },
    },
}


# Gemma 4 family: emits ``<|tool_call>call:NAME{...}<tool_call|>`` blocks with
# the model's custom JSON dialect. ``x-parser: "gemma4-tool-call"`` is a
# built-in HF parser that translates the dialect to standard JSON. Adapted
# from the upstream HuggingFace test schema (see module docstring).
_GEMMA4_SCHEMA: ResponseSchema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {"type": "string", "x-regex": r"<\|channel\>thought\n(.*?)\<channel\|\>"},
        "content": {
            "type": "string",
            "x-regex": r"(?:<channel\|\>)?((?:(?!<\|tool_call\>).)*)",
        },
        "tool_calls": {
            "x-regex-iterator": r"<\|tool_call>(.*?)<tool_call\|>",
            "type": "array",
            "items": {
                "type": "object",
                "x-regex": r"call\:(?P<name>\w+)(?P<arguments>\{.*\})",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        "type": "object",
                        "x-parser": "gemma4-tool-call",
                        "additionalProperties": True,
                    },
                },
            },
        },
    },
}


SCHEMAS: dict[ModelFamily, ResponseSchema] = {
    ModelFamily.QWEN3: _QWEN3_SCHEMA,
    ModelFamily.QWEN3_CODER: _QWEN3_CODER_SCHEMA,
    ModelFamily.MISTRAL: _MISTRAL_SCHEMA,
    ModelFamily.GEMMA4: _GEMMA4_SCHEMA,
}
"""Response schemas indexed by detected model family. ``ModelFamily.UNKNOWN``
intentionally has no entry: when family detection cannot classify the loaded
model's template, tool extraction is skipped and the raw output is returned."""
