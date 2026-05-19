"""Response schemas keyed by detected ``ModelFamily``.

Each schema lives as a JSON file in ``schemas/`` next to this module and is
consumed by ``transformers.utils.chat_parsing_utils.recursive_parse``. New
families: drop a ``schemas/{family}.json`` file and add the enum value +
detection marker. No Python edits to the schema bodies themselves.

Long-term: llama.cpp's ``common/chat-auto-parser`` introspects the GGUF
chat template and synthesises a PEG parser at runtime, so per-family
schemas disappear entirely. That code is not reachable from
``llama-cpp-python`` today (the autoparser lives in llama.cpp's ``common/``
directory, outside ``libllama``'s C ABI). When a binding appears,
``src/lilbee/providers/worker/response_parser/`` gets deleted in one
commit. Until then we ship the small set below.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from lilbee.providers.worker.response_parser.families import ModelFamily

ResponseSchema = dict[str, Any]

_SCHEMA_FILES: dict[ModelFamily, str] = {
    ModelFamily.QWEN3: "qwen3.json",
    ModelFamily.QWEN3_CODER: "qwen3_coder.json",
    ModelFamily.MISTRAL: "mistral.json",
    ModelFamily.GEMMA4: "gemma4.json",
    ModelFamily.COHERE: "cohere.json",
    ModelFamily.ERNIE: "ernie.json",
    ModelFamily.GPT_OSS: "gpt_oss.json",
    ModelFamily.SMOLLM: "smollm.json",
    ModelFamily.HERMES: "hermes.json",
    ModelFamily.DEEPSEEK_V31: "deepseek_v31.json",
    ModelFamily.GRANITE: "granite.json",
    ModelFamily.PHI4MINI: "phi4mini.json",
    ModelFamily.FUNCTIONARY_V3: "functionary_v3.json",
    ModelFamily.LLAMA3: "llama3.json",
    ModelFamily.GLM46: "glm46.json",
    ModelFamily.GLM47: "glm47.json",
    ModelFamily.KIMI_K2: "kimi_k2.json",
    ModelFamily.INTERNLM2: "internlm2.json",
    ModelFamily.OLMO3: "olmo3.json",
    ModelFamily.LFM2: "lfm2.json",
}


def _load(filename: str) -> ResponseSchema:
    """Read one schema JSON file from the ``schemas/`` package directory."""
    text = resources.files(__package__).joinpath("schemas", filename).read_text("utf-8")
    schema: ResponseSchema = json.loads(text)
    return schema


SCHEMAS: dict[ModelFamily, ResponseSchema] = {
    family: _load(filename) for family, filename in _SCHEMA_FILES.items()
}
