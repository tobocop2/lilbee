"""Response schemas keyed by detected ``TemplateFamily``.

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
import logging
from importlib import resources
from typing import Any

from lilbee.providers.worker.response_parser.families import TemplateFamily

log = logging.getLogger(__name__)

ResponseSchema = dict[str, Any]

_SCHEMA_FILES: dict[TemplateFamily, str] = {
    TemplateFamily.QWEN3: "qwen3.json",
    TemplateFamily.QWEN3_CODER: "qwen3_coder.json",
    TemplateFamily.MISTRAL: "mistral.json",
    TemplateFamily.GEMMA4: "gemma4.json",
    TemplateFamily.COHERE: "cohere.json",
    TemplateFamily.ERNIE: "ernie.json",
    TemplateFamily.GPT_OSS: "gpt_oss.json",
    TemplateFamily.SMOLLM: "smollm.json",
    TemplateFamily.HERMES: "hermes.json",
    TemplateFamily.DEEPSEEK_V31: "deepseek_v31.json",
    TemplateFamily.GRANITE: "granite.json",
    TemplateFamily.PHI4MINI: "phi4mini.json",
    TemplateFamily.FUNCTIONARY_V3: "functionary_v3.json",
    TemplateFamily.LLAMA3: "llama3.json",
    TemplateFamily.GLM46: "glm46.json",
    TemplateFamily.GLM47: "glm47.json",
    TemplateFamily.KIMI_K2: "kimi_k2.json",
    TemplateFamily.INTERNLM2: "internlm2.json",
    TemplateFamily.OLMO3: "olmo3.json",
    TemplateFamily.LFM2: "lfm2.json",
}


def _load(filename: str) -> ResponseSchema | None:
    """Read one schema JSON file; return ``None`` on missing/invalid file."""
    try:
        text = resources.files(__package__).joinpath("schemas", filename).read_text("utf-8")
        schema: ResponseSchema = json.loads(text)
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.warning("Could not load response schema %r: %s", filename, exc)
        return None
    return schema


def _load_all() -> dict[TemplateFamily, ResponseSchema]:
    """Load every shipped schema; broken files degrade to no extraction for that family."""
    out: dict[TemplateFamily, ResponseSchema] = {}
    for family, filename in _SCHEMA_FILES.items():
        loaded = _load(filename)
        if loaded is not None:
            out[family] = loaded
    return out


SCHEMAS: dict[TemplateFamily, ResponseSchema] = _load_all()
