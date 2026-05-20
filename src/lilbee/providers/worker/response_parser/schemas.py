"""Response schemas keyed by detected ``TemplateFamily``."""

from __future__ import annotations

import functools
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


@functools.cache
def get_schemas() -> dict[TemplateFamily, ResponseSchema]:
    """Return the per-family response schemas, loading lazily on first call."""
    return _load_all()
