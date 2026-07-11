"""Per-model default generation settings.

A frozen snapshot of a model's default generation parameters, applied to config
via ``cfg.apply_model_defaults`` and read back through ``cfg.model_defaults``
(e.g. the reasoning layer's per-model ``max_reasoning_chars`` override).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefaults:
    """Frozen snapshot of a model's default generation parameters."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None
    max_reasoning_chars: int | None = None
