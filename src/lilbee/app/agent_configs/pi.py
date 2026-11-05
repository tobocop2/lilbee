"""pi model definitions for the lilbee provider that the lilbee pi extension registers."""

from __future__ import annotations

from typing import Any

from lilbee.catalog import agent_model_id, display_label_for_ref

_OUTPUT_TOKEN_LIMIT = 8192
"""Per-response output cap reported to pi (it reserves this from the context)."""

_ZERO_COST = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}


def pi_models(model_refs: list[str], chat_ctx: int) -> list[dict[str, Any]]:
    """Return one complete pi model definition per ref, sorted by ref.

    pi fills no defaults for a provider an extension registers and fails on a
    missing window or output cap, so every field its model list reads is set.
    """
    return [
        {
            "id": agent_model_id(ref),
            "name": display_label_for_ref(ref),
            "reasoning": False,
            "input": ["text"],
            "cost": dict(_ZERO_COST),
            "contextWindow": chat_ctx,
            "maxTokens": max(1, min(chat_ctx // 2, _OUTPUT_TOKEN_LIMIT)),
        }
        for ref in sorted(model_refs)
    ]
