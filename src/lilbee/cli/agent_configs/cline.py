"""Cline settings block (https://docs.cline.bot/getting-started/installing-cline)."""

from __future__ import annotations

from typing import Any


def cline_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
) -> dict[str, Any]:
    """Return a Cline settings block pointing Cline's Anthropic provider at lilbee."""
    first_model = sorted(model_refs)[0] if model_refs else ""
    return {
        "apiProvider": "anthropic",
        "apiModelId": first_model,
        "anthropicBaseUrl": base_url,
        "apiKey": api_key,
    }
