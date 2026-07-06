"""Non-destructive deep-merge of a lilbee config fragment into a user's agent config."""

from __future__ import annotations

from typing import Any

LILBEE_PROVIDER_KEY = "lilbee"


def deep_merge(base: dict[str, Any], fragment: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *fragment* into *base*; fragment wins on leaf conflicts,
    dict values merge key-by-key so unrelated keys in *base* survive."""
    for key, value in fragment.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            deep_merge(existing, value)
        else:
            base[key] = value
    return base


def prune_lilbee(config: dict[str, Any], container_key: str) -> None:
    """Remove the lilbee entry from ``config[container_key]`` if present (for --no-mcp)."""
    container = config.get(container_key)
    if isinstance(container, dict):
        container.pop(LILBEE_PROVIDER_KEY, None)
