"""LiteLLM proxy config.yaml snippet (https://docs.litellm.ai/docs/proxy/configs)."""

from __future__ import annotations

import yaml


def litellm_config(
    *,
    base_url: str,
    api_key: str,
    model_refs: list[str],
) -> str:
    """Return a LiteLLM `config.yaml` snippet with one entry per chat model."""
    entries = [
        {
            "model_name": f"lilbee/{ref}",
            "litellm_params": {
                "model": f"openai/{ref}",
                "api_base": f"{base_url}/v1",
                "api_key": api_key,
            },
        }
        for ref in sorted(model_refs)
    ]
    return yaml.safe_dump({"model_list": entries}, sort_keys=False)
