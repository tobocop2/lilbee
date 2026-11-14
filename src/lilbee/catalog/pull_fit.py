"""The memory a safetensors chat model needs, worked out from the Hub before it downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lilbee.catalog import hf_client
from lilbee.providers.fleet.model_memory import (
    kv_per_token_from_config,
    safetensors_chat_estimate,
    weights_from_header,
)

_GIB = 1024**3
_CONFIG_TEXT_KEY = "text_config"
_CONFIG_TRAINED_KEY = "max_position_embeddings"


@dataclass(frozen=True)
class PullFit:
    """The memory one chat instance needs, and how far to trust the figure."""

    device_bytes: int
    ctx: int
    size_known: bool
    upper_bound: bool
    """True when a sliding window means the real cache is smaller than the figure."""

    def text(self) -> str:
        """One sentence for the user: the figure, or the reserve with the reason it is a guess."""
        gib = self.device_bytes / _GIB
        if not self.size_known:
            return f"The size of this model is unknown, so {gib:.1f} GiB is reserved for it."
        qualifier = "at most " if self.upper_bound else ""
        return f"This model needs {qualifier}{gib:.1f} GiB at a context of {self.ctx} tokens."


def _trained_ctx(config: dict[str, Any]) -> int | None:
    """The context the model was trained for, from its fetched ``config.json``."""
    section = config.get(_CONFIG_TEXT_KEY)
    for keys in (section, config):
        value = keys.get(_CONFIG_TRAINED_KEY) if isinstance(keys, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def chat_pull_fit(
    repo: str, configured_ctx: int, *, token: str | bool | None = None, slots: int = 1
) -> PullFit:
    """The fit of *repo* as a chat model at *configured_ctx*, capped by its trained context.

    Reads the Hub twice: one request for the tensor counts and one for ``config.json``.
    A read that fails gives an unknown size, never an error.
    """
    config = hf_client.read_model_config(repo, token=token)
    trained = _trained_ctx(config) if config is not None else None
    ctx = configured_ctx if trained is None else min(configured_ctx, trained)
    kv = kv_per_token_from_config(config) if config is not None else None
    weights = weights_from_header(hf_client.read_parameter_count(repo, token=token))
    estimate = safetensors_chat_estimate(weights, kv, ctx, slots)
    return PullFit(
        device_bytes=estimate.device_bytes,
        ctx=ctx,
        size_known=estimate.size_known,
        upper_bound=kv.upper_bound if kv is not None else False,
    )
