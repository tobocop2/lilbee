"""The memory a safetensors chat model needs, worked out from the Hub before it downloads."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.catalog import hf_client
from lilbee.providers.fleet.model_memory import (
    kv_per_token_from_config,
    safetensors_chat_estimate,
    weights_from_header,
)
from lilbee.providers.fleet.planning import planned_chat_ctx

if TYPE_CHECKING:
    from lilbee.providers.fleet.engines.base import EngineAdapter

_GIB = 1024**3
_CONFIG_FILE = "config.json"


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


def _launch_ctx(repo: str, config: dict[str, Any] | None, engine: EngineAdapter) -> int:
    """The context a launch of *repo* on *engine* is planned at, worked out as the planner does.

    The engine reads the window from the ``config.json`` in a model folder, so the
    fetched config goes into a scratch folder named for the repo.
    """
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch) / repo.rsplit("/", 1)[-1]
        folder.mkdir()
        if config is not None:
            (folder / _CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
        return planned_chat_ctx(folder, engine.model_meta(folder))


def chat_pull_fit(
    repo: str, engine: EngineAdapter, *, token: str | bool | None = None, slots: int = 1
) -> PullFit:
    """The fit of *repo* as a chat model at the context *engine* launches it with.

    Reads the Hub twice: one request for the tensor counts and one for ``config.json``.
    A read that fails gives an unknown size, never an error.
    """
    config = hf_client.read_model_config(repo, token=token)
    ctx = _launch_ctx(repo, config, engine)
    kv = kv_per_token_from_config(config) if config is not None else None
    weights = weights_from_header(hf_client.read_parameter_count(repo, token=token))
    estimate = safetensors_chat_estimate(weights, kv, ctx, slots)
    return PullFit(
        device_bytes=estimate.device_bytes,
        ctx=ctx,
        size_known=estimate.size_known,
        upper_bound=kv.upper_bound if kv is not None else False,
    )
