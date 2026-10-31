"""Inference engine adapters for the fleet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.providers.fleet.engines.base import (
    EngineAdapter,
    EngineName,
    LaunchRequest,
)
from lilbee.providers.fleet.engines.llama import LLAMA

if TYPE_CHECKING:
    from lilbee.core.config import Config


def select_engine(config: Config) -> EngineAdapter:
    """The engine adapter that *config* selects."""
    return LLAMA


__all__ = [
    "LLAMA",
    "EngineAdapter",
    "EngineName",
    "LaunchRequest",
    "select_engine",
]
