"""A test patches an engine method on its class, so the undo leaves the engine as it was."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX

_PATCHED_METHODS = [
    pytest.param(LLAMA, "pin_record", id="llama-pin_record"),
    pytest.param(LLAMA, "swap_extras", id="llama-swap_extras"),
    pytest.param(LLAMA, "request_profile", id="llama-request_profile"),
    pytest.param(MISTRAL, "pin_record", id="mistral-pin_record"),
    pytest.param(MLX, "binary", id="mlx-binary"),
]


def _stand_in(*_args: Any) -> str:
    return "stand-in"


@pytest.mark.parametrize(("engine", "method"), _PATCHED_METHODS)
def test_a_class_patch_reaches_an_engine_whose_earlier_test_patched_it(
    monkeypatch: pytest.MonkeyPatch, engine: Any, method: str
) -> None:
    """An undone patch on the engine object leaves a bound method that hides this one."""
    monkeypatch.setattr(type(engine), method, _stand_in)
    seen: Callable[..., Any] = getattr(engine, method)
    assert seen() == "stand-in"


@pytest.mark.parametrize(("engine", "method"), _PATCHED_METHODS)
def test_no_test_leaves_a_method_on_the_engine_object(engine: Any, method: str) -> None:
    """The engine object holds data only; its methods come from its class."""
    assert method not in vars(engine)
