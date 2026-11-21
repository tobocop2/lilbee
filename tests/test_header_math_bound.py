"""The header-math chat window fallback stays within the one planned bound."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines import LLAMA

_TRAINED = 40960
_TARGET = 16384


def _fit_fails(*_args: object, **_kwargs: object) -> int:
    raise ProviderError("no fit")


@pytest.mark.parametrize("cap", [3000, 20000, None])
def test_the_header_math_fallback_stays_within_the_planned_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int | None
) -> None:
    """A missing model file makes the header arithmetic fail, so the static cap answers."""
    monkeypatch.setattr(cfg, "chat_n_ctx_target", _TARGET)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", cap)
    monkeypatch.setattr(planning, "fit_chat_ctx", _fit_fails)
    path = tmp_path / "absent.gguf"
    meta = {"context_length": str(_TRAINED)}
    resolved = engine_params.resolve_chat_ctx(path, meta, engine=LLAMA, available_bytes=1 << 40)
    assert resolved == engine_params.chat_ctx_bound(meta, path)
    assert resolved == min(cap or _TRAINED, _TARGET)
