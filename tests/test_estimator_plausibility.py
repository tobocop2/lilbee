"""An estimate the parser returns confidently can still be impossible.

The fallback floor fires only when the estimator cannot answer. An answer below
the model's own weights plus the cache it was asked to hold is not a small error,
it is a number that cannot describe any load, and placement acting on it commits
a card it will then overrun.
"""

from __future__ import annotations

from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


def test_an_estimate_below_the_analytic_floor_is_rejected() -> None:
    assert planning._estimate_is_implausible(estimated=1 * _GB, floor=8 * _GB)


def test_an_estimate_at_or_above_the_floor_is_kept() -> None:
    assert not planning._estimate_is_implausible(estimated=8 * _GB, floor=8 * _GB)
    assert not planning._estimate_is_implausible(estimated=20 * _GB, floor=8 * _GB)


def test_an_unknown_floor_never_rejects() -> None:
    # No floor means nothing to compare against, and a guess is not grounds to
    # throw away the only measurement there is.
    assert not planning._estimate_is_implausible(estimated=1 * _GB, floor=0)


def test_the_floor_replaces_an_impossible_estimate(monkeypatch, caplog) -> None:
    from lilbee.providers.fleet.placement import ModelPlacementInput

    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 8 * _GB)
    monkeypatch.setattr(planning, "_fallback_floor_for", lambda *_a: 9 * _GB)
    tiny = ModelPlacementInput(WorkerRole.CHAT, 1 * _GB)
    with caplog.at_level("WARNING"):
        corrected = planning._floor_implausible_estimate(tiny, WorkerRole.CHAT, "org/m")
    assert corrected.est_vram_bytes == 9 * _GB
    assert "org/m" in caplog.text
