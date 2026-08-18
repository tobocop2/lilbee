"""An estimate the parser returns confidently can still be impossible.

The fallback floor fires only when the estimator cannot answer. An answer below
the weight bytes the card must hold is not a small error, it is a number that
cannot describe any load, and placement acting on it commits a card it will then
overrun. Everything above that bound is the estimator's to report: how much cache
a layer holds depends on the attention it runs, and the planner does not model that.
"""

from __future__ import annotations

from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


def test_an_estimate_below_the_floor_is_rejected() -> None:
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

    monkeypatch.setattr(planning, "_cpu_offload_in_play", lambda: False)
    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 8 * _GB)
    tiny = ModelPlacementInput(WorkerRole.CHAT, 1 * _GB)
    with caplog.at_level("WARNING"):
        corrected = planning._floor_implausible_estimate(tiny, WorkerRole.CHAT, "org/m")
    assert corrected.est_vram_bytes == 8 * _GB
    assert "org/m" in caplog.text


def test_a_cache_smaller_than_dense_attention_would_hold_is_kept(monkeypatch, caplog) -> None:
    # Qwen3.6-27B on an A40: the parser said 25.7 GiB and the load took 26.4,
    # while header math over a window this size predicts far more, because most
    # of the model's layers run linear attention and hold no per-token cache.
    from lilbee.providers.fleet.placement import ModelPlacementInput

    def _refuse(*_a: object) -> int:
        raise AssertionError("the analytic floor must not weigh in on a well-formed estimate")

    monkeypatch.setattr(planning, "_cpu_offload_in_play", lambda: False)
    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 17 * _GB)
    monkeypatch.setattr(planning, "_fallback_floor_for", _refuse)
    measured = ModelPlacementInput(WorkerRole.CHAT, 26 * _GB)
    with caplog.at_level("WARNING"):
        kept = planning._floor_implausible_estimate(measured, WorkerRole.CHAT, "org/qwen")
    assert kept.est_vram_bytes == 26 * _GB
    assert not caplog.text


def test_an_offloaded_model_may_hold_less_than_its_weights(monkeypatch, caplog) -> None:
    # Part of the model lives in system memory, so the card holds less than the
    # file: the weight bytes stop being a bound on what the GPU takes.
    from lilbee.providers.fleet.placement import ModelPlacementInput

    monkeypatch.setattr(planning, "_cpu_offload_in_play", lambda: True)
    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 60 * _GB)
    split = ModelPlacementInput(WorkerRole.CHAT, 12 * _GB)
    with caplog.at_level("WARNING"):
        kept = planning._floor_implausible_estimate(split, WorkerRole.CHAT, "org/moe")
    assert kept.est_vram_bytes == 12 * _GB
    assert not caplog.text
