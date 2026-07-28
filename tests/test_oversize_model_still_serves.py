"""A model bigger than the GPUs is a partial offload, not a refusal.

The engine picks how many layers fit and keeps the rest in system memory. That
only works if lilbee launches it, and the refusal meant the fit never ran: the
role was skipped and chat was unavailable until the user hand-tuned
n_gpu_layers, which is the state the offload work was opened about.
"""

from __future__ import annotations

import logging

from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


def test_a_dense_model_larger_than_every_card_is_still_planned(monkeypatch, caplog) -> None:
    # Both pools are stated rather than read from the machine: the bound is
    # VRAM plus system memory, so a runner with less RAM than the developer's
    # laptop reaches the opposite verdict on an unpatched host.
    monkeypatch.setattr(planning.model_cache, "total_system_memory", lambda: 64 * _GB)
    with caplog.at_level(logging.WARNING):
        refused = planning._weights_exceed_hardware(40 * _GB, 24 * _GB, is_moe=False)
    assert not refused


def test_the_user_is_told_what_will_happen(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        planning._warn_weights_spill(WorkerRole.CHAT, "org/big", 40 * _GB, 24 * _GB)
    assert "system memory" in caplog.text
    assert "slower" in caplog.text or "slow" in caplog.text


def test_a_model_larger_than_vram_and_ram_together_is_still_refused() -> None:
    # Past both, there is nowhere for the layers to go and the load cannot win.
    assert planning._weights_exceed_everything(40 * _GB, total_vram=8 * _GB, total_ram=8 * _GB)
    assert not planning._weights_exceed_everything(
        40 * _GB, total_vram=24 * _GB, total_ram=64 * _GB
    )
