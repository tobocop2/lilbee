"""The headroom that decides admission has to be reachable by the user."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg


class TestTheUsableVramFractionIsTunable:
    """Hardcoded at 0.9, a 16 GB Mac with a 12-13 GB chat footprint could never
    serve chat at all, and there was no way for the owner to intervene."""

    def test_the_default_is_the_value_that_was_hardcoded(self) -> None:
        from lilbee.providers.fleet.vram import usable_vram_fraction

        assert cfg.usable_vram_fraction == pytest.approx(0.9)
        assert usable_vram_fraction() == pytest.approx(0.9)

    def test_raising_it_widens_what_placement_will_charge(self, monkeypatch) -> None:
        from lilbee.providers.fleet.vram import usable_vram_fraction

        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.97)
        assert usable_vram_fraction() == pytest.approx(0.97)


class TestTheSystemMemoryFloorIsTunable:
    """The OS reserve on a shared-memory host had the same problem."""

    def test_the_default_is_the_value_that_was_hardcoded(self) -> None:
        assert cfg.system_memory_reserve_gb == pytest.approx(4.0)

    def test_lowering_it_frees_budget_on_a_small_host(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr(cfg, "system_memory_reserve_gb", 1.0)
        monkeypatch.setattr(
            "lilbee.providers.model_cache.total_system_memory", lambda: 64 * 1024**3
        )
        # A quarter of RAM still caps it, so the cap is what moved.
        assert planning_mod._system_memory_floor() == 1 * 1024**3

    def test_a_quarter_of_ram_still_caps_a_small_host(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr(cfg, "system_memory_reserve_gb", 4.0)
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 8 * 1024**3)
        assert planning_mod._system_memory_floor() == 2 * 1024**3
