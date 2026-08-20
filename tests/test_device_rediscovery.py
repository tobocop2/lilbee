"""Hardware can leave. The structural snapshot has to notice."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice

_GB = 1024**3


@pytest.fixture(autouse=True)
def _reset():
    planning_mod.clear_plan_probe()
    yield
    planning_mod.clear_plan_probe()


def _snapshot(monkeypatch, devices: list[FleetDevice], free_ram: int = 64 * _GB) -> None:
    monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
    monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda *_a: None
    )
    monkeypatch.setattr(planning_mod, "_resolve_devices_and_refusal", lambda _b: (devices, False))
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: free_ram)
    planning_mod.capture_plan_probe()


class TestAReloadRediscoversDevices:
    """The snapshot is taken once and only a full teardown clears it, so an eGPU
    unplug, a driver reset or a VM hot-remove left the fleet pinning a device
    that is no longer there, and every rebuild replanned onto it."""

    def test_a_departed_card_leaves_the_snapshot(self, monkeypatch) -> None:
        two = [
            FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB),
            FleetDevice("CUDA", 1, "B", 24 * _GB, 24 * _GB),
        ]
        _snapshot(monkeypatch, two)
        assert len(planning_mod._plan_devices(Path("/bin/srv"))) == 2

        monkeypatch.setattr(
            planning_mod, "_resolve_devices_and_refusal", lambda _b: (two[:1], False)
        )
        planning_mod.refresh_plan_devices()
        assert [d.index for d in planning_mod._plan_devices(Path("/bin/srv"))] == [0]

    def test_the_clean_box_memory_figures_survive_the_refresh(self, monkeypatch) -> None:
        # Only the structural half is restated. The memory snapshot is what makes
        # a reload plan like the boot did, and re-taking it under a loaded fleet
        # would charge the fleet against itself.
        card = FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB)
        _snapshot(monkeypatch, [card], free_ram=64 * _GB)
        before = planning_mod._plan_free_system_memory()

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1 * _GB)
        planning_mod.refresh_plan_devices()
        assert planning_mod._plan_free_system_memory() == before

    def test_a_refresh_without_a_snapshot_does_nothing(self, monkeypatch) -> None:
        # Nothing has been captured, so there is nothing to restate and the next
        # capture will read the hardware anyway.
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        planning_mod.refresh_plan_devices()
        assert planning_mod._plan_probe_store.get() is None


class TestTheReloadPassAsksForRediscovery:
    """Wiring, not logic: the refresh only helps if the reload calls it."""

    def test_a_reload_pass_refreshes_the_device_list(self, monkeypatch) -> None:
        from lilbee.providers.fleet import provider as provider_mod

        called: list[int] = []
        monkeypatch.setattr(provider_mod.planning, "refresh_plan_devices", lambda: called.append(1))
        prov = provider_mod.FleetProvider.__new__(provider_mod.FleetProvider)
        import threading

        prov._build_lock = threading.RLock()
        prov._lock = threading.RLock()
        prov._shut_down = True  # returns immediately, after the refresh
        prov._reload_pass()
        assert called == [1]


class TestARefreshThatCannotProbe:
    """A probe that will not run is not evidence the hardware left."""

    def test_the_previous_device_list_is_kept(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        card = FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB)
        _snapshot(monkeypatch, [card])

        def _wedged(_binary):
            raise ProviderError("probe wedged")

        monkeypatch.setattr(planning_mod, "_resolve_devices_and_refusal", _wedged)
        planning_mod.refresh_plan_devices()
        assert [d.index for d in planning_mod._plan_devices(Path("/bin/srv"))] == [0]

    def test_an_unchanged_list_is_left_in_place(self, monkeypatch, caplog) -> None:
        import logging

        card = FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB)
        _snapshot(monkeypatch, [card])
        with caplog.at_level(logging.INFO, logger="lilbee.providers.fleet.planning"):
            planning_mod.refresh_plan_devices()
        assert "changed since" not in caplog.text


class TestTheRefreshKeepsThePerDeviceFreeFigures:
    """A reload re-probes while the outgoing fleet is still resident, so the live
    free readings are deflated by the very memory the reload is about to release.
    Adopting them makes a model swap size the incoming chat model against the
    outgoing model's residency (a 512-token window on cards that back a full one)."""

    def test_a_resident_fleet_does_not_replace_the_snapshot(self, monkeypatch, caplog) -> None:
        import logging

        card = FleetDevice("CUDA", 0, "A", 24 * _GB, 23 * _GB)
        _snapshot(monkeypatch, [card])
        deflated = [FleetDevice("CUDA", 0, "A", 24 * _GB, 2 * _GB)]
        monkeypatch.setattr(
            planning_mod, "_resolve_devices_and_refusal", lambda _b: (deflated, False)
        )
        with caplog.at_level(logging.INFO, logger="lilbee.providers.fleet.planning"):
            planning_mod.refresh_plan_devices()
        probe = planning_mod._plan_probe_store.get()
        assert probe is not None
        assert [d.free_bytes for d in probe.devices] == [23 * _GB]
        # A free-memory swing is not a hardware change and must not read as one.
        assert "changed since" not in caplog.text

    def test_a_surviving_card_keeps_its_clean_box_figure_when_a_card_leaves(
        self, monkeypatch
    ) -> None:
        two = [
            FleetDevice("CUDA", 0, "A", 24 * _GB, 23 * _GB),
            FleetDevice("CUDA", 1, "B", 24 * _GB, 22 * _GB),
        ]
        _snapshot(monkeypatch, two)
        remaining = [FleetDevice("CUDA", 0, "A", 24 * _GB, 2 * _GB)]
        monkeypatch.setattr(
            planning_mod, "_resolve_devices_and_refusal", lambda _b: (remaining, False)
        )
        planning_mod.refresh_plan_devices()
        probe = planning_mod._plan_probe_store.get()
        assert probe is not None
        assert [d.free_bytes for d in probe.devices] == [23 * _GB]
