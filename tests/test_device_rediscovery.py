"""Hardware can leave. The structural snapshot has to notice."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import _REPORTED_BACKEND, FleetDevice
from lilbee.providers.roles import EngineBackend

_GB = 1024**3


@pytest.fixture(autouse=True)
def _reset():
    planning_mod.clear_plan_probe()
    yield
    planning_mod.clear_plan_probe()


def _reading(devices: list[FleetDevice]) -> planning_mod.DeviceReading:
    """A device reading as one probe run would answer it."""
    return planning_mod.DeviceReading(devices, EngineBackend.CUDA)


def _snapshot(monkeypatch, devices: list[FleetDevice], free_ram: int = 64 * _GB) -> None:
    monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
    monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda *_a: None
    )
    monkeypatch.setattr(planning_mod, "_read_devices", lambda _b: _reading(devices))
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

        monkeypatch.setattr(planning_mod, "_read_devices", lambda _b: _reading(two[:1]))
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

        monkeypatch.setattr(planning_mod, "_read_devices", _wedged)
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
        monkeypatch.setattr(planning_mod, "_read_devices", lambda _b: _reading(deflated))
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
        monkeypatch.setattr(planning_mod, "_read_devices", lambda _b: _reading(remaining))
        planning_mod.refresh_plan_devices()
        probe = planning_mod._plan_probe_store.get()
        assert probe is not None
        assert [d.free_bytes for d in probe.devices] == [23 * _GB]


class TestAnEngineThatChangesUnderARunningServe:
    """A probe result describes one engine binary and must not outlive it.

    A serve started while the engine wheel shipped an empty stub placed every
    model on the CPU, and kept placing there after the real wheel was installed,
    because the snapshot taken against the stub was still the answer.
    """

    def _engine(self, monkeypatch, binary: Path, backend: str = "CUDA") -> list[int]:
        """Point planning at *binary* and report cards only while it has content."""
        runs: list[int] = []
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: binary)
        monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda *_a: None
        )
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 64 * _GB)

        def _probe(_binary: Path) -> planning_mod.DeviceReading:
            runs.append(1)
            if not binary.read_bytes():
                # An empty stub never answers, so it names no backend either.
                return planning_mod.DeviceReading([], EngineBackend.UNKNOWN)
            return planning_mod.DeviceReading(
                [
                    FleetDevice(backend, 0, "A", 24 * _GB, 24 * _GB),
                    FleetDevice(backend, 1, "B", 24 * _GB, 24 * _GB),
                ],
                _REPORTED_BACKEND[backend],
            )

        monkeypatch.setattr(planning_mod, "_read_devices", _probe)
        return runs

    def _wedge(self, monkeypatch, runs: list[int]) -> None:
        """Make the probe raise, counting each attempt in *runs*."""
        from lilbee.providers.base import ProviderError

        def _raise(_binary: Path) -> planning_mod.DeviceReading:
            runs.append(1)
            raise ProviderError("probe wedged", provider="llama-server")

        monkeypatch.setattr(planning_mod, "_read_devices", _raise)

    def test_the_cards_appear_once_the_real_engine_lands(self, monkeypatch, tmp_path) -> None:
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        assert planning_mod._plan_devices(binary) == []

        binary.write_bytes(b"the real engine")

        assert [d.index for d in planning_mod._plan_devices(binary)] == [0, 1]

    def test_an_unchanged_engine_is_probed_once(self, monkeypatch, tmp_path) -> None:
        # The control the fix has to survive: re-probing every read would pass
        # every other arm here and quietly reintroduce the cost the cache exists
        # to avoid.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"the real engine")
        runs = self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        assert len(runs) == 1

        for _ in range(5):
            planning_mod._plan_devices(binary)
            planning_mod.plan_sizing_budget()
            planning_mod.probed_devices()

        assert len(runs) == 1

    def test_a_broken_engine_stops_the_cards_being_planned_onto(
        self, monkeypatch, tmp_path
    ) -> None:
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"the real engine")
        self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        assert len(planning_mod._plan_devices(binary)) == 2

        binary.write_bytes(b"")

        assert planning_mod._plan_devices(binary) == []

    def test_the_cuda_and_rocm_paths_recover_alike(self, monkeypatch, tmp_path) -> None:
        # One cache serves both, so the recovery must not differ by backend.
        def _recovered(name: str, backend: str) -> list[FleetDevice]:
            with pytest.MonkeyPatch.context() as patch:
                binary = tmp_path / name
                binary.write_bytes(b"")
                self._engine(patch, binary, backend)
                planning_mod.capture_plan_probe()
                assert planning_mod._plan_devices(binary) == []
                binary.write_bytes(b"the real engine")
                return planning_mod._plan_devices(binary)

        cuda = _recovered("cuda-server", "CUDA")
        planning_mod.clear_plan_probe()
        rocm = _recovered("rocm-server", "ROCm")

        assert len(cuda) == 2
        assert [(d.index, d.total_bytes, d.free_bytes) for d in cuda] == [
            (d.index, d.total_bytes, d.free_bytes) for d in rocm
        ]

    def test_a_probe_that_failed_once_does_not_latch_the_stale_list(
        self, monkeypatch, tmp_path
    ) -> None:
        # The reported incident: the real engine lands under the serve and the
        # first read after it probes, and that probe fails for a transient reason.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        runs = self._engine(monkeypatch, binary)
        healthy = planning_mod._read_devices
        planning_mod.capture_plan_probe()
        assert planning_mod._plan_devices(binary) == []

        binary.write_bytes(b"the real engine")
        self._wedge(monkeypatch, runs)
        assert planning_mod._plan_devices(binary) == []

        monkeypatch.setattr(planning_mod, "_read_devices", healthy)
        monkeypatch.setattr(planning_mod, "_PLAN_RESTATE_FAILURE_WAIT_S", 0.0)

        assert [d.index for d in planning_mod._plan_devices(binary)] == [0, 1]

    def test_a_repaired_binary_is_not_held_behind_the_failed_wait(
        self, monkeypatch, tmp_path
    ) -> None:
        # The wait is keyed on the engine identity, so a repair that changes the
        # bytes is a different engine and gets its own probe. The full wait stays
        # in force: the arm above zeroes it, which proves only that it expires.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        runs = self._engine(monkeypatch, binary)
        healthy = planning_mod._read_devices
        planning_mod.capture_plan_probe()
        assert planning_mod._plan_devices(binary) == []

        binary.write_bytes(b"an engine that cannot probe")
        self._wedge(monkeypatch, runs)
        assert planning_mod._plan_devices(binary) == []
        broken = planning_mod._engine_identity()
        probed_while_broken = len(runs)

        binary.write_bytes(b"the real engine, repaired in place")
        monkeypatch.setattr(planning_mod, "_read_devices", healthy)

        assert planning_mod._plan_probe_store.probe_failed_recently(broken)
        assert [d.index for d in planning_mod._plan_devices(binary)] == [0, 1]
        assert len(runs) == probed_while_broken + 1

    def test_a_probe_that_keeps_failing_is_not_retried_on_every_read(
        self, monkeypatch, tmp_path
    ) -> None:
        # The retry ladder costs seconds of sleeps, so a broken engine must not
        # pay it once per read while the snapshot stays stale.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        runs = self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        binary.write_bytes(b"the real engine")
        self._wedge(monkeypatch, runs)
        before = len(runs)

        for _ in range(5):
            planning_mod._plan_devices(binary)

        assert len(runs) == before + 1

    def test_a_binary_replaced_while_the_capture_probes_is_not_recorded(
        self, monkeypatch, tmp_path
    ) -> None:
        # The install races the capture, which is the incident's own window: the
        # answer belongs to the binary that gave it, never to the one that landed.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        runs = self._engine(monkeypatch, binary)
        healthy = planning_mod._read_devices

        def _probe_then_install(probed: Path) -> planning_mod.DeviceReading:
            answer = healthy(probed)
            binary.write_bytes(b"the real engine")
            return answer

        monkeypatch.setattr(planning_mod, "_read_devices", _probe_then_install)
        planning_mod.capture_plan_probe()
        monkeypatch.setattr(planning_mod, "_read_devices", healthy)
        probed_at_capture = len(runs)

        assert [d.index for d in planning_mod._plan_devices(binary)] == [0, 1]
        assert len(runs) > probed_at_capture

    def test_a_host_that_lost_its_engine_keeps_trying_to_reach_one(
        self, monkeypatch, tmp_path
    ) -> None:
        # Nothing can answer here, so a read that stops asking is a snapshot
        # tagged with a binary no probe ever reached.
        from lilbee.providers.base import ProviderError

        binary = tmp_path / "llama-server"
        binary.write_bytes(b"the real engine")
        self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        attempts: list[int] = []
        probe_engine = planning_mod._probe_engine_devices

        def _counted() -> planning_mod.DeviceReading:
            attempts.append(1)
            return probe_engine()

        def _gone() -> Path:
            raise ProviderError("no engine binary", provider="llama-server")

        monkeypatch.setattr(planning_mod, "_probe_engine_devices", _counted)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", _gone)
        assert len(planning_mod._plan_devices(binary)) == 2
        assert len(attempts) == 1

        monkeypatch.setattr(planning_mod, "_PLAN_RESTATE_FAILURE_WAIT_S", 0.0)

        assert len(planning_mod._plan_devices(binary)) == 2
        assert len(attempts) == 2

    def test_the_cpu_pin_follows_the_engine_that_refused_the_cards(
        self, monkeypatch, tmp_path
    ) -> None:
        # The stub listed nothing and refused nothing; the real engine lists a
        # paravirtual adapter lilbee will not plan onto. Both answer with an
        # empty device list, so a restate that keeps the stub's refusal leaves
        # the pin off and ggml falls back onto the adapter just refused.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()
        assert planning_mod._cpu_pin_when_every_device_was_refused() == ()

        binary.write_bytes(b"the real engine")
        # Every listed GPU was refused, so the selection is empty and the engine
        # that answered names CPU, exactly as a host with no GPU would.
        monkeypatch.setattr(
            planning_mod,
            "_read_devices",
            lambda _b: planning_mod.DeviceReading([], EngineBackend.CPU, True),
        )

        assert planning_mod._cpu_pin_when_every_device_was_refused() == ("none",)

    def test_a_planning_pass_answers_about_one_binary(self, monkeypatch, tmp_path) -> None:
        # A pass reads the snapshot several times. An engine that lands between
        # two of them must not size the plan against one and place it against
        # another; the pass asks about the binary it started on.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"the real engine")
        runs = self._engine(monkeypatch, binary)
        planning_mod.capture_plan_probe()

        with planning_mod._one_engine_per_pass():
            pinned = planning_mod._pass_engine_identity()
            binary.write_bytes(b"a newer engine")
            with planning_mod._one_engine_per_pass():
                assert planning_mod._pass_engine_identity() == pinned
            planning_mod._plan_devices(binary)
            planning_mod.plan_sizing_budget()

        assert len(runs) == 1
        assert len(planning_mod._plan_devices(binary)) == 2
        assert len(runs) == 2

    def test_a_second_stale_read_takes_the_first_ones_answer(self, monkeypatch, tmp_path) -> None:
        # A reader that queues behind the restate takes its answer, not another probe.
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        runs = self._engine(monkeypatch, binary)
        healthy = planning_mod._read_devices
        planning_mod.capture_plan_probe()
        started = threading.Event()
        release = threading.Event()

        def _blocking(probed: Path) -> planning_mod.DeviceReading:
            started.set()
            release.wait(10)
            return healthy(probed)

        monkeypatch.setattr(planning_mod, "_read_devices", _blocking)
        binary.write_bytes(b"the real engine")
        before = len(runs)
        seen: list[list[FleetDevice]] = []
        readers = [
            threading.Thread(target=lambda: seen.append(planning_mod._plan_devices(binary)))
            for _ in range(2)
        ]
        for reader in readers:
            reader.start()
        started.wait(10)
        release.set()
        for reader in readers:
            reader.join(10)

        assert len(runs) == before + 1
        assert [len(devices) for devices in seen] == [2, 2]


class TestTheReadCacheDescribesOneBinary:
    """The short-TTL read cache answers for the binary it probed, not the next one."""

    def _probed(self, monkeypatch) -> list[Path]:
        seen: list[Path] = []

        def _resolve(binary: Path) -> planning_mod.DeviceReading:
            seen.append(binary)
            return _reading([FleetDevice("CUDA", len(seen) - 1, "A", 24 * _GB, 24 * _GB)])

        monkeypatch.setattr(planning_mod, "_read_devices", _resolve)
        return seen

    def test_a_second_binary_is_probed_for_itself(self, monkeypatch, tmp_path) -> None:
        seen = self._probed(monkeypatch)
        first = tmp_path / "old"
        first.write_bytes(b"old engine")
        second = tmp_path / "new"
        second.write_bytes(b"new engine")
        cache = planning_mod._ReadDeviceCache(60.0, 60.0)

        cache.get(first)
        cache.get(first)
        cache.get(second)

        assert seen == [first, second]

    def test_the_same_path_with_new_bytes_is_probed_again(self, monkeypatch, tmp_path) -> None:
        # The reported defect's own shape: one path whose contents were replaced.
        seen = self._probed(monkeypatch)
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"stub")
        cache = planning_mod._ReadDeviceCache(60.0, 60.0)
        cache.get(binary)

        binary.write_bytes(b"the real engine, longer")
        cache.get(binary)

        assert seen == [binary, binary]

    def test_a_cached_failure_is_not_reraised_for_another_binary(
        self, monkeypatch, tmp_path
    ) -> None:
        from lilbee.providers.base import ProviderError

        stub = tmp_path / "stub"
        stub.write_bytes(b"")
        real = tmp_path / "real"
        real.write_bytes(b"the real engine")

        def _resolve(binary: Path) -> planning_mod.DeviceReading:
            if binary == stub:
                raise ProviderError("probe wedged", provider="llama-server")
            return _reading([FleetDevice("CUDA", 0, "A", 24 * _GB, 24 * _GB)])

        monkeypatch.setattr(planning_mod, "_read_devices", _resolve)
        cache = planning_mod._ReadDeviceCache(60.0, 60.0)
        with pytest.raises(ProviderError):
            cache.get(stub)

        assert [d.index for d in cache.get(real).devices] == [0]
