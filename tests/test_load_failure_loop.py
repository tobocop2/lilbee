"""An estimate that was too optimistic has to be recoverable."""

from __future__ import annotations

from itertools import pairwise

import pytest

from lilbee.providers.base import ProviderErrorKind

_GB = 1024**3


class TestAnEngineOomIsItsOwnFailure:
    """A load OOM was indistinguishable from any other death, so the same launch
    was respawned into a crash loop. Naming it is what lets a caller step the
    context down instead of retrying the identical command."""

    def test_a_cuda_oom_tail_is_classified_as_capacity(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = (
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4096.00 MiB "
            "on device 0: cudaMalloc failed: out of memory"
        )
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_a_generic_alloc_failure_counts_too(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = "llama_init_from_model: failed to allocate compute buffers"
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_a_vulkan_device_oom_counts(self) -> None:
        # Vulkan words it differently from every CUDA-shaped backend, and it is
        # the backend every AMD and Intel GPU lands on.
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = "ggml_vulkan: Device memory allocation of size 4294967296 failed."
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_a_vulkan_out_of_device_memory_error_counts(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = "vk::Device::allocateMemory: ErrorOutOfDeviceMemory"
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_a_metal_buffer_failure_counts(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = "ggml_metal_device_init: error: failed to allocate buffer, size =  4096.00 MiB"
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_an_unrecognised_death_is_left_alone(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        assert classify_upstream_death("something else entirely") is None


class TestATightRolePlacesAcrossCardsNotOntoOne:
    """Pinned to the single most-free card with the others excluded by the pin,
    a model that does not fit has nowhere to go. Given the whole group it can at
    least be split, which is what makes the tight placement a placement rather
    than a slower refusal."""

    def test_the_widest_group_is_used_when_no_single_card_fits(self) -> None:
        from lilbee.providers.fleet.placement import _place_tight

        remaining = {0: 10.0 * _GB, 1: 9.0 * _GB, 2: 8.0 * _GB}
        placed, _refunds, shortfall = _place_tight(_oversize(24 * _GB), remaining, charges={})
        assert placed.plan.devices == (0, 1, 2)
        assert shortfall <= 0  # across the three it fits; on any one of them it does not

    def test_a_model_too_big_for_every_card_together_still_reports_a_shortfall(self) -> None:
        from lilbee.providers.fleet.placement import _place_tight

        remaining = {0: 10.0 * _GB, 1: 9.0 * _GB}
        placed, _refunds, shortfall = _place_tight(_oversize(40 * _GB), remaining, charges={})
        assert placed.plan.devices == (0, 1)
        assert shortfall == 21 * _GB

    def test_a_model_that_fits_one_card_still_takes_one(self) -> None:
        from lilbee.providers.fleet.placement import _place_tight

        remaining = {0: 30.0 * _GB, 1: 9.0 * _GB}
        placed, _refunds, _shortfall = _place_tight(_oversize(24 * _GB), remaining, charges={})
        assert placed.plan.devices == (0,)


def _oversize(size: int):
    from lilbee.providers.fleet.placement import ModelPlacementInput
    from lilbee.providers.roles import WorkerRole

    return ModelPlacementInput(WorkerRole.CHAT, size)


class TestAnAutoContextStepsDownAfterAnOom:
    """Re-predicting the same number after a load OOM respawns the same launch.
    Stepping the auto context down changes the prediction, so the retry is a
    different launch rather than the same one again."""

    def test_a_step_halves_the_auto_chat_context(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
        planning.clear_ctx_downshift()
        before = planning.apply_ctx_downshift(WorkerRole.CHAT, 32768)
        assert planning.record_ctx_downshift(WorkerRole.CHAT) is True
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) == before // 2

    def test_stepping_stops_at_the_usable_floor(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
        planning.clear_ctx_downshift()
        steps = 0
        while planning.record_ctx_downshift(WorkerRole.CHAT):
            steps += 1
            planning.apply_ctx_downshift(WorkerRole.CHAT, 32768)  # as a plan would
            assert steps < 20, "the ladder has to terminate"
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) == planning.MIN_DOWNSHIFT_CTX
        planning.clear_ctx_downshift()

    def test_a_pinned_context_is_never_stepped_down(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", 16384, raising=False)
        planning.clear_ctx_downshift()
        assert planning.record_ctx_downshift(WorkerRole.CHAT) is False
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 16384) == 16384

    def test_the_estimate_uses_the_stepped_context(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers import engine_params
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 32768, raising=False)
        monkeypatch.setattr(engine_params, "chat_ctx_ceiling", lambda *_: 32768)
        planning.clear_ctx_downshift()
        full = planning._placement_estimate_ctx(WorkerRole.CHAT, _PATH, None)
        planning.record_ctx_downshift(WorkerRole.CHAT)
        assert planning._placement_estimate_ctx(WorkerRole.CHAT, _PATH, None) == full // 2
        planning.clear_ctx_downshift()


_PATH = __import__("pathlib").Path("/nonexistent/model.gguf")


class TestALoadOomRebuildsTheRoleInsteadOfRespawningIt:
    """Retrying a launch that ran out of memory produces the same launch and the
    same OOM. The retry has to be preceded by a smaller plan or not happen."""

    def test_an_oom_steps_the_context_down_and_rebuilds_once(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        planning.clear_ctx_downshift()
        p = FleetProvider()
        rebuilt: list[WorkerRole] = []
        monkeypatch.setattr(p, "_rebuild_role", rebuilt.append)
        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _oom_error()
            return "ok"

        assert p._with_rediscover(_call, role=WorkerRole.CHAT) == "ok"
        assert rebuilt == [WorkerRole.CHAT]
        assert planning._ctx_downshift_store.steps(WorkerRole.CHAT) == 1
        planning.clear_ctx_downshift()

    def test_a_second_oom_surfaces_rather_than_looping(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet import planning
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        planning.clear_ctx_downshift()
        p = FleetProvider()
        monkeypatch.setattr(p, "_rebuild_role", lambda _role: None)
        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            raise _oom_error()

        with pytest.raises(ProviderError):
            p._with_rediscover(_call, role=WorkerRole.CHAT)
        assert calls["n"] == 2  # one retry against the smaller plan, then it surfaces
        planning.clear_ctx_downshift()

    def test_a_pinned_context_does_not_retry_at_all(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet import planning
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", 16384, raising=False)
        planning.clear_ctx_downshift()
        p = FleetProvider()
        monkeypatch.setattr(p, "_rebuild_role", lambda _role: None)
        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            raise _oom_error()

        with pytest.raises(ProviderError):
            p._with_rediscover(_call, role=WorkerRole.CHAT)
        assert calls["n"] == 1  # nothing to step down, so nothing to retry


def _oom_error():
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    return ProviderError("cudaMalloc failed: out of memory", kind=ProviderErrorKind.CAPACITY)


class TestPortsAreNotPickedFromTheEphemeralRange:
    """The pick-then-bind gap spans the whole lazy-spawn wait, so a port the OS
    can hand to a passing outbound connection is a race the retry cannot win.
    Ports the OS never auto-assigns cannot be taken that way at all."""

    def test_a_picked_port_sits_below_the_ephemeral_range(self, monkeypatch) -> None:
        from lilbee.providers.fleet import swap_manager

        monkeypatch.setattr(swap_manager, "_ephemeral_range", lambda: (32768, 60999))
        ports = swap_manager._pick_free_ports(3)
        assert len(set(ports)) == 3
        assert all(port < 32768 for port in ports)

    def test_an_unknown_range_still_yields_usable_ports(self, monkeypatch) -> None:
        from lilbee.providers.fleet import swap_manager

        monkeypatch.setattr(swap_manager, "_ephemeral_range", lambda: None)
        ports = swap_manager._pick_free_ports(2)
        assert len(set(ports)) == 2
        assert all(port > 0 for port in ports)

    def test_the_range_is_read_from_the_running_kernel(self) -> None:
        from lilbee.providers.fleet.swap_manager import _ephemeral_range

        found = _ephemeral_range()
        if found is None:  # a platform that exposes neither source
            return
        low, high = found
        assert 1024 < low < high <= 65535


class TestAPersistentBindFailureRepicksThePort:
    """Re-driving llama-swap's spawn wins the ordinary bind race. When something
    else holds the port for good, the same spawn against the same baked-in port
    fails forever; only a rebuild picks a different one."""

    def test_a_bind_death_is_its_own_kind(self) -> None:
        from lilbee.providers.fleet.client import classify_upstream_death

        tail = "main: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 39100"
        assert classify_upstream_death(tail) is ProviderErrorKind.PORT_CONFLICT

    def test_a_bind_death_is_still_retried_by_the_client(self) -> None:
        from lilbee.providers.fleet.client import _TRANSIENT_KINDS

        assert ProviderErrorKind.PORT_CONFLICT in _TRANSIENT_KINDS

    def test_a_surviving_bind_failure_rebuilds_the_role_once(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet import planning
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        planning.clear_ctx_downshift()
        p = FleetProvider()
        rebuilt: list[WorkerRole] = []
        monkeypatch.setattr(p, "_rebuild_role", rebuilt.append)
        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderError("bind", kind=ProviderErrorKind.PORT_CONFLICT)
            return "ok"

        assert p._with_rediscover(_call, role=WorkerRole.EMBED) == "ok"
        assert rebuilt == [WorkerRole.EMBED]  # a fresh plan means a freshly picked port
        # A held port is not a memory shortfall: shrinking the context would
        # serve a smaller window for a problem that has nothing to do with size.
        assert planning._ctx_downshift_store.steps(WorkerRole.EMBED) == 0

    def test_a_bind_failure_that_survives_the_rebuild_surfaces(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet.provider import FleetProvider
        from lilbee.providers.roles import WorkerRole

        p = FleetProvider()
        monkeypatch.setattr(p, "_rebuild_role", lambda _role: None)
        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            raise ProviderError("bind", kind=ProviderErrorKind.PORT_CONFLICT)

        with pytest.raises(ProviderError):
            p._with_rediscover(_call, role=WorkerRole.EMBED)
        assert calls["n"] == 2


class TestTheUncoveredEdgesOfTheseLadders:
    """The guards that only fire on inputs the ordinary paths never produce."""

    def test_a_capacity_failure_is_recognised_directly(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet.client import is_load_capacity_failure

        assert is_load_capacity_failure(_oom_error())
        assert not is_load_capacity_failure(ProviderError("x", kind=ProviderErrorKind.SERVER))
        assert not is_load_capacity_failure(ValueError("not a provider error"))

    def test_a_rebuildable_failure_covers_both_kinds(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet.client import is_rebuildable_failure

        assert is_rebuildable_failure(_oom_error())
        assert is_rebuildable_failure(ProviderError("x", kind=ProviderErrorKind.PORT_CONFLICT))
        assert not is_rebuildable_failure(ProviderError("x", kind=ProviderErrorKind.SERVER))

    def test_a_tight_group_with_no_cards_at_all_is_empty(self) -> None:
        from lilbee.providers.fleet.placement import _tight_device_group

        assert _tight_device_group(1, {}) == ()

    def test_every_card_already_drained_still_yields_one(self) -> None:
        from lilbee.providers.fleet.placement import _tight_device_group

        assert _tight_device_group(_GB, {0: 0.0, 1: 0.0}) == (0,)

    def test_the_proc_range_is_read_when_present(self, tmp_path) -> None:
        from lilbee.providers.fleet.swap_manager import _port_range_from

        proc = tmp_path / "ip_local_port_range"
        proc.write_text("32768\t60999\n")
        assert _port_range_from(proc) == (32768, 60999)

    def test_an_absent_or_garbled_proc_range_is_no_answer(self, tmp_path) -> None:
        from lilbee.providers.fleet.swap_manager import _port_range_from

        assert _port_range_from(tmp_path / "missing") is None
        garbled = tmp_path / "garbled"
        garbled.write_text("not numbers here\n")
        assert _port_range_from(garbled) is None

    def test_the_proc_file_answers_before_sysctl_is_asked(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import swap_manager

        proc = tmp_path / "ip_local_port_range"
        proc.write_text("32768 60999\n")
        monkeypatch.setattr(swap_manager, "_PROC_PORT_RANGE", proc)
        monkeypatch.setattr(
            swap_manager.subprocess,
            "run",
            lambda *a, **k: pytest.fail("sysctl must not be asked when procfs answered"),
        )
        assert swap_manager._ephemeral_range() == (32768, 60999)

    def test_a_sysctl_that_cannot_run_is_no_answer(self, monkeypatch, tmp_path) -> None:
        import subprocess

        from lilbee.providers.fleet import swap_manager

        monkeypatch.setattr(swap_manager, "_PROC_PORT_RANGE", tmp_path / "missing")
        monkeypatch.setattr(
            swap_manager.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(subprocess.SubprocessError("no sysctl")),
        )
        assert swap_manager._ephemeral_range() is None


class TestADownshiftNeverAsksForMore:
    """The floor is a stopping point, not a target. Applied to a context already
    below it, a bare max() raises the number, so the retry after a load OOM asks
    for more memory than the launch that just ran out of it."""

    def test_a_small_non_chat_context_is_never_raised(self) -> None:
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        planning.clear_ctx_downshift()
        planning.record_ctx_downshift(WorkerRole.EMBED)
        assert planning.apply_ctx_downshift(WorkerRole.EMBED, 512) <= 512
        planning.clear_ctx_downshift()

    def test_a_chat_model_with_a_short_window_is_never_raised(self, monkeypatch) -> None:
        # A model trained for 2048 tokens sits below the floor; stepping it must
        # not hand back 4096, which the model cannot serve at all.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
        planning.clear_ctx_downshift()
        planning.record_ctx_downshift(WorkerRole.CHAT)
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 2048) <= 2048
        planning.clear_ctx_downshift()

    def test_every_step_of_a_normal_ladder_still_shrinks(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning
        from lilbee.providers.roles import WorkerRole

        monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
        # The ladder's length comes from the configured target, so pin it rather
        # than asserting against whatever this machine's config happens to say.
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 32768, raising=False)
        planning.clear_ctx_downshift()
        seen = [32768]
        while planning.record_ctx_downshift(WorkerRole.CHAT):
            seen.append(planning.apply_ctx_downshift(WorkerRole.CHAT, 32768))
        assert seen == [32768, 16384, 8192, 4096]
        assert all(b < a for a, b in pairwise(seen))
        planning.clear_ctx_downshift()
