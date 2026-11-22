"""What the plan charges when layers are pushed off the GPU.

Every byte the engine keeps in system memory is a byte the plan does not charge:
gguf-parser reports it, the estimate carries it, and nothing has ever read it. On
a dedicated-VRAM host that is invisible until the box starts swapping.
"""

from __future__ import annotations

import logging

import pytest

from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_REF = "some-org/some-model"


@pytest.fixture
def offload_configured(monkeypatch):
    from lilbee.core.config import cfg
    from lilbee.providers.fleet import planning

    monkeypatch.setattr(cfg, "cpu_moe", True, raising=False)
    # These cases are about the size comparison, not about whether the bytes can
    # page out; residency has its own tests in test_host_memory_bound.
    monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda *_a: True)
    return cfg


class TestHostMemoryIsChargedWhenLayersLeaveTheGpu:
    def test_a_model_whose_host_half_exceeds_the_machine_is_refused(
        self, monkeypatch, offload_configured
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "total_system_memory", lambda: 16 * _GB)
        assert planning._host_memory_refuses(WorkerRole.CHAT, _REF, 40 * _GB, 0, lambda: 8 * _GB)

    def test_a_model_that_exceeds_only_free_memory_is_allowed_with_a_warning(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
        with caplog.at_level(logging.WARNING):
            refused = planning._host_memory_refuses(
                WorkerRole.CHAT, _REF, 40 * _GB, 0, lambda: 8 * _GB
            )
        assert not refused  # short on free memory is a warning, not a refusal
        assert "system memory" in caplog.text

    def test_a_model_that_fits_free_memory_is_silent(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
        with caplog.at_level(logging.WARNING):
            refused = planning._host_memory_refuses(
                WorkerRole.CHAT, _REF, 8 * _GB, 0, lambda: 40 * _GB
            )
        assert not refused
        assert caplog.text == ""

    def test_nothing_is_charged_when_no_layer_ever_leaves_the_gpu(self, monkeypatch) -> None:
        # Without an offload setting the engine keeps everything on the card, so
        # the host figure describes memory nobody will allocate.
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(cfg, "cpu_moe", False, raising=False)
        monkeypatch.setattr(cfg, "n_cpu_moe", None, raising=False)
        monkeypatch.setattr(cfg, "n_gpu_layers", None, raising=False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 1, raising=False)
        assert not planning._host_memory_refuses(WorkerRole.CHAT, _REF, 999 * _GB, 0, lambda: 0)

    def test_a_partial_gpu_layer_budget_counts_as_offload(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(cfg, "cpu_moe", False, raising=False)
        monkeypatch.setattr(cfg, "n_cpu_moe", None, raising=False)
        monkeypatch.setattr(cfg, "n_gpu_layers", 12, raising=False)
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda *_a: True)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 16 * _GB)
        assert planning._host_memory_refuses(WorkerRole.CHAT, _REF, 40 * _GB, 0, lambda: 8 * _GB)

    def test_a_cpu_only_role_counts_as_offload(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(cfg, "cpu_moe", False, raising=False)
        monkeypatch.setattr(cfg, "n_cpu_moe", None, raising=False)
        monkeypatch.setattr(cfg, "n_gpu_layers", 0, raising=False)
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda *_a: True)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 16 * _GB)
        assert planning._host_memory_refuses(WorkerRole.CHAT, _REF, 40 * _GB, 0, lambda: 8 * _GB)


class TestTheFreeFigureIsThePlansSnapshot:
    def test_admission_warns_from_the_plan_snapshot_and_never_reads_live_free_memory(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
        monkeypatch.setattr(planning, "_plan_free_system_memory", lambda _engine: 8 * _GB)
        monkeypatch.setattr(
            planning.model_cache, "free_system_memory", lambda: pytest.fail("live free read")
        )
        with caplog.at_level(logging.WARNING):
            planning._admit_estimate(
                _estimate(),
                WorkerRole.CHAT,
                _REF,
                total_vram=64 * _GB,
                ram_bytes=40 * _GB,
                engine=LLAMA,
            )
        assert "Offloading the chat model" in caplog.text
        assert "only 8.0 GiB is free" in caplog.text

    def test_a_free_figure_below_one_gib_prints_in_mib_never_as_zero_gib(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
        monkeypatch.setattr(planning, "_plan_free_system_memory", lambda _engine: 30 * 1024**2)
        with caplog.at_level(logging.WARNING):
            planning._admit_estimate(
                _estimate(),
                WorkerRole.CHAT,
                _REF,
                total_vram=64 * _GB,
                ram_bytes=40 * _GB,
                engine=LLAMA,
            )
        assert "only 30 MiB is free" in caplog.text
        assert "0.0 GiB" not in caplog.text

    def test_a_role_that_nothing_offloads_never_reads_free_memory(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(cfg, "cpu_moe", False, raising=False)
        monkeypatch.setattr(cfg, "n_cpu_moe", None, raising=False)
        monkeypatch.setattr(cfg, "n_gpu_layers", None, raising=False)
        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(
            planning, "_plan_free_system_memory", lambda _engine: pytest.fail("free read")
        )
        estimate = _estimate()
        admitted = planning._admit_estimate(
            estimate,
            WorkerRole.CHAT,
            _REF,
            total_vram=64 * _GB,
            ram_bytes=40 * _GB,
            engine=LLAMA,
        )
        assert admitted is estimate

    def test_a_refused_role_never_reads_free_memory(self, monkeypatch, offload_configured) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 16 * _GB)
        monkeypatch.setattr(
            planning, "_plan_free_system_memory", lambda _engine: pytest.fail("free read")
        )
        assert (
            planning._admit_estimate(
                _estimate(),
                WorkerRole.CHAT,
                _REF,
                total_vram=64 * _GB,
                ram_bytes=40 * _GB,
                engine=LLAMA,
            )
            is None
        )


class TestTheHostBoundReachesAdmission:
    """The figure has to be read where a role is admitted, not just computed."""

    def test_a_refused_host_footprint_drops_the_role(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "_host_memory_refuses", lambda *_a: True)
        assert (
            planning._admit_estimate(
                _estimate(),
                WorkerRole.CHAT,
                _REF,
                total_vram=64 * _GB,
                ram_bytes=40 * _GB,
                engine=LLAMA,
            )
            is None
        )

    def test_a_warned_host_footprint_keeps_the_role(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "_host_memory_refuses", lambda *_a: False)
        estimate = _estimate()
        assert (
            planning._admit_estimate(
                estimate,
                WorkerRole.CHAT,
                _REF,
                total_vram=64 * _GB,
                ram_bytes=40 * _GB,
                engine=LLAMA,
            )
            is estimate
        )


def _estimate():
    from lilbee.providers.fleet.placement import ModelPlacementInput

    return ModelPlacementInput(WorkerRole.CHAT, 4 * _GB)


def test_the_estimators_own_host_figure_is_what_gets_judged(monkeypatch, tmp_path) -> None:
    # Not a stubbed number: the value gguf-parser reported has to be the value
    # admission reads, or the whole bound is decorative.
    from lilbee.providers import engine_params, gguf_meta
    from lilbee.providers.fleet import model_memory as model_memory_mod
    from lilbee.providers.fleet import planning
    from lilbee.providers.fleet.vram import GgufVramEstimate

    seen: list[int] = []
    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
    monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
    monkeypatch.setattr(
        planning, "_host_memory_refuses", lambda _r, _f, ram, _c, _free: bool(seen.append(ram))
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda *_a, **_k: model)
    monkeypatch.setattr(gguf_meta, "read_gguf_metadata", lambda *_a, **_k: {})
    monkeypatch.setattr(
        model_memory_mod,
        "estimate_instance_footprint",
        lambda *_a, **_k: GgufVramEstimate(
            vram_bytes=4 * _GB, unified_bytes=4 * _GB, ram_bytes=7 * _GB
        ),
    )
    estimate = planning._estimate_role(
        WorkerRole.EMBED, _REF, slots=1, device_count=1, engine=LLAMA
    )
    planning._admit_estimate(
        estimate,
        WorkerRole.EMBED,
        _REF,
        total_vram=64 * _GB,
        ram_bytes=estimate.est_ram_bytes,
        engine=LLAMA,
    )
    assert seen == [7 * _GB]


class TestTheSpillWarningFollowsAdmission:
    """A role that is refused must not also be told that it will run."""

    def _spilling(self, monkeypatch, *, total: int, free: int) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 40 * _GB)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: total)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: free)

    def _admit(self):
        from lilbee.providers.fleet import planning

        estimate = _estimate()
        admitted = planning._admit_estimate(
            estimate,
            WorkerRole.CHAT,
            _REF,
            total_vram=24 * _GB,
            ram_bytes=40 * _GB,
            engine=LLAMA,
        )
        return estimate, admitted

    def test_a_refused_spilling_role_logs_only_the_refusal(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        self._spilling(monkeypatch, total=16 * _GB, free=8 * _GB)
        with caplog.at_level(logging.WARNING):
            _, admitted = self._admit()
        assert admitted is None
        assert "cannot load" in caplog.text
        assert _REF in caplog.text
        assert "It will run" not in caplog.text

    def test_an_admitted_spilling_role_is_told_it_will_run_slower(
        self, monkeypatch, offload_configured, caplog
    ) -> None:
        self._spilling(monkeypatch, total=64 * _GB, free=64 * _GB)
        with caplog.at_level(logging.WARNING):
            estimate, admitted = self._admit()
        assert admitted is estimate
        assert "It will run" in caplog.text
        assert _REF in caplog.text

    @pytest.mark.parametrize(
        ("total_vram", "weights", "warned"),
        [
            pytest.param(0, 40 * _GB, False, id="no-gpu-reported"),
            pytest.param(40 * _GB, 40 * _GB, False, id="weights-equal-vram"),
            pytest.param(1, 2, True, id="one-byte-over-the-smallest-gpu"),
        ],
    )
    def test_the_spill_warning_needs_weights_past_a_nonzero_vram(
        self, monkeypatch, offload_configured, caplog, total_vram, weights, warned
    ) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: weights)
        monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
        monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: 64 * _GB)
        with caplog.at_level(logging.WARNING):
            admitted = planning._admit_estimate(
                _estimate(),
                WorkerRole.CHAT,
                _REF,
                total_vram=total_vram,
                ram_bytes=1,
                engine=LLAMA,
            )
        assert admitted is not None
        assert ("It will run" in caplog.text) is warned
