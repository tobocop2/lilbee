"""The fleet-warm-during-ingest signal (bb-opgxa): keep replicas resident so an
unevenly loaded fleet cannot idle-unload and collapse mid-run."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet.ingest_warmth import ingest_keep_warm, keep_fleet_warm
from lilbee.providers.fleet.provider import _warm_ttl_seconds


@pytest.fixture(autouse=True)
def _restore_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestKeepFleetWarm:
    def test_default_ttl_is_the_idle_window(self):
        cfg.keep_engine_warm = False
        cfg.engine_idle_ttl_minutes = 5
        assert not ingest_keep_warm()
        assert _warm_ttl_seconds() == 300

    def test_ingest_holds_the_fleet_resident(self):
        cfg.keep_engine_warm = False
        cfg.engine_idle_ttl_minutes = 5
        with keep_fleet_warm():
            assert ingest_keep_warm()
            assert _warm_ttl_seconds() == 0  # never idle-unload mid-ingest
        # Released after the block: back to the on-demand idle window.
        assert not ingest_keep_warm()
        assert _warm_ttl_seconds() == 300

    def test_keep_engine_warm_pins_weights_resident(self):
        # keep_engine_warm keeps the model resident (ttl 0), not merely alive
        # across restarts: a user who opted to keep it warm never waits out a
        # mid-session reload. Independent of an active ingest.
        cfg.keep_engine_warm = True
        cfg.engine_idle_ttl_minutes = 5
        assert not ingest_keep_warm()
        assert _warm_ttl_seconds() == 0

    def test_nested_scopes_restore_correctly(self):
        cfg.keep_engine_warm = False
        cfg.engine_idle_ttl_minutes = 10
        with keep_fleet_warm():
            with keep_fleet_warm():
                assert _warm_ttl_seconds() == 0
            assert _warm_ttl_seconds() == 0  # still inside the outer hold
        assert _warm_ttl_seconds() == 600

    def test_max_inflight_override_raises_admission(self):
        # bb-emkt0: on a multi-GPU box the auto cap is CPU-bound; the override
        # lets more files run their compute phase so every embed replica is fed.
        # A positive override returns early, before any fleet/CPU sizing call.
        from lilbee.data.ingest.pipeline import _max_concurrent

        cfg.vision_model = ""
        cfg.ingest_max_inflight = 48
        assert _max_concurrent() == 48

    def test_auto_admission_scales_with_the_embed_fleet(self, monkeypatch):
        # 0 = auto: admission scales with the detected embed replicas so a
        # multi-GPU box is fed without a manual cap.
        from lilbee.data.ingest import offload, pipeline

        monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
        cfg.vision_model = ""
        cfg.ingest_max_inflight = 0
        monkeypatch.setattr(pipeline, "cpu_quota", lambda: 8)
        # Emulate an 8-replica fleet: 8 * _EMBED_INFLIGHT_PER_REPLICA (8) = 64.
        # Patch both bound names (pipeline imported the helper by name).
        monkeypatch.setattr(pipeline, "embed_inflight_target", lambda: 64)
        monkeypatch.setattr(offload, "embed_inflight_target", lambda: 64)
        assert pipeline._max_concurrent() == 64  # max(cpu_quota=8, 64)
        assert offload._max_workers() == 64  # pool lifts to feed it too

    def test_embed_inflight_target_single_card_is_zero(self, monkeypatch):
        from lilbee.data.ingest import offload

        monkeypatch.setattr(
            "lilbee.providers.fleet.replicas.resolve_replica_count", lambda *_a, **_k: 1
        )
        monkeypatch.setattr("lilbee.providers.fleet.replicas.gpu_device_count", lambda: 1)
        assert offload.embed_inflight_target() == 0  # single card: CPU sizing fits

    def test_max_inflight_lifts_the_worker_pool_ceiling(self, monkeypatch):
        # In adaptive mode the admission gate's permit_max is max_workers(), so
        # the override must raise the pool too or a multi-GPU fleet stays clamped
        # at 32. An explicit LILBEE_INGEST_MAX_WORKERS still wins.
        from lilbee.data.ingest.offload import _max_workers

        monkeypatch.delenv("LILBEE_INGEST_MAX_WORKERS", raising=False)
        cfg.ingest_max_inflight = 0
        assert _max_workers() <= 32  # default cap
        cfg.ingest_max_inflight = 96
        assert _max_workers() == 96

    def test_signal_propagates_into_a_worker_thread(self):
        # to_ingest_thread copies the context, so the fleet (which spawns on an
        # ingest worker thread) sees the hold. Emulate that with copy_context.
        import contextvars

        cfg.keep_engine_warm = False
        cfg.engine_idle_ttl_minutes = 5
        with keep_fleet_warm():
            ctx = contextvars.copy_context()
        # Outside the block the live context is released...
        assert _warm_ttl_seconds() == 300
        # ...but the snapshot captured mid-ingest still reports the hold.
        assert ctx.run(_warm_ttl_seconds) == 0
