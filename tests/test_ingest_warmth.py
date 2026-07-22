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

    def test_keep_engine_warm_still_wins_when_not_ingesting(self):
        cfg.keep_engine_warm = True
        cfg.engine_idle_ttl_minutes = 5
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
