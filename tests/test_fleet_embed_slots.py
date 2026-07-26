"""Tests for the embed server's continuous-batching slot count."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole


@pytest.fixture(autouse=True)
def _restore_embed_slots():
    before = cfg.embed_slots
    yield
    cfg.embed_slots = before


class TestEmbedSlots:
    def test_defaults_to_one_so_existing_fleets_are_unchanged(self):
        cfg.embed_slots = 0
        assert planning._embed_slots() == 1

    @pytest.mark.parametrize("configured", [2, 4, 8])
    def test_an_explicit_count_is_used(self, configured):
        cfg.embed_slots = configured
        assert planning._embed_slots() == configured

    def test_the_launch_and_the_placement_estimate_agree(self):
        """They must, or placement reserves KV for fewer slots than the server opens.

        --ctx-size is ctx_per_slot x slots, so an estimate that assumes one slot
        while the launch opens eight under-reserves the KV allocation.
        """
        cfg.embed_slots = 8

        estimated = planning._placement_estimate_slots(WorkerRole.EMBED, None)

        assert estimated == planning._embed_slots() == 8

    def test_the_estimate_also_tracks_the_default(self):
        cfg.embed_slots = 0
        assert planning._placement_estimate_slots(WorkerRole.EMBED, None) == 1

    def test_other_roles_are_untouched(self):
        """The finding is about embed's request shape; rerank keeps its single slot."""
        cfg.embed_slots = 8
        assert planning._placement_estimate_slots(WorkerRole.RERANK, None) == 1
