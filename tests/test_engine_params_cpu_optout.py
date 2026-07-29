"""An explicit zero gpu-layers means CPU, for every role."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.engine_params import resolve_n_gpu_layers


class TestZeroLayersIsHonouredEverywhere:
    """Setting n_gpu_layers to 0 is how a user says "run this on the CPU".

    The embedding roles took a sentinel that forced full offload before the
    setting was ever read, so embed, rerank and vision kept loading onto the GPU
    the user had just excluded.
    """

    @pytest.mark.parametrize("embedding", [True, False], ids=["embed", "chat"])
    def test_zero_stays_zero(self, monkeypatch, embedding: bool) -> None:
        monkeypatch.setattr(cfg, "n_gpu_layers", 0)
        assert resolve_n_gpu_layers(embedding=embedding) == 0

    def test_an_unset_value_still_means_all_layers(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "n_gpu_layers", None)
        assert resolve_n_gpu_layers(embedding=False) == -1

    def test_a_partial_chat_offload_does_not_reach_the_search_roles(self, monkeypatch) -> None:
        # A chat-shaped layer budget says nothing about a small embed model, which
        # still offloads fully. Only the zero is a statement about the whole host.
        monkeypatch.setattr(cfg, "n_gpu_layers", 20)
        assert resolve_n_gpu_layers(embedding=True) == -1
        assert resolve_n_gpu_layers(embedding=False) == 20
