"""Small independent correctness fixes with no shared mechanism."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg


class TestTheChatServeBudgetCannotBecomeADiscount:
    """The factor compensates for chat sizing KV against the smaller budget.

    Above the usable fraction the ratio inverts and the same line starts
    shrinking the footprint, so a user who raises gpu_memory_fraction to fill
    their card gets chat charged for less than it takes.
    """

    @pytest.mark.parametrize("fraction", [0.95, 1.0], ids=["above", "full"])
    def test_a_high_fraction_does_not_shrink_the_charge(self, monkeypatch, fraction: float) -> None:
        from lilbee.providers.fleet.planning import _chat_serve_budget_footprint

        monkeypatch.setattr(cfg, "gpu_memory_fraction", fraction)
        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
        assert _chat_serve_budget_footprint(1000) == 1000

    def test_the_default_still_scales_up(self, monkeypatch) -> None:
        from lilbee.providers.fleet.planning import _chat_serve_budget_footprint

        monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
        monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
        assert _chat_serve_budget_footprint(1000) == 1200


class TestANumCtxPinIsClampedToTheModel:
    """Every unpinned resolver clamps to the trained window; the pin did not,
    and both docstrings claimed it did."""

    def test_a_pin_past_the_trained_window_is_lowered(self, monkeypatch, caplog, tmp_path) -> None:
        import logging

        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.roles import WorkerRole

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        meta = {"architecture": "qwen3", "context_length": "8192"}
        monkeypatch.setattr(cfg, "num_ctx", 200_000)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            ctx = planning_mod._role_ctx(WorkerRole.CHAT, model, meta)
        assert ctx == 8192
        assert "200000" in caplog.text.replace(",", "")

    def test_a_pin_inside_the_window_is_untouched(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.roles import WorkerRole

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        meta = {"architecture": "qwen3", "context_length": "8192"}
        assert planning_mod._role_ctx(WorkerRole.CHAT, model, meta) == 4096

    def test_the_placement_estimate_clamps_the_same_way(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.roles import WorkerRole

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg, "num_ctx", 200_000)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        meta = {"architecture": "qwen3", "context_length": "8192"}
        assert planning_mod._placement_estimate_ctx(WorkerRole.CHAT, model, meta) == 8192


class TestANotAvailableUtilKeepsTheMemoryReading:
    """nvidia-smi prints [N/A] for utilization on some cards and in some VMs.

    Dropping the whole row threw away that card's free and total VRAM too, so a
    GPU that reports memory perfectly well vanished from the panel.
    """

    def test_the_row_survives_with_no_utilization(self) -> None:
        from lilbee.providers.fleet.gpu_backends.nvidia import _parse_smi_output

        samples = _parse_smi_output("0, [N/A], 1024, 4096")
        assert samples[0].utilization_pct is None
        assert samples[0].total_bytes == 4096 * 1024 * 1024

    def test_a_complete_row_is_unchanged(self) -> None:
        from lilbee.providers.fleet.gpu_backends.nvidia import _parse_smi_output

        samples = _parse_smi_output("1, 42, 1024, 4096")
        assert samples[1].utilization_pct == 42

    def test_an_unreadable_index_still_drops_the_row(self) -> None:
        # Without an index there is nothing to attribute the reading to.
        from lilbee.providers.fleet.gpu_backends.nvidia import _parse_smi_output

        assert _parse_smi_output("[N/A], 42, 1024, 4096") == {}

    def test_an_unreadable_header_leaves_the_pin_alone(self, monkeypatch, tmp_path) -> None:
        # Some published GGUFs report no usable context length. Contradicting an
        # explicit pin with the fallback default would break exactly those hosts.
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.roles import WorkerRole

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg, "num_ctx", 200_000)
        monkeypatch.setattr(cfg, "num_ctx_max", None)
        assert planning_mod._role_ctx(WorkerRole.CHAT, model, None) == 200_000

    def test_num_ctx_max_still_caps_a_pin_without_metadata(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.roles import WorkerRole

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(cfg, "num_ctx", 200_000)
        monkeypatch.setattr(cfg, "num_ctx_max", 16_384)
        assert planning_mod._role_ctx(WorkerRole.CHAT, model, None) == 16_384
