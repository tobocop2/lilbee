"""What placement charges when the estimator cannot answer."""

from __future__ import annotations

from lilbee.providers.roles import WorkerRole

_GB = 1024**3


class TestTheFallbackIncludesTheCacheItWillAllocate:
    """Charging weight bytes alone is a knowing under-charge.

    The engine allocates weights plus a KV cache sized by context and slots plus
    compute buffers. Charging only the first means placement fits a model that
    cannot fit, and the load, which the comment says decides, decides by OOMing.
    """

    def test_the_floor_exceeds_the_weights_alone(self, monkeypatch) -> None:
        from lilbee.providers.fleet.planning import _analytic_footprint_floor

        weights = 4 * _GB
        floor = _analytic_footprint_floor(
            weights,
            meta={
                "block_count": "32",
                "head_count_kv": "8",
                "key_length": "128",
                "value_length": "128",
            },
            ctx=8192,
            slots=4,
        )
        assert floor > weights

    def test_more_slots_cost_more(self, monkeypatch) -> None:
        from lilbee.providers.fleet.planning import _analytic_footprint_floor

        meta = {
            "block_count": "32",
            "head_count_kv": "8",
            "key_length": "128",
            "value_length": "128",
        }
        one = _analytic_footprint_floor(4 * _GB, meta=meta, ctx=8192, slots=1)
        four = _analytic_footprint_floor(4 * _GB, meta=meta, ctx=8192, slots=4)
        assert four > one

    def test_unknown_metadata_still_charges_more_than_weights(self) -> None:
        # No header to size KV from; the fallback per-token estimate still applies,
        # because zero KV is the one answer that is certainly wrong.
        from lilbee.providers.fleet.planning import _analytic_footprint_floor

        assert _analytic_footprint_floor(4 * _GB, meta=None, ctx=8192, slots=1) > 4 * _GB


class TestTheFallbackIsUsedAndSaidOutLoud:
    def test_a_failed_estimate_charges_the_floor_not_the_file_size(
        self, monkeypatch, caplog, tmp_path
    ) -> None:
        import logging

        from lilbee.providers.fleet import planning as planning_mod

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 4096)
        monkeypatch.setattr(planning_mod, "_role_weights_bytes", lambda _r, _ref: 4 * _GB)
        monkeypatch.setattr(planning_mod, "_ref_is_moe", lambda _ref: False)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            placed = planning_mod._sizing_failure_fallback(
                WorkerRole.CHAT,
                "org/m.gguf",
                RuntimeError("parser died"),
                device_count=1,
                total_vram=80 * _GB,
            )
        assert placed is not None
        assert placed.est_vram_bytes > 4 * _GB
        assert "cache" in caplog.text.lower()


class TestTheEstimateIsKeyedToTheEngineThatPricesIt:
    """The memo key held the model, the sizing and the parser's arguments, and no
    trace of which engine the numbers describe. Swap the engine, keep the answers."""

    def test_a_different_engine_build_does_not_reuse_the_old_estimate(
        self, monkeypatch, tmp_path
    ) -> None:
        from lilbee.providers.fleet import vram as vram_mod

        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")
        vram_mod._cached_footprint.cache_clear()
        runs: list[str] = []

        def _fake_parser(argv, _path):
            runs.append("run")
            return (
                '{"estimate": {"items": [{"ram": {"uma": 0, "nonuma": 0},'
                ' "vrams": [{"uma": 1, "nonuma": 1}]}]}}'
            )

        monkeypatch.setattr(vram_mod, "_run_parser", _fake_parser)
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: tmp_path / "gguf-parser")

        from lilbee.core.config.enums import KvCacheType

        kwargs = {
            "ctx": 4096,
            "slots": 1,
            "gpu_layers": -1,
            "flash_attn": True,
            "kv_cache_type": KvCacheType.F16,
        }
        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "wheel:1")
        vram_mod.estimate_instance_footprint(model, **kwargs)
        vram_mod.estimate_instance_footprint(model, **kwargs)
        assert len(runs) == 1, "same engine should hit the memo"

        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "custom:/opt/other")
        vram_mod.estimate_instance_footprint(model, **kwargs)
        assert len(runs) == 2, "a different engine must re-price"


class TestTheFloorWhenTheModelFileIsGone:
    """The estimator failing and the file being unreadable are different failures."""

    def test_an_unreadable_model_still_gets_a_cache_charge(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet import planning as planning_mod

        def _missing(_ref):
            raise ProviderError("not installed")

        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _missing)
        floor = planning_mod._fallback_floor_for(WorkerRole.CHAT, "org/m.gguf", 4 * _GB)
        # No header to read, so the per-token fallback and a usable-context floor
        # apply rather than charging the weights alone.
        assert floor > 4 * _GB
