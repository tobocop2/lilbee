"""Tests for gguf-parser-backed, UMA-aware instance memory estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.vram import (
    GgufVramEstimate,
    estimate_instance_footprint,
)


def _sample_json(*, ram_uma: int, ram_nonuma: int, vrams: list[tuple[int, int]]) -> str:
    """gguf-parser-shaped JSON with the given ram and (uma, nonuma) vram devices."""
    return json.dumps(
        {
            "estimate": {
                "items": [
                    {
                        "ram": {"uma": ram_uma, "nonuma": ram_nonuma},
                        "vrams": [{"uma": u, "nonuma": n} for u, n in vrams],
                    }
                ]
            }
        }
    )


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A stand-in GGUF file (content irrelevant; the parser is mocked)."""
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return path


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    vram_mod._cached_footprint.cache_clear()


def _patch_parser(
    monkeypatch: pytest.MonkeyPatch, *, stdout: str, recorder: list[list[str]] | None = None
) -> None:
    """Stub out the gguf-parser binary path and its subprocess invocation."""
    monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))

    def fake_run(argv: list[str], **_kwargs: object) -> tuple[str, int]:
        if recorder is not None:
            recorder.append(argv)
        return stdout, 0

    monkeypatch.setattr(vram_mod, "run_bounded", fake_run)


class TestFootprint:
    def test_selects_memory_model(self) -> None:
        est = GgufVramEstimate(vram_bytes=10, ram_bytes=3, unified_bytes=7)
        assert est.footprint(unified=False) == 10
        assert est.footprint(unified=True) == 7


class TestEstimateInstanceFootprint:
    def test_parses_uma_and_nonuma(self, model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=100, ram_nonuma=250, vrams=[(700, 2000)]),
        )
        est = estimate_instance_footprint(
            model_file,
            ctx=4096,
            slots=2,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
        )
        # Discrete GPU charges the device VRAM; unified host charges everything resident.
        assert est.vram_bytes == 2000
        assert est.ram_bytes == 250
        assert est.unified_bytes == 100 + 700

    def test_sums_multiple_vram_devices(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=10, ram_nonuma=20, vrams=[(300, 1000), (400, 1500)]),
        )
        est = estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        assert est.vram_bytes == 1000 + 1500
        assert est.unified_bytes == 10 + 300 + 400

    def test_passes_sizing_flags_to_parser(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=calls,
        )
        estimate_instance_footprint(
            model_file,
            ctx=8192,
            slots=4,
            gpu_layers=33,
            flash_attn=True,
            kv_cache_type=KvCacheType.Q8_0,
        )
        argv = calls[0]
        assert argv[argv.index("--ctx-size") + 1] == "8192"
        assert argv[argv.index("--parallel") + 1] == "4"
        assert argv[argv.index("--gpu-layers") + 1] == "33"
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
        assert "--flash-attention" in argv
        assert "--mmproj-path" not in argv

    def test_tensor_split_passes_ratio_and_returns_per_device(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A multi-GPU split sends the ratio so gguf-parser breaks the estimate out
        # per card; the peak (busiest card) gates placement, not the summed total.
        calls: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=2, vrams=[(10, 100), (10, 110), (10, 120)]),
            recorder=calls,
        )
        est = estimate_instance_footprint(
            model_file,
            ctx=131072,
            slots=4,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
            tensor_split=(1, 1, 1),
        )
        argv = calls[0]
        assert argv[argv.index("--tensor-split") + 1] == "1,1,1"
        assert argv[argv.index("--split-mode") + 1] == "layer"
        assert est.per_device_vram == (100, 110, 120)
        assert est.peak_footprint(unified=False) == 120
        assert est.vram_bytes == 100 + 110 + 120

    def test_disabled_flash_passes_no_flash_flag(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=calls,
        )
        estimate_instance_footprint(
            model_file,
            ctx=512,
            slots=1,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        assert "--no-flash-attention" in calls[0]
        assert "--flash-attention" not in calls[0]

    def test_includes_mmproj_when_given(
        self, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"GGUF")
        calls: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=calls,
        )
        estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
            mmproj_path=mmproj,
        )
        assert calls[0][calls[0].index("--mmproj-path") + 1] == str(mmproj)

    def test_memoizes_until_mtime_changes(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=calls,
        )
        kwargs = {
            "ctx": 2048,
            "slots": 1,
            "gpu_layers": -1,
            "flash_attn": True,
            "kv_cache_type": KvCacheType.F16,
        }
        estimate_instance_footprint(model_file, **kwargs)
        estimate_instance_footprint(model_file, **kwargs)
        assert len(calls) == 1  # second call served from the cache
        # Re-touch the file: a re-pull at the same path must invalidate the cache.
        # Bump by a full second so Windows' coarse mtime resolution registers it.
        import os

        st = model_file.stat()
        os.utime(model_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        estimate_instance_footprint(model_file, **kwargs)
        assert len(calls) == 2

    def _estimate(self, model_file: Path) -> None:
        estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
        )

    def test_a_nonzero_exit_raises_provider_error(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        monkeypatch.setattr(vram_mod, "run_bounded", lambda *a, **k: ("", 1))
        with pytest.raises(ProviderError, match="failed to run"):
            self._estimate(model_file)

    def test_a_spawn_or_timeout_failure_raises_provider_error(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))

        def boom(*_a: object, **_k: object) -> tuple[str, int]:
            raise OSError("no such binary")

        monkeypatch.setattr(vram_mod, "run_bounded", boom)
        with pytest.raises(ProviderError, match="failed to run"):
            self._estimate(model_file)

    def test_unparseable_output_raises_provider_error_with_cause(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The message carries the actual parse failure, so a future schema change
        # reads as a one-line log instead of a wrong-subsystem debugging session.
        _patch_parser(monkeypatch, stdout="not json at all")
        with pytest.raises(ProviderError, match="unexpected estimator output"):
            estimate_instance_footprint(
                model_file,
                ctx=2048,
                slots=1,
                gpu_layers=-1,
                flash_attn=True,
                kv_cache_type=KvCacheType.F16,
            )

    def test_parses_renamed_memory_schema(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Upstream gguf-parser renamed estimate.items to estimate.memory after
        # v0.24.1; an engine built past the pin must still size.
        renamed = _sample_json(ram_uma=100, ram_nonuma=250, vrams=[(700, 2000)]).replace(
            '"items"', '"memory"'
        )
        _patch_parser(monkeypatch, stdout=renamed)
        est = estimate_instance_footprint(
            model_file,
            ctx=4096,
            slots=2,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
        )
        assert est.vram_bytes == 2000
        assert est.ram_bytes == 250
        assert est.unified_bytes == 100 + 700


class TestBatchSizeFlags:
    def test_batch_size_adds_batch_and_ubatch_flags(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=recorder,
        )
        estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
            batch_size=2048,
        )
        (argv,) = recorder
        assert argv[argv.index("--batch-size") + 1] == "2048"
        assert argv[argv.index("--ubatch-size") + 1] == "2048"

    def test_no_batch_flags_by_default(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=recorder,
        )
        estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        (argv,) = recorder
        assert "--batch-size" not in argv
        assert "--ubatch-size" not in argv

    def test_batch_size_participates_in_the_memo_key(
        self, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder: list[list[str]] = []
        _patch_parser(
            monkeypatch,
            stdout=_sample_json(ram_uma=1, ram_nonuma=1, vrams=[(1, 1)]),
            recorder=recorder,
        )
        common = {
            "ctx": 2048,
            "slots": 1,
            "gpu_layers": -1,
            "flash_attn": False,
            "kv_cache_type": KvCacheType.F16,
        }
        estimate_instance_footprint(model_file, **common)
        estimate_instance_footprint(model_file, **common, batch_size=2048)
        assert len(recorder) == 2  # a different batch size is a different estimate


class TestProjectorCorrection:
    """A multimodal projector is charged at its unified-memory delta, floored at its
    weights, instead of gguf-parser's inflated discrete-GPU merge (v0.24.x adds ~10
    GiB of phantom compute buffer to any model+mmproj estimate)."""

    _GB = 1024**3

    def _patch_two_run_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        base_json: str,
        mmproj_json: str,
        recorder: list[list[str]] | None = None,
    ) -> None:
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))

        def fake_run(argv: list[str], **_kwargs: object) -> tuple[str, int]:
            if recorder is not None:
                recorder.append(argv)
            return (mmproj_json if "--mmproj-path" in argv else base_json), 0

        monkeypatch.setattr(vram_mod, "run_bounded", fake_run)

    def _estimate(self, model_file: Path, mmproj: Path) -> GgufVramEstimate:
        return estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
            mmproj_path=mmproj,
        )

    def test_projector_vram_is_the_uma_delta_not_the_nonuma_merge(
        self, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without the projector: 3GB nonuma / 2GB uma. With it the parser claims
        # 13GB nonuma but a sane 2.8GB uma. The projector must cost the 0.8GB the
        # unified model attributes to it, not the 10GB phantom.
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"G" * 1024)  # far below the uma delta
        self._patch_two_run_parser(
            monkeypatch,
            base_json=_sample_json(ram_uma=0, ram_nonuma=0, vrams=[(2 * self._GB, 3 * self._GB)]),
            mmproj_json=_sample_json(
                ram_uma=0, ram_nonuma=0, vrams=[(int(2.8 * self._GB), 13 * self._GB)]
            ),
        )
        est = self._estimate(model_file, mmproj)
        assert est.vram_bytes == 3 * self._GB + int(0.8 * self._GB)
        # The unified estimate was never inflated; it passes through untouched.
        assert est.unified_bytes == int(2.8 * self._GB)

    def test_projector_charge_is_floored_at_its_weights(
        self, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A projector whose weights are mmap-shared can show a near-zero uma delta,
        # but offloaded weights still occupy VRAM: floor the charge at the file size.
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"G" * 4096)  # weights larger than the 1000-byte uma delta
        self._patch_two_run_parser(
            monkeypatch,
            base_json=_sample_json(ram_uma=0, ram_nonuma=0, vrams=[(2_000, 3_000)]),
            mmproj_json=_sample_json(ram_uma=0, ram_nonuma=0, vrams=[(3_000, 13_000)]),
        )
        est = self._estimate(model_file, mmproj)
        assert est.vram_bytes == 3_000 + 4096  # base plus the projector weights floor

    def test_projector_lands_on_the_first_device_of_a_split(
        self, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"G" * 4096)
        self._patch_two_run_parser(
            monkeypatch,
            base_json=_sample_json(
                ram_uma=0,
                ram_nonuma=0,
                vrams=[(2_000, 3_000), (2_000, 3_000)],
            ),
            mmproj_json=_sample_json(
                ram_uma=0,
                ram_nonuma=0,
                vrams=[(2_000, 9_000), (2_000, 9_000)],
            ),
        )
        est = estimate_instance_footprint(
            model_file,
            ctx=2048,
            slots=1,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
            mmproj_path=mmproj,
            tensor_split=(1, 1),
        )
        assert est.per_device_vram == (3_000 + 4096, 3_000)


def test_the_v_cache_type_reaches_the_estimator_command_line(monkeypatch, tmp_path) -> None:
    """K and V differ wherever flash attention is left to the engine.

    An estimate that reuses K for V sizes a cache the server will not allocate.
    """
    from lilbee.core.config.enums import KvCacheType
    from lilbee.providers.fleet import vram as vram_mod

    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 64)
    captured: list[list[str]] = []

    def _capture(argv, _path):
        captured.append(argv)
        return '{"estimate": {}}'

    monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
    monkeypatch.setattr(vram_mod, "_run_parser", _capture)
    monkeypatch.setattr(vram_mod, "_parse_estimate", lambda *_a: object())
    vram_mod._cached_footprint.cache_clear()

    vram_mod.estimate_instance_footprint(
        model,
        ctx=4096,
        slots=1,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.Q8_0,
        kv_cache_type_v=KvCacheType.F16,
    )

    (argv,) = captured
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "f16"
