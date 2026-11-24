"""A CPU llama.cpp load is charged its measured peak resident size, within a tenth above it.

The peaks are the highest resident size of the llama.cpp CPU server (pack 1.12032.1, a 32-thread
AVX2 host) after every slot took a prompt of the context length minus 32 tokens. The parser figures
are its recorded answers for the same launches at the default ubatch. Each case builds its request
the way the planner builds it, from the launch's flash mode and cache type.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import model_memory as model_memory_mod
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.openvino import OPENVINO
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.fleet.model_memory import CPU_RUNTIME_ALLOWANCE_BYTES
from lilbee.providers.roles import WorkerRole

BAND = 1.1


@dataclass(frozen=True)
class Model:
    file_bytes: int
    repacked_bytes: int
    heads: int
    kv_heads: int
    vocab: int


MODELS = {
    "llama-1b-q8": Model(1_321_083_008, 0, 32, 8, 128_256),
    "llama-1b-q4km": Model(807_694_464, 467_140_608, 32, 8, 128_256),
    "qwen-7b-q4km": Model(4_683_074_240, 3_427_909_632, 28, 4, 152_064),
    "llama-8b-q4km": Model(4_920_739_232, 3_655_139_328, 32, 8, 128_256),
}


class Figures(NamedTuple):
    """The parser's unified and device bytes, at the default ubatch and at one row per slot."""

    uma: int
    vram: int
    slot_uma: int
    slot_vram: int


OTHER = Model(0, 0, 1, 1, 1)

# (model, slots, ctx, flash on, cache) -> the parser's figures.
PARSER = {
    ("llama-1b-q8", 1, 4096, True, "q8_0"): Figures(
        677_008_664, 392_693_760, 135_696_676, 270_610_436
    ),
    ("llama-1b-q8", 1, 4096, True, "f16"): Figures(
        739_923_224, 396_625_920, 202_543_396, 262_316_036
    ),
    ("llama-1b-q8", 1, 4096, False, "f16"): Figures(
        739_923_224, 564_135_936, 202_543_396, 271_106_052
    ),
    ("llama-1b-q8", 2, 4096, True, "q8_0"): Figures(
        756_700_440, 409_733_120, 212_034_864, 279_126_024
    ),
    ("llama-1b-q8", 2, 4096, True, "f16"): Figures(
        882_529_560, 417_597_440, 345_728_304, 262_537_224
    ),
    ("llama-1b-q8", 2, 4096, False, "f16"): Figures(
        882_529_560, 849_348_608, 345_728_304, 281_149_448
    ),
    ("llama-1b-q4km", 1, 4096, True, "q8_0"): Figures(
        677_008_664, 384_567_296, 135_696_676, 270_610_436
    ),
    ("llama-1b-q4km", 1, 4096, True, "f16"): Figures(
        739_923_224, 388_499_456, 202_543_396, 262_316_036
    ),
    ("llama-1b-q4km", 1, 4096, False, "f16"): Figures(
        739_923_224, 564_135_936, 202_543_396, 271_106_052
    ),
    ("qwen-7b-q4km", 2, 8192, True, "q8_0"): Figures(
        1_224_824_088, 575_555_584, 565_620_016, 296_075_272
    ),
    ("qwen-7b-q4km", 2, 8192, True, "f16"): Figures(
        1_665_226_008, 591_284_224, 1_021_750_576, 262_967_304
    ),
    ("qwen-7b-q4km", 2, 8192, False, "f16"): Figures(
        1_665_226_008, 1_298_139_136, 1_021_750_576, 299_614_216
    ),
    ("llama-8b-q4km", 2, 8192, True, "f16"): Figures(
        2_791_612_360, 548_407_296, 2_275_880_928, 262_881_288
    ),
    ("llama-8b-q4km", 2, 8192, False, "f16"): Figures(
        2_791_612_360, 1_470_105_600, 2_275_880_928, 333_709_320
    ),
}


@dataclass(frozen=True)
class Cell:
    model: str
    slots: int
    ctx: int
    flash: str
    cache: str
    peak: int


CELLS = {
    "llama-1b-q8-on-q8": Cell("llama-1b-q8", 1, 4096, "on", "q8_0", 1_545_236_480),
    "llama-1b-q8-on-f16": Cell("llama-1b-q8", 1, 4096, "on", "f16", 1_608_118_272),
    "llama-1b-q8-auto": Cell("llama-1b-q8", 1, 4096, "auto", "f16", 1_608_105_984),
    "llama-1b-q8-off": Cell("llama-1b-q8", 1, 4096, "off", "f16", 1_962_184_704),
    "llama-1b-q8-2slot-on-q8": Cell("llama-1b-q8", 2, 4096, "on", "q8_0", 1_618_427_904),
    "llama-1b-q8-2slot-auto": Cell("llama-1b-q8", 2, 4096, "auto", "f16", 1_742_745_600),
    "llama-1b-q8-2slot-off": Cell("llama-1b-q8", 2, 4096, "off", "f16", 2_096_865_280),
    "llama-1b-q4km-on-q8": Cell("llama-1b-q4km", 1, 4096, "on", "q8_0", 1_520_324_608),
    "llama-1b-q4km-off": Cell("llama-1b-q4km", 1, 4096, "off", "f16", 1_919_549_440),
    "qwen-7b-q4km-on-q8": Cell("qwen-7b-q4km", 2, 8192, "on", "q8_0", 8_548_925_440),
    "qwen-7b-q4km-off": Cell("qwen-7b-q4km", 2, 8192, "off", "f16", 9_551_179_776),
    "qwen-7b-q4km-auto": Cell("qwen-7b-q4km", 2, 8192, "auto", "f16", 8_986_632_192),
    "llama-8b-q4km-off": Cell("llama-8b-q4km", 2, 8192, "off", "f16", 11_346_698_240),
    "llama-8b-q4km-auto": Cell("llama-8b-q4km", 2, 8192, "auto", "f16", 10_650_578_944),
}
DEFAULT_UBATCH = 512


def _header(model: Model) -> dict[str, object]:
    """The slice of the parser's raw header that the shape reader uses."""
    return {
        "header": {
            "metadataKV": [
                {"key": "general.architecture", "valueType": 8, "value": "llama"},
                {"key": "llama.attention.head_count", "valueType": 4, "value": model.heads},
                {"key": "llama.attention.head_count_kv", "valueType": 4, "value": model.kv_heads},
                {
                    "key": "tokenizer.ggml.tokens",
                    "valueType": 9,
                    "value": {"type": 8, "len": model.vocab, "startOffset": 1224, "size": 1},
                },
            ]
        }
    }


class _Parser:
    """A stand-in for the estimator binary that answers by the flags it is given."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.argvs: list[list[str]] = []
        self.header: dict[str, object] | None = None

    def __call__(self, argv: list[str], _path: str) -> str:
        self.argvs.append(argv)
        model = MODELS[self.name]
        if "--raw" in argv:
            return json.dumps(self.header or _header(model))
        slots = int(argv[argv.index("--parallel") + 1])
        ctx = int(argv[argv.index("--ctx-size") + 1]) // slots
        flash = "--no-flash-attention" not in argv
        cache = argv[argv.index("--cache-type-k") + 1]
        figures = PARSER[(self.name, slots, ctx, flash, cache)]
        if "--ubatch-size" in argv:
            assert int(argv[argv.index("--ubatch-size") + 1]) == slots
            uma, vram = figures.slot_uma, figures.slot_vram
        else:
            uma, vram = figures.uma, figures.vram
        return json.dumps(
            {
                "metadata": {"fileSize": model.file_bytes},
                "estimate": {
                    "items": [
                        {"ram": {"uma": uma, "nonuma": uma}, "vrams": [{"uma": 0, "nonuma": vram}]}
                    ]
                },
            }
        )

    def sized(self) -> list[list[str]]:
        return [argv for argv in self.argvs if "--ctx-size" in argv]


ParserFor = Callable[[str], tuple[Path, _Parser]]


@pytest.fixture
def parser_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParserFor:
    def install(name: str) -> tuple[Path, _Parser]:
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(b"GGUF")
        vram_mod._cached_footprint.cache_clear()
        vram_mod._cached_attention_shape.cache_clear()
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        monkeypatch.setattr(
            model_memory_mod, "repacked_weight_bytes", lambda _p: MODELS[name].repacked_bytes
        )
        fake = _Parser(name)
        monkeypatch.setattr(vram_mod, "_run_parser", fake)
        return path, fake

    return install


def _launch_as_planned(monkeypatch: pytest.MonkeyPatch, flash: str, cache: str) -> None:
    """Set the config and backend that make the planner launch chat with *flash* and *cache*."""
    monkeypatch.setattr(cfg, "flash_attention", {"on": True, "off": False, "auto": None}[flash])
    monkeypatch.setattr(
        planning_mod, "_fleet_backend", lambda _e: "Vulkan" if flash == "auto" else "CUDA"
    )
    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType(cache))


def _planned_request(path: Path, cell: Cell) -> MemoryRequest:
    """The request the planner sends for a CPU-placed chat role."""
    return MemoryRequest(
        model_path=path,
        ctx=cell.ctx,
        slots=cell.slots,
        gpu_layers=0,
        flash_attn=planning_mod._role_flash(WorkerRole.CHAT, engine=LLAMA, on_cpu=True),
        kv_cache_type=planning_mod._role_kv_cache_type(WorkerRole.CHAT, engine=LLAMA),
        on_cpu=True,
    )


def _charge(request: MemoryRequest) -> int:
    return LLAMA.estimate_memory(request).resident_footprint(unified=True)


def _scores(cell: Cell) -> int:
    model = MODELS[cell.model]
    return DEFAULT_UBATCH * cell.ctx * (4 * model.heads + 6 * model.kv_heads)


class TestTheChargeAgainstThePeak:
    @pytest.mark.parametrize("case", sorted(CELLS))
    def test_it_covers_the_peak_and_stays_within_a_tenth_above(
        self, case: str, parser_for: ParserFor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = CELLS[case]
        path, _ = parser_for(cell.model)
        _launch_as_planned(monkeypatch, cell.flash, cell.cache)
        charge = _charge(_planned_request(path, cell))
        assert cell.peak <= charge <= cell.peak * BAND

    def test_a_card_charge_is_the_parsers_own_figure_and_file(self, parser_for: ParserFor) -> None:
        cell = CELLS["qwen-7b-q4km-off"]
        path, fake = parser_for(cell.model)
        request = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=False,
            kv_cache_type=KvCacheType.F16, on_cpu=False,
        )  # fmt: skip
        charge = _charge(request)
        assert charge == (
            PARSER[(cell.model, cell.slots, cell.ctx, False, "f16")].uma
            + MODELS[cell.model].file_bytes
        )
        assert len(fake.sized()) == 1


class TestWhatTheCpuChargeAsksFor:
    def test_the_parser_is_asked_once_at_the_default_ubatch(
        self, parser_for: ParserFor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = CELLS["qwen-7b-q4km-off"]
        path, fake = parser_for(cell.model)
        _launch_as_planned(monkeypatch, cell.flash, cell.cache)
        _charge(_planned_request(path, cell))
        sized = fake.sized()
        assert len(sized) == 1
        assert "--ubatch-size" not in sized[0]

    @pytest.mark.parametrize("slots", [1, 2])
    def test_the_logits_are_one_row_per_slot(
        self, slots: int, parser_for: ParserFor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = Cell("llama-1b-q8", slots, 4096, "on", "f16", 0)
        path, _ = parser_for(cell.model)
        _launch_as_planned(monkeypatch, "on", "f16")
        model = MODELS[cell.model]
        assert _charge(_planned_request(path, cell)) == (
            PARSER[(cell.model, slots, 4096, True, "f16")].uma
            - (DEFAULT_UBATCH - slots) * model.vocab * 8
            + model.file_bytes
            + CPU_RUNTIME_ALLOWANCE_BYTES
        )

    def test_the_runtime_allowance_is_96_mib(self) -> None:
        assert CPU_RUNTIME_ALLOWANCE_BYTES == 96 * 1024 * 1024

    def test_the_openvino_engine_keeps_the_estimators_own_figure(
        self, parser_for: ParserFor
    ) -> None:
        cell = CELLS["llama-1b-q8-off"]
        path, fake = parser_for(cell.model)
        request = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=False,
            kv_cache_type=KvCacheType.F16, on_cpu=True,
        )  # fmt: skip
        charge = OPENVINO.estimate_memory(request).resident_footprint(unified=True)
        figures = PARSER[(cell.model, 1, 4096, False, "f16")]
        assert charge == figures.uma + MODELS[cell.model].file_bytes
        assert len(fake.sized()) == 1

    def test_a_pooled_launch_keeps_its_own_batch(self, parser_for: ParserFor) -> None:
        cell = CELLS["llama-1b-q8-off"]
        path, fake = parser_for(cell.model)
        request = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=False,
            kv_cache_type=KvCacheType.F16, batch_size=1, on_cpu=True,
        )  # fmt: skip
        charge = _charge(request)
        sized = fake.sized()
        assert [argv[argv.index("--ubatch-size") + 1] for argv in sized] == ["1"]
        figures = PARSER[(cell.model, 1, 4096, False, "f16")]
        # The parser books the batch buffer to the device; a CPU load holds it in memory.
        assert charge == figures.slot_uma + figures.slot_vram + MODELS[cell.model].file_bytes

    def test_a_device_booked_below_the_unified_figure_adds_no_buffer(self) -> None:
        """A device figure below the unified one is clamped away, not a negative buffer."""
        parsed = vram_mod.GgufVramEstimate(
            vram_bytes=0,
            ram_bytes=0,
            unified_bytes=0,
            per_device_vram=(100, 900),
            per_device_unified=(300, 800),
        )
        assert model_memory_mod._device_booked_buffer_bytes(parsed) == 100


class TestTheScoreTerm:
    def _off_minus_on(self, cell: Cell, parser_for: ParserFor) -> int:
        path, _ = parser_for(cell.model)
        kv = KvCacheType.F16
        on = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=True,
            kv_cache_type=kv, on_cpu=True,
        )  # fmt: skip
        off = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=False,
            kv_cache_type=kv, on_cpu=True,
        )  # fmt: skip
        return _charge(off) - _charge(on)

    def test_it_is_512_rows_of_the_context_per_slot_times_the_heads(
        self, parser_for: ParserFor
    ) -> None:
        cell = CELLS["llama-8b-q4km-off"]
        assert self._off_minus_on(cell, parser_for) == _scores(cell)

    def test_it_is_the_same_at_one_slot_and_two_for_the_same_context_per_slot(
        self, parser_for: ParserFor
    ) -> None:
        one = Cell("llama-1b-q8", 1, 4096, "off", "f16", 0)
        two = CELLS["llama-1b-q8-2slot-off"]
        assert self._off_minus_on(one, parser_for) == self._off_minus_on(two, parser_for)
        assert self._off_minus_on(one, parser_for) == _scores(one) > 0

    def test_it_is_zero_with_flash_on(self, parser_for: ParserFor) -> None:
        cell = CELLS["llama-1b-q8-on-f16"]
        path, _ = parser_for(cell.model)
        on = MemoryRequest(
            model_path=path, ctx=cell.ctx, slots=cell.slots, gpu_layers=0, flash_attn=True,
            kv_cache_type=KvCacheType.F16, on_cpu=True,
        )  # fmt: skip
        model = MODELS[cell.model]
        assert _charge(on) == (
            PARSER[(cell.model, 1, 4096, True, "f16")].uma
            - (DEFAULT_UBATCH - 1) * model.vocab * 8
            + model.file_bytes
            + CPU_RUNTIME_ALLOWANCE_BYTES
        )


class TestTheAttentionShape:
    def test_it_reads_the_heads_and_the_vocabulary_from_the_raw_header(
        self, parser_for: ParserFor
    ) -> None:
        path, _ = parser_for("qwen-7b-q4km")
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(28, 4, 152_064)

    def test_the_parser_is_asked_once_per_file(self, parser_for: ParserFor) -> None:
        path, fake = parser_for("qwen-7b-q4km")
        vram_mod.attention_shape(path)
        vram_mod.attention_shape(path)
        assert Counter("--raw" in argv for argv in fake.argvs) == {True: 1}

    def test_the_parser_is_asked_about_the_model_file(self, parser_for: ParserFor) -> None:
        path, fake = parser_for("qwen-7b-q4km")
        vram_mod.attention_shape(path)
        assert fake.argvs == [[str(Path("/fake/gguf-parser")), "--path", str(path), "--raw"]]

    def test_a_rewritten_file_is_read_again(
        self, parser_for: ParserFor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path, _ = parser_for("qwen-7b-q4km")
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(28, 4, 152_064)
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps(_header(OTHER)))
        written = path.stat().st_mtime_ns
        path.write_bytes(b"GGUF2")
        # A clock that ticks in 16 ms steps, as on Windows, can give both writes one mtime.
        os.utime(path, ns=(written + 1_000_000_000, written + 1_000_000_000))
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(1, 1, 1)

    def test_another_engine_build_is_read_again(
        self, parser_for: ParserFor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path, _ = parser_for("qwen-7b-q4km")
        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "build-a")
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(28, 4, 152_064)
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps(_header(OTHER)))
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(28, 4, 152_064)
        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "build-b")
        assert vram_mod.attention_shape(path) == vram_mod.AttentionShape(1, 1, 1)

    def test_unreadable_raw_output_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(b"GGUF")
        vram_mod._cached_attention_shape.cache_clear()
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps({"header": {}}))
        with pytest.raises(ProviderError, match="unreadable output"):
            vram_mod.attention_shape(path)


LFM2_KV = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]
"""LFM2-350M's per-layer KV heads: zero on a convolution layer, 8 on an attention layer."""


def _array(items: list[int]) -> dict[str, object]:
    """A per-layer count as the raw header writes it."""
    return {"type": 4, "len": len(items), "array": items, "startOffset": 1, "size": 4}


def _arch_header(
    heads: object, kv_heads: object, *, vocab: int | None = 65_536
) -> dict[str, object]:
    """A raw header of architecture ``hybrid``; a ``None`` field is left out."""
    fields = {
        "hybrid.attention.head_count": heads,
        "hybrid.attention.head_count_kv": kv_heads,
        "tokenizer.ggml.tokens": None if vocab is None else {"type": 8, "len": vocab},
    }
    entries = [{"key": "general.architecture", "value": "hybrid"}] + [
        {"key": key, "value": value} for key, value in fields.items() if value is not None
    ]
    return {"header": {"metadataKV": entries}}


GRANITE_H_KV = [0] * 10 + [4, 0, 0, 4, 0, 0, 0, 4] + [0] * 9 + [4, 0, 0, 0, 0]
"""granite-4.0-h-350m's per-layer KV heads: four attention layers among 32."""

PER_LAYER = {
    "lfm2": (_arch_header(16, _array(LFM2_KV)), 16, 8),
    "granite-h": (_arch_header(12, _array(GRANITE_H_KV)), 12, 4),
    "query-head-array": (_arch_header(_array([32, 16]), 8), 32, 8),
}
"""Per-layer headers with the query and key-value heads of their largest layer."""

UNSCALED = {
    "no-kv-heads": _arch_header(32, None),
    "no-heads": _arch_header(None, None),
    "no-vocabulary": _arch_header(32, 8, vocab=None),
    "text-heads": _arch_header("32", 8),
    "array-without-items": _arch_header(32, {"type": 4, "len": 16}),
    "empty-array": _arch_header(32, _array([])),
    "text-array": _arch_header(32, _array(["8"])),
}
"""Headers the CPU rule has no measurement for: absent or unreadable head counts."""


class TestAHeaderWithoutScalarHeads:
    def _shape(self, parser_for: ParserFor, header: dict[str, object]) -> object:
        path, fake = parser_for("llama-1b-q8")
        fake.header = header
        return vram_mod.attention_shape(path)

    def test_a_scalar_header_has_its_heads_and_vocabulary(self, parser_for: ParserFor) -> None:
        shape = self._shape(parser_for, _arch_header(32, 8))
        assert shape == vram_mod.AttentionShape(32, 8, 65_536)

    @pytest.mark.parametrize("case", sorted(PER_LAYER))
    def test_a_per_layer_header_has_the_shape_of_its_largest_layer(
        self, case: str, parser_for: ParserFor
    ) -> None:
        header, heads, kv_heads = PER_LAYER[case]
        shape = self._shape(parser_for, header)
        assert shape == vram_mod.AttentionShape(heads, kv_heads, 65_536, per_layer=True)

    @pytest.mark.parametrize("case", sorted(UNSCALED))
    def test_it_has_no_shape(self, case: str, parser_for: ParserFor) -> None:
        assert self._shape(parser_for, UNSCALED[case]) is None

    @pytest.mark.parametrize("flash", [True, False])
    @pytest.mark.parametrize("case", sorted(UNSCALED))
    def test_it_keeps_the_parsers_own_figure_without_a_shape(
        self, case: str, flash: bool, parser_for: ParserFor
    ) -> None:
        path, fake = parser_for("llama-1b-q8")
        fake.header = UNSCALED[case]
        request = MemoryRequest(
            model_path=path, ctx=4096, slots=1, gpu_layers=0, flash_attn=flash,
            kv_cache_type=KvCacheType.F16, on_cpu=True,
        )  # fmt: skip
        assert _charge(request) == (
            PARSER[("llama-1b-q8", 1, 4096, flash, "f16")].uma + MODELS["llama-1b-q8"].file_bytes
        )


class TestAPerLayerHeaderOnTheCpu:
    """A hybrid recurrent model pays one layer's flash-off scores and nothing else new."""

    def _request(
        self, path: Path, *, flash: bool, on_cpu: bool = True, slots: int = 1, ctx: int = 4096
    ) -> MemoryRequest:
        return MemoryRequest(
            model_path=path, ctx=ctx, slots=slots, gpu_layers=0 if on_cpu else 99, flash_attn=flash,
            kv_cache_type=KvCacheType.F16, on_cpu=on_cpu,
        )  # fmt: skip

    @pytest.mark.parametrize("case", sorted(PER_LAYER))
    def test_flash_off_adds_the_scores_of_the_largest_layer(
        self, case: str, parser_for: ParserFor
    ) -> None:
        path, fake = parser_for("llama-1b-q8")
        fake.header, heads, kv_heads = PER_LAYER[case]
        base = PARSER[("llama-1b-q8", 1, 4096, False, "f16")].uma + MODELS["llama-1b-q8"].file_bytes
        scores = DEFAULT_UBATCH * 4096 * (4 * heads + 6 * kv_heads)
        assert _charge(self._request(path, flash=False)) == base + scores

    def test_the_lfm2_charge_covers_the_measured_flash_off_rise(
        self, parser_for: ParserFor
    ) -> None:
        """LFM2-350M measured 194 MB more flash off than on at 1 slot and 4096 tokens."""
        path, fake = parser_for("llama-1b-q8")
        fake.header = PER_LAYER["lfm2"][0]
        rise = _charge(self._request(path, flash=False)) - _charge(self._request(path, flash=True))
        assert 194_000_000 <= rise <= 194_000_000 * 1.25

    @pytest.mark.parametrize("case", sorted(PER_LAYER))
    def test_flash_on_keeps_the_parsers_own_figure(self, case: str, parser_for: ParserFor) -> None:
        path, fake = parser_for("llama-1b-q8")
        fake.header = PER_LAYER[case][0]
        assert _charge(self._request(path, flash=True)) == (
            PARSER[("llama-1b-q8", 1, 4096, True, "f16")].uma + MODELS["llama-1b-q8"].file_bytes
        )

    @pytest.mark.parametrize("flash", [True, False])
    @pytest.mark.parametrize("case", sorted(PER_LAYER))
    def test_the_gpu_price_does_not_move(
        self, case: str, flash: bool, parser_for: ParserFor
    ) -> None:
        path, fake = parser_for("llama-1b-q8")
        scalar = LLAMA.estimate_memory(self._request(path, flash=flash, on_cpu=False))
        fake.header = PER_LAYER[case][0]
        assert LLAMA.estimate_memory(self._request(path, flash=flash, on_cpu=False)) == scalar
        assert scalar.device_bytes == PARSER[("llama-1b-q8", 1, 4096, flash, "f16")].vram
