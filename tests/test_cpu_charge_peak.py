"""A CPU llama.cpp load is charged its measured peak resident size, within a tenth above it.

The figures are recorded with the llama.cpp CPU pack 1.12032.1 on a 32-thread AVX2 host, each
cell run twice (the higher peak is kept): Llama-3.2-1B Q8_0 at 4096 tokens and one slot, and
Qwen2.5-7B-Instruct Q4_K_M at 8192 tokens per slot and two busy slots. The peak is the highest
resident size of the server after the slots took prompts of the context length minus 32 tokens.
The parser figures are its answers for the same launches, at the default ubatch and at one row
per slot, with flash attention on and off.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import model_memory as model_memory_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.openvino import OPENVINO
from lilbee.providers.fleet.memory_types import MemoryRequest

DEFAULT_UBATCH = 512


@dataclass(frozen=True)
class Cell:
    slots: int
    ctx: int
    file_bytes: int
    repacked_bytes: int
    ram_uma_by_ubatch: dict[int, int]
    device_flash_off: int
    device_flash_on: int


CELLS = {
    "llama-1b-q8": Cell(
        slots=1,
        ctx=4096,
        file_bytes=1_321_083_008,
        repacked_bytes=0,
        ram_uma_by_ubatch={DEFAULT_UBATCH: 739_923_224, 1: 202_543_396},
        device_flash_off=564_135_936,
        device_flash_on=396_625_920,
    ),
    "qwen-7b-q4km": Cell(
        slots=2,
        ctx=8192,
        file_bytes=4_683_074_240,
        repacked_bytes=3_427_909_632,
        ram_uma_by_ubatch={DEFAULT_UBATCH: 1_665_226_008, 2: 1_021_750_576},
        device_flash_off=1_298_139_136,
        device_flash_on=591_284_224,
    ),
}
PEAKS = {
    ("llama-1b-q8", "auto"): 1_605_230_592,
    ("qwen-7b-q4km", "off"): 9_551_196_160,
    ("qwen-7b-q4km", "auto"): 8_986_681_344,
}


class _Parser:
    """A stand-in for the estimator binary that answers by the flags it is given."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.argvs: list[list[str]] = []

    def __call__(self, argv: list[str], _path: str) -> str:
        self.argvs.append(argv)
        ubatch = int(argv[argv.index("--ubatch-size") + 1]) if "--ubatch-size" in argv else None
        uma = self.cell.ram_uma_by_ubatch[ubatch or DEFAULT_UBATCH]
        device = (
            self.cell.device_flash_off
            if "--no-flash-attention" in argv
            else self.cell.device_flash_on
        )
        return json.dumps(
            {
                "metadata": {"fileSize": self.cell.file_bytes},
                "estimate": {
                    "items": [
                        {
                            "ram": {"uma": uma, "nonuma": uma},
                            "vrams": [{"uma": 0, "nonuma": device}],
                        }
                    ]
                },
            }
        )


ParserFor = Callable[[str], tuple[Path, _Parser]]


@pytest.fixture
def parser_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParserFor:
    def install(name: str) -> tuple[Path, _Parser]:
        cell = CELLS[name]
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(b"GGUF")
        vram_mod._cached_footprint.cache_clear()
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        monkeypatch.setattr(
            model_memory_mod, "repacked_weight_bytes", lambda _p: cell.repacked_bytes
        )
        fake = _Parser(cell)
        monkeypatch.setattr(vram_mod, "_run_parser", fake)
        return path, fake

    return install


def _request(
    path: Path, cell: Cell, *, flash: bool, on_cpu: bool = True, batch_size: int | None = None
) -> MemoryRequest:
    return MemoryRequest(
        model_path=path,
        ctx=cell.ctx,
        slots=cell.slots,
        gpu_layers=0,
        flash_attn=flash,
        kv_cache_type=KvCacheType.F16,
        batch_size=batch_size,
        on_cpu=on_cpu,
    )


def _charge(request: MemoryRequest) -> int:
    return LLAMA.estimate_memory(request).resident_footprint(unified=True)


def _ubatches(fake: _Parser) -> list[int | None]:
    sized = [argv for argv in fake.argvs if "--ctx-size" in argv]
    return [int(a[a.index("--ubatch-size") + 1]) if "--ubatch-size" in a else None for a in sized]


class TestTheChargeAgainstThePeak:
    @pytest.mark.parametrize(("name", "flash"), sorted(PEAKS))
    def test_it_covers_the_peak_and_stays_within_a_tenth_above(
        self, name: str, flash: str, parser_for: ParserFor
    ) -> None:
        peak = PEAKS[(name, flash)]
        path, _ = parser_for(name)
        charge = _charge(_request(path, CELLS[name], flash=False))
        assert peak <= charge <= peak * 1.1

    def test_a_card_charge_is_the_parsers_own_figure_and_file(self, parser_for: ParserFor) -> None:
        cell = CELLS["qwen-7b-q4km"]
        path, fake = parser_for("qwen-7b-q4km")
        charge = _charge(_request(path, cell, flash=False, on_cpu=False))
        assert charge == cell.ram_uma_by_ubatch[DEFAULT_UBATCH] + cell.file_bytes
        assert _ubatches(fake) == [None]


class TestWhatTheCpuChargeAsksFor:
    def test_the_logits_are_one_row_per_slot(self, parser_for: ParserFor) -> None:
        path, fake = parser_for("qwen-7b-q4km")
        _charge(_request(path, CELLS["qwen-7b-q4km"], flash=True))
        assert _ubatches(fake) == [2]

    def test_the_scores_are_priced_at_the_real_ubatch_when_flash_is_off(
        self, parser_for: ParserFor
    ) -> None:
        cell = CELLS["qwen-7b-q4km"]
        path, fake = parser_for("qwen-7b-q4km")
        flash_off = _charge(_request(path, cell, flash=False))
        assert Counter(_ubatches(fake)) == {2: 1, None: 2}
        flash_on = _charge(_request(path, cell, flash=True))
        assert flash_off - flash_on == cell.device_flash_off - cell.device_flash_on

    def test_the_openvino_engine_keeps_the_estimators_own_figure(
        self, parser_for: ParserFor
    ) -> None:
        cell = CELLS["llama-1b-q8"]
        path, fake = parser_for("llama-1b-q8")
        charge = OPENVINO.estimate_memory(_request(path, cell, flash=False)).resident_footprint(
            unified=True
        )
        assert charge == cell.ram_uma_by_ubatch[DEFAULT_UBATCH] + cell.file_bytes
        assert _ubatches(fake) == [None]

    def test_a_pooled_launch_keeps_its_own_batch(self, parser_for: ParserFor) -> None:
        cell = CELLS["llama-1b-q8"]
        path, fake = parser_for("llama-1b-q8")
        charge = _charge(_request(path, cell, flash=False, batch_size=1))
        assert _ubatches(fake) == [1]
        assert charge == cell.ram_uma_by_ubatch[1] + cell.file_bytes
