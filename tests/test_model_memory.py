"""The one memory estimator for each model format."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import GgufVramEstimate
from tests._safetensors import safetensors_bytes
from tests.conftest import child_pythonpath

RESERVE = mm.UNKNOWN_SIZE_RESERVE_BYTES


def _large_vocab_charge(weights: int) -> int:
    return weights * 11 // 10 + 3584 * 1024**2


def _request(path: Path, **overrides: object) -> MemoryRequest:
    fields: dict[str, object] = {
        "model_path": path,
        "ctx": 4096,
        "slots": 2,
        "gpu_layers": 7,
        "flash_attn": True,
        "kv_cache_type": KvCacheType.F16,
    }
    fields.update(overrides)
    return MemoryRequest(**fields)  # type: ignore[arg-type]


def _write(folder: Path, name: str, size: int) -> None:
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    whole = name.endswith(".safetensors") and size >= 10
    target.write_bytes(safetensors_bytes(size) if whole else b"\0" * size)


def _tensor(start: int, end: int) -> dict[str, object]:
    return {"dtype": "U8", "shape": [end - start], "data_offsets": [start, end]}


def _safetensors(header: object, data: int) -> bytes:
    """A safetensors file: the length prefix, *header* as JSON, then *data* bytes."""
    raw = json.dumps(header).encode()
    return len(raw).to_bytes(8, "little") + raw + b"\0" * data


def _raw_header(raw: bytes) -> bytes:
    """A safetensors file whose header text is exactly *raw*, with no data."""
    return len(raw).to_bytes(8, "little") + raw


_BRACKET_NOTE = {"note": 'say "hi" [[[{{{ \\" ]]]}}}'}
"""A metadata string full of brackets and quotes: it is text, not nesting."""
_WHOLE_FILE = _safetensors(
    {"__metadata__": _BRACKET_NOTE, "a": _tensor(600, 1000), "b": _tensor(0, 600)}, 1000
)
_CUT_FILES: dict[str, bytes] = {
    "cut-in-the-data": _WHOLE_FILE[:1100],
    "cut-in-the-header": _WHOLE_FILE[:50],
    "shorter-than-a-prefix": b"\0" * 3,
    "under-a-header": b"\0" * 9,
    "extra-bytes": _WHOLE_FILE + b"\0" * 5,
    "header-past-the-file": (10**6).to_bytes(8, "little") + b"{}",
    "header-not-json": (4).to_bytes(8, "little") + b"nope",
    "header-nested-lists": _raw_header(b"[" * 200_000 + b"]" * 200_000),
    "header-nested-objects": _raw_header(b'{"a":' * 100_000 + b"1" + b"}" * 100_000),
    "header-one-level-too-deep": _raw_header(b'{"a":{"b":{"c":[0,0]}}}'),
    "header-not-an-object": (2).to_bytes(8, "little") + b"[]",
    "tensor-not-an-object": _safetensors({"a": 5}, 0),
    "no-offsets": _safetensors({"a": {"dtype": "U8"}}, 0),
    "one-offset": _safetensors({"a": {"data_offsets": [0]}}, 0),
    "offset-not-a-number": _safetensors({"a": {"data_offsets": [0, "x"]}}, 0),
    "offsets-reversed": _safetensors({"a": {"data_offsets": [5, 2]}}, 0),
}


SMALL_KV_BYTES = 32
"""The cache bytes per token of `_write_small_config`: 2 layers, 1 head of width 4, K and V, 2 B."""


def _write_small_config(folder: Path) -> None:
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


class TestWeightsFromHeader:
    def test_sums_each_dtype_at_its_own_width(self) -> None:
        assert mm.weights_from_header({"BF16": 10, "U32": 5, "F8_E4M3": 3}) == 20 + 20 + 3

    def test_unknown_dtype_counts_at_the_widest_width(self) -> None:
        assert mm.weights_from_header({"F4_NEW": 3}) == 24

    def test_an_empty_header_is_unknown(self) -> None:
        assert mm.weights_from_header({}) is None

    @pytest.mark.parametrize(
        "count", [-5, 2.5, "7", None, 2**62, True], ids=lambda value: repr(value)
    )
    def test_a_malformed_count_is_unknown(self, count: object) -> None:
        assert mm.weights_from_header({"BF16": 4, "F32": count}) is None  # type: ignore[dict-item]

    def test_a_whole_number_float_is_a_count(self) -> None:
        assert mm.weights_from_header({"BF16": 6.0}) == 12  # type: ignore[dict-item]

    def test_a_header_of_zero_tensors_is_unknown(self) -> None:
        assert mm.weights_from_header({"BF16": 0}) is None

    def test_narrowing_counts_float32_at_two_bytes_and_leaves_other_dtypes(self) -> None:
        counts = {"F32": 1000, "BF16": 10, "I64": 1}
        assert mm.weights_from_header(counts) == 4000 + 20 + 8
        assert mm.weights_from_header(counts, narrow_f32=True) == 2000 + 20 + 8


GQA = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2560,
}


class TestKvPerToken:
    def test_a_grouped_query_config_prices_a_key_and_a_value_per_layer(self) -> None:
        assert mm.kv_per_token_from_config(GQA) == mm.KvPerToken(147456, upper_bound=False)

    def test_the_head_width_falls_back_to_the_hidden_size_over_the_heads(self) -> None:
        config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        assert mm.kv_per_token_from_config(config) == mm.KvPerToken(131072, upper_bound=False)

    @pytest.mark.parametrize("kv_heads", [None, 0])
    def test_absent_key_value_heads_mean_one_for_each_attention_head(
        self, kv_heads: int | None
    ) -> None:
        config = {**GQA, "num_key_value_heads": kv_heads}
        assert mm.kv_per_token_from_config(config) == mm.KvPerToken(
            2 * 36 * 32 * 128 * 2, upper_bound=False
        )

    def test_a_multimodal_config_is_read_from_its_text_section(self) -> None:
        config = {"text_config": GQA, "vision_config": {"num_hidden_layers": 99}}
        assert mm.kv_per_token_from_config(config) == mm.KvPerToken(147456, upper_bound=False)

    def test_a_text_section_that_is_not_a_mapping_leaves_the_top_level_in_use(self) -> None:
        assert mm.kv_per_token_from_config({**GQA, "text_config": "none"}) == mm.KvPerToken(
            147456, upper_bound=False
        )

    @pytest.mark.parametrize(
        "window",
        [{"sliding_window": 4096}, {"layer_types": ["sliding_attention", "full_attention"]}],
    )
    def test_a_window_marks_the_figure_as_an_upper_bound(self, window: dict) -> None:
        result = mm.kv_per_token_from_config({**GQA, **window})
        assert result == mm.KvPerToken(147456, upper_bound=True)

    @pytest.mark.parametrize(
        "full",
        [{"sliding_window": None}, {"layer_types": ["full_attention"] * 36}, {"layer_types": None}],
    )
    def test_a_window_that_is_absent_or_covers_every_layer_is_not_an_upper_bound(
        self, full: dict
    ) -> None:
        result = mm.kv_per_token_from_config({**GQA, **full})
        assert result == mm.KvPerToken(147456, upper_bound=False)

    def test_a_latent_attention_config_is_unknown(self) -> None:
        config = {**GQA, "kv_lora_rank": 512}
        assert mm.kv_per_token_from_config(config) is None

    @pytest.mark.parametrize(
        "missing", ["num_hidden_layers", "num_attention_heads"], ids=lambda key: f"no {key}"
    )
    def test_a_config_without_layers_or_heads_is_unknown(self, missing: str) -> None:
        config = {key: value for key, value in GQA.items() if key != missing}
        assert mm.kv_per_token_from_config(config) is None

    def test_a_config_with_no_width_is_unknown(self) -> None:
        config = {
            key: value for key, value in GQA.items() if key not in ("head_dim", "hidden_size")
        }
        assert mm.kv_per_token_from_config(config) is None

    def test_a_hidden_size_below_the_head_count_is_unknown(self) -> None:
        config = {**GQA, "head_dim": None, "hidden_size": 16}
        assert mm.kv_per_token_from_config(config) is None

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "36", True, None])
    def test_a_layer_count_that_is_not_a_positive_whole_number_is_unknown(
        self, bad: object
    ) -> None:
        assert mm.kv_per_token_from_config({**GQA, "num_hidden_layers": bad}) is None


class TestChatEstimate:
    KV = mm.KvPerToken(100, upper_bound=False)

    def test_charges_weights_one_cache_per_slot_and_the_runtime_allowance(self) -> None:
        est = mm.safetensors_chat_estimate(1000, self.KV, 50, 3)
        assert est == MemoryEstimate(
            kind=MemoryKind.FIXED_RESERVE,
            device_bytes=1000 + 100 * 50 * 3 + mm.CHAT_RUNTIME_BYTES,
            host_bytes=0,
            shared_bytes=1000 + 100 * 50 * 3 + mm.CHAT_RUNTIME_BYTES,
        )
        assert est.size_known is True

    @pytest.mark.parametrize(("weights", "kv"), [(None, KV), (1000, None), (None, None)])
    def test_an_unknown_weight_or_cache_charges_the_reserve_and_says_so(
        self, weights: int | None, kv: mm.KvPerToken | None
    ) -> None:
        est = mm.safetensors_chat_estimate(weights, kv, 50, 3)
        assert (est.device_bytes, est.shared_bytes, est.size_known) == (RESERVE, RESERVE, False)

    def test_unknown_cache_over_a_model_larger_than_the_reserve_charges_the_weights(self) -> None:
        est = mm.safetensors_chat_estimate(RESERVE + 5, None, 50, 3)
        assert (est.device_bytes, est.size_known) == (RESERVE + 5, False)


class TestRuntimeAllowances:
    def test_a_chat_engine_holds_512_mib_beyond_its_weights_and_cache(self) -> None:
        est = mm.safetensors_chat_estimate(0, mm.KvPerToken(1, upper_bound=False), 1, 1)
        assert est.device_bytes == 1 + 512 * 1024**2

    def test_an_encoder_holds_500_mib_beyond_its_weights(self) -> None:
        assert mm.encoder_estimate(0).device_bytes == 500 * 1024**2


class TestEncoderEstimate:
    def test_charges_weights_and_the_runtime_allowance_with_no_cache(self) -> None:
        est = mm.encoder_estimate(1000)
        assert (est.device_bytes, est.shared_bytes, est.size_known) == (
            1000 + mm.ENCODER_RUNTIME_BYTES,
            1000 + mm.ENCODER_RUNTIME_BYTES,
            True,
        )

    def test_unknown_weights_charge_the_reserve_and_say_so(self) -> None:
        est = mm.encoder_estimate(None)
        assert (est.device_bytes, est.size_known) == (RESERVE, False)


class TestFixedWeightsEstimate:
    def test_a_size_is_charged_whole(self) -> None:
        est = mm.fixed_weights_estimate(1234)
        assert (est.device_bytes, est.shared_bytes, est.size_known) == (1234, 1234, True)

    @pytest.mark.parametrize("weights", [None, 0, -4])
    def test_no_size_is_the_reserve_and_says_so(self, weights: int | None) -> None:
        est = mm.fixed_weights_estimate(weights)
        assert (est.device_bytes, est.size_known) == (RESERVE, False)


class TestAnUnknownSizeChargesHostMemory:
    @pytest.mark.parametrize(
        ("estimate", "charge"),
        [
            (mm.fixed_weights_estimate(None), RESERVE),
            (mm.encoder_estimate(None), RESERVE),
            (mm.safetensors_chat_estimate(None, None, 50, 3), RESERVE),
            (mm.safetensors_chat_estimate(RESERVE + 5, None, 50, 3), RESERVE + 5),
        ],
    )
    def test_the_host_figure_is_the_charge_not_zero(
        self, estimate: MemoryEstimate, charge: int
    ) -> None:
        assert not estimate.size_known
        assert (estimate.host_bytes, estimate.device_bytes) == (charge, charge)


class TestInstalledConfig:
    def test_reads_the_config_beside_a_weight_file(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text('{"a": 1}', encoding="utf-8")
        _write(tmp_path, "model.safetensors", 10)
        assert mm.installed_config(tmp_path / "model.safetensors") == {"a": 1}
        assert mm.installed_config(tmp_path) == {"a": 1}

    @pytest.mark.parametrize("body", [None, "{broken", "[1, 2]"], ids=["absent", "broken", "list"])
    def test_a_missing_or_unusable_config_is_none(self, body: str | None, tmp_path: Path) -> None:
        if body is not None:
            (tmp_path / "config.json").write_text(body, encoding="utf-8")
        assert mm.installed_config(tmp_path) is None


class TestInstalledEncoder:
    @pytest.mark.parametrize("key", ["torch_dtype", "dtype"])
    def test_a_float32_config_halves_the_weights_when_narrowing(
        self, key: str, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        (tmp_path / "config.json").write_text(f'{{"{key}": "float32"}}', encoding="utf-8")
        narrow = mm.estimate_installed_encoder(tmp_path, narrow_f32=True)
        plain = mm.estimate_installed_encoder(tmp_path, narrow_f32=False)
        assert narrow.device_bytes == 2000 + mm.ENCODER_RUNTIME_BYTES
        assert plain.device_bytes == _large_vocab_charge(4000)

    def test_a_float32_text_section_halves_the_weights_when_narrowing(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        (tmp_path / "config.json").write_text(
            '{"text_config": {"torch_dtype": "float32"}}', encoding="utf-8"
        )
        assert mm.estimate_installed_encoder(tmp_path, narrow_f32=True).device_bytes == (
            2000 + mm.ENCODER_RUNTIME_BYTES
        )

    @pytest.mark.parametrize("config", ['{"torch_dtype": "bfloat16"}', "{}", None])
    def test_a_config_that_is_not_float32_leaves_the_weights_whole(
        self, config: str | None, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        if config is not None:
            (tmp_path / "config.json").write_text(config, encoding="utf-8")
        assert mm.estimate_installed_encoder(tmp_path, narrow_f32=True).device_bytes == (
            4000 + mm.ENCODER_RUNTIME_BYTES
        )

    @pytest.mark.parametrize(
        ("vocab", "charge"),
        [
            (30_522, 4000 * 3 // 2 + 128 * 1024**2),
            (99_999, 4000 * 3 // 2 + 128 * 1024**2),
            (100_000, 4000 * 11 // 10 + 3584 * 1024**2),
            (250_002, 4000 * 11 // 10 + 3584 * 1024**2),
            (None, 4000 * 11 // 10 + 3584 * 1024**2),
        ],
    )
    def test_a_cpu_build_charges_by_the_vocabulary(
        self, vocab: int | None, charge: int, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        config = {} if vocab is None else {"vocab_size": vocab}
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        assert mm.estimate_installed_encoder(tmp_path, narrow_f32=False).device_bytes == charge

    @pytest.mark.parametrize("vocab", ['"30522"', "true", "[30522]"])
    def test_a_vocabulary_that_is_not_a_number_is_charged_as_a_large_one(
        self, vocab: str, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        (tmp_path / "config.json").write_text(f'{{"vocab_size": {vocab}}}', encoding="utf-8")
        assert mm.estimate_installed_encoder(tmp_path, narrow_f32=False).device_bytes == (
            _large_vocab_charge(4000)
        )

    def test_a_cuda_build_ignores_the_vocabulary(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        (tmp_path / "config.json").write_text('{"vocab_size": 250002}', encoding="utf-8")
        assert mm.estimate_installed_encoder(tmp_path, narrow_f32=True).device_bytes == (
            4000 + mm.ENCODER_RUNTIME_BYTES
        )

    def test_a_folder_with_no_weights_is_unknown(self, tmp_path: Path) -> None:
        est = mm.estimate_installed_encoder(tmp_path, narrow_f32=True)
        assert (est.device_bytes, est.size_known) == (RESERVE, False)

    @pytest.mark.parametrize("shape", sorted(_CUT_FILES))
    def test_a_weight_file_that_is_not_whole_is_unknown(self, shape: str, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(_CUT_FILES[shape])
        est = mm.estimate_installed_encoder(tmp_path, narrow_f32=False)
        assert (est.device_bytes, est.size_known) == (RESERVE, False)

    @pytest.mark.parametrize("size", [10, 11, 135, 136, 4000])
    def test_a_whole_weight_file_is_known_at_its_size(self, size: int, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", size)
        est = mm.estimate_installed_encoder(tmp_path, narrow_f32=True)
        assert (est.device_bytes, est.size_known) == (size + mm.ENCODER_RUNTIME_BYTES, True)

    def test_tensors_in_any_order_and_a_metadata_entry_make_a_whole_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "model.safetensors").write_bytes(_WHOLE_FILE)
        est = mm.estimate_installed_encoder(tmp_path, narrow_f32=True)
        assert (est.device_bytes, est.size_known) == (
            len(_WHOLE_FILE) + mm.ENCODER_RUNTIME_BYTES,
            True,
        )

    def test_one_cut_off_shard_makes_the_whole_folder_unknown(self, tmp_path: Path) -> None:
        _write(tmp_path, "model-00001-of-00002.safetensors", 4000)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(_CUT_FILES["cut-in-the-data"])
        est = mm.estimate_installed_encoder(tmp_path, narrow_f32=False)
        assert (est.device_bytes, est.size_known) == (RESERVE, False)

    def test_a_single_cut_off_file_is_unknown(self, tmp_path: Path) -> None:
        target = tmp_path / "model.safetensors"
        target.write_bytes(_CUT_FILES["cut-in-the-data"])
        assert mm.whole_safetensors_bytes(target) == 0

    def test_a_weight_file_that_cannot_be_opened_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)

        real_open = Path.open

        def refuse(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self.suffix == ".safetensors":
                raise PermissionError("no access")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse)
        assert mm.whole_safetensors_bytes(tmp_path) == 0

    def test_a_header_past_the_size_cap_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, "model.safetensors", 4000)
        monkeypatch.setattr(mm, "_MAX_SAFETENSORS_HEADER_BYTES", 100)
        assert mm.whole_safetensors_bytes(tmp_path) == 0

    def test_the_on_disk_sum_still_counts_a_cut_off_file(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(_CUT_FILES["cut-in-the-data"])
        assert mm.installed_safetensors_bytes(tmp_path) == 1100

    def test_a_cut_folder_over_the_reserve_is_charged_its_bytes_on_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_disk = RESERVE * 2
        (tmp_path / "model.safetensors").write_bytes(_CUT_FILES["cut-in-the-data"])
        monkeypatch.setattr(mm, "_installed_sizes", lambda _path: {"model.safetensors": on_disk})
        _write_small_config(tmp_path)
        encoder = mm.estimate_installed_encoder(tmp_path, narrow_f32=False)
        chat = mm.estimate_memory(ModelFormat.SAFETENSORS, _request(tmp_path))
        assert [(e.device_bytes, e.size_known) for e in (encoder, chat)] == [(on_disk, False)] * 2

    def test_a_header_of_escaped_quotes_is_read_in_linear_time(self, tmp_path: Path) -> None:
        raw_header = b'{"a":"' + b'\\"' * 524_288 + b'"}'
        blob = len(raw_header).to_bytes(8, "little") + raw_header
        (tmp_path / "model.safetensors").write_bytes(blob)
        start = time.monotonic()
        assert mm.whole_safetensors_bytes(tmp_path) == 0
        assert time.monotonic() - start < 5


class TestWeightsFromFiles:
    def test_counts_only_the_ir_model_files(self) -> None:
        sizes = {
            "openvino_model.xml": 10,
            "openvino_model.bin": 1000,
            "openvino_tokenizer.bin": 500,
            "pytorch_model.bin": 9000,
            "model.onnx": 9000,
        }
        assert mm.weights_from_files(sizes) == 1010

    def test_quantized_variant_is_not_counted_twice(self) -> None:
        sizes = {
            "openvino_model.xml": 10,
            "openvino_model.bin": 1000,
            "openvino_model_qint8_quantized.xml": 10,
            "openvino_model_qint8_quantized.bin": 400,
        }
        assert mm.weights_from_files(sizes) == 1010

    def test_uses_the_shallowest_folder_that_holds_the_ir(self) -> None:
        sizes = {
            "openvino/openvino_model.xml": 1,
            "openvino/openvino_model.bin": 100,
            "a/b/openvino_model.xml": 1,
            "a/b/openvino_model.bin": 5000,
        }
        assert mm.weights_from_files(sizes) == 101

    def test_a_vision_model_counts_every_pair_but_the_tokenizers(self) -> None:
        sizes = {
            "openvino_language_model.xml": 10,
            "openvino_language_model.bin": 1000,
            "openvino_vision_embeddings_model.xml": 10,
            "openvino_vision_embeddings_model.bin": 300,
            "openvino_text_embeddings_model.xml": 10,
            "openvino_text_embeddings_model.bin": 200,
            "openvino_tokenizer.xml": 10,
            "openvino_tokenizer.bin": 500,
            "openvino_detokenizer.xml": 10,
            "openvino_detokenizer.bin": 500,
        }
        assert mm.weights_from_files(sizes) == 1530

    def test_a_vision_part_without_its_bin_is_not_counted(self) -> None:
        sizes = {"openvino_language_model.xml": 10, "openvino_vision_model.bin": 70}
        assert mm.weights_from_files(sizes) == RESERVE

    def test_a_vision_model_uses_the_shallowest_folder_that_holds_a_pair(self) -> None:
        sizes = {
            "openvino_language_model.xml": 1,
            "openvino_language_model.bin": 100,
            "a/b/openvino_vision_model.xml": 1,
            "a/b/openvino_vision_model.bin": 5000,
        }
        assert mm.weights_from_files(sizes) == 101

    def test_a_listing_with_no_ir_is_the_reserve(self) -> None:
        assert mm.weights_from_files({"pytorch_model.bin": 9000}) == RESERVE

    def test_an_empty_listing_is_the_reserve(self) -> None:
        assert mm.weights_from_files({}) == RESERVE

    def test_a_repo_with_only_a_variant_counts_its_bin_files(self) -> None:
        sizes = {
            "openvino_model_qint8_quantized.xml": 10,
            "openvino_model_qint8_quantized.bin": 400,
            "pytorch_model.bin": 9000,
        }
        assert mm.weights_from_files(sizes) == 400

    def test_an_xml_without_its_bin_falls_back_to_the_variant_bins(self) -> None:
        sizes = {"openvino_model.xml": 10, "openvino_model_fp16.bin": 70}
        assert mm.weights_from_files(sizes) == 70

    def test_zero_byte_ir_files_are_the_reserve(self) -> None:
        sizes = {"openvino_model.xml": 0, "openvino_model.bin": 0}
        assert mm.weights_from_files(sizes) == RESERVE

    def test_a_vision_ir_is_the_sum_of_its_part_files(self) -> None:
        parts = ("language", "text_embeddings", "vision_embeddings")
        sizes = {
            f"openvino_{p}_model.{e}": 1000 * (i + 1)
            for i, p in enumerate(parts)
            for e in ("xml", "bin")
        }
        sizes |= {
            "openvino_tokenizer.bin": 500,
            "openvino_tokenizer.xml": 5,
            "pytorch_model.bin": 9000,
        }
        assert mm.weights_from_files(sizes) == 2 * (1000 + 2000 + 3000)

    def test_a_vision_ir_part_without_its_bin_is_not_counted_as_a_pair(self) -> None:
        sizes = {
            "openvino_language_model.xml": 10,
            "openvino_language_model.bin": 1000,
            "openvino_vision_embeddings_model.xml": 10,
        }
        assert mm.weights_from_files(sizes) == RESERVE

    def test_a_vision_ir_counts_only_the_shallowest_folder(self) -> None:
        sizes = {
            "ov/openvino_language_model.xml": 1,
            "ov/openvino_language_model.bin": 100,
            "a/b/openvino_language_model.xml": 1,
            "a/b/openvino_language_model.bin": 5000,
        }
        assert mm.weights_from_files(sizes) == 101


class TestWeightsFromGgufReport:
    def test_reads_the_file_size(self) -> None:
        assert mm.weights_from_gguf_report({"metadata": {"fileSize": 639}}) == 639

    @pytest.mark.parametrize(
        "report",
        [
            {},
            {"metadata": None},
            {"metadata": {}},
            {"metadata": []},
            {"metadata": {"fileSize": None}},
            {"metadata": {"fileSize": 0}},
            {"metadata": {"fileSize": "abc"}},
            {"metadata": {"fileSize": "12"}},
            {"metadata": {"fileSize": -4}},
            {"metadata": {"fileSize": float("nan")}},
            {"metadata": {"fileSize": float("inf")}},
        ],
        ids=repr,
    )
    def test_a_report_without_a_usable_size_is_the_reserve(self, report: dict) -> None:
        assert mm.weights_from_gguf_report(report) == RESERVE


class TestInstalledEstimate:
    @pytest.mark.parametrize(
        "fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS, ModelFormat.OPENVINO]
    )
    def test_a_folder_that_cannot_be_read_is_the_reserve(
        self, fmt: ModelFormat, monkeypatch, tmp_path: Path
    ) -> None:
        real_rglob = Path.rglob

        def _fail(self: Path, pattern: str) -> object:
            if self == tmp_path:
                raise PermissionError("denied")
            return real_rglob(self, pattern)

        monkeypatch.setattr(Path, "rglob", _fail)
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert (est.device_bytes, est.shared_bytes) == (RESERVE, RESERVE)

    @pytest.mark.parametrize("fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS])
    def test_safetensors_formats_sum_the_weight_files_of_the_folder(
        self, fmt: ModelFormat, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model-00001-of-00002.safetensors", 300)
        _write(tmp_path, "model-00002-of-00002.safetensors", 200)
        _write(tmp_path, "tokenizer.json", 9999)
        _write_small_config(tmp_path)
        charge = 500 + SMALL_KV_BYTES * 4096 * 2 + mm.CHAT_RUNTIME_BYTES
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert est == MemoryEstimate(
            kind=MemoryKind.FIXED_RESERVE, device_bytes=charge, host_bytes=0, shared_bytes=charge
        )

    def test_a_single_weight_file_is_sized_by_its_own_size_and_the_config_beside_it(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 123)
        _write_small_config(tmp_path)
        est = mm.estimate_memory(ModelFormat.SAFETENSORS, _request(tmp_path / "model.safetensors"))
        assert est.shared_bytes == 123 + SMALL_KV_BYTES * 4096 * 2 + mm.CHAT_RUNTIME_BYTES

    @pytest.mark.parametrize("fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS])
    def test_a_chat_weight_file_cut_short_is_the_unknown_size_reserve(
        self, fmt: ModelFormat, tmp_path: Path
    ) -> None:
        (tmp_path / "model.safetensors").write_bytes(_CUT_FILES["cut-in-the-data"])
        _write_small_config(tmp_path)
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert (est.device_bytes, est.shared_bytes, est.size_known) == (RESERVE, RESERVE, False)

    @pytest.mark.parametrize("fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS])
    def test_the_cache_grows_with_the_context_and_the_slots(
        self, fmt: ModelFormat, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 1000)
        _write_small_config(tmp_path)
        base = mm.estimate_memory(fmt, _request(tmp_path, ctx=1000, slots=1)).device_bytes
        longer = mm.estimate_memory(fmt, _request(tmp_path, ctx=3000, slots=1)).device_bytes
        wider = mm.estimate_memory(fmt, _request(tmp_path, ctx=1000, slots=3)).device_bytes
        assert (longer - base, wider - base) == (SMALL_KV_BYTES * 2000, SMALL_KV_BYTES * 2000)

    @pytest.mark.parametrize("fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS])
    def test_the_cache_type_setting_changes_nothing(self, fmt: ModelFormat, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", 1000)
        _write_small_config(tmp_path)
        figures = {
            mm.estimate_memory(fmt, _request(tmp_path, kv_cache_type=kind)).device_bytes
            for kind in KvCacheType
        }
        assert len(KvCacheType) > 1
        assert len(figures) == 1

    @pytest.mark.parametrize("fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS])
    @pytest.mark.parametrize(
        "config",
        [None, "{}", '{"num_hidden_layers": 2, "num_attention_heads": 2, "kv_lora_rank": 8}'],
    )
    def test_a_folder_whose_cache_cannot_be_priced_is_the_unknown_size_reserve(
        self, fmt: ModelFormat, config: str | None, tmp_path: Path
    ) -> None:
        _write(tmp_path, "model.safetensors", 1000)
        if config is not None:
            (tmp_path / "config.json").write_text(config, encoding="utf-8")
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert (est.device_bytes, est.shared_bytes, est.size_known) == (RESERVE, RESERVE, False)

    def test_a_measured_folder_says_its_size_is_known(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", 1000)
        _write_small_config(tmp_path)
        assert mm.estimate_memory(ModelFormat.SAFETENSORS, _request(tmp_path)).size_known is True

    def test_openvino_counts_the_ir_files_of_the_folder(self, tmp_path: Path) -> None:
        _write(tmp_path, "openvino_model.xml", 10)
        _write(tmp_path, "openvino_model.bin", 400)
        _write(tmp_path, "pytorch_model.bin", 9000)
        _write_small_config(tmp_path)
        est = mm.estimate_memory(ModelFormat.OPENVINO, _request(tmp_path))
        charge = 2 * 410 + SMALL_KV_BYTES * 4096 * 2 + 928 * 1024**2
        assert (est.kind, est.device_bytes, est.shared_bytes) == (
            MemoryKind.FIXED_RESERVE,
            charge,
            charge,
        )

    @pytest.mark.parametrize(
        "fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS, ModelFormat.OPENVINO]
    )
    def test_a_missing_model_is_the_reserve_and_does_not_raise(
        self, fmt: ModelFormat, tmp_path: Path
    ) -> None:
        est = mm.estimate_memory(fmt, _request(tmp_path / "absent"))
        assert (est.kind, est.device_bytes, est.shared_bytes) == (
            MemoryKind.FIXED_RESERVE,
            RESERVE,
            RESERVE,
        )

    @pytest.mark.parametrize(
        "fmt", [ModelFormat.MLX, ModelFormat.SAFETENSORS, ModelFormat.OPENVINO]
    )
    @pytest.mark.parametrize("shape", ["empty", "only-gguf", "zero-byte", "dangling-link"])
    def test_a_folder_with_no_known_weights_is_the_reserve(
        self, fmt: ModelFormat, shape: str, tmp_path: Path
    ) -> None:
        if shape == "only-gguf":
            _write(tmp_path, "model.gguf", 5000)
        elif shape == "zero-byte":
            _write(tmp_path, "model.safetensors", 0)
            _write(tmp_path, "openvino_model.xml", 0)
            _write(tmp_path, "openvino_model.bin", 0)
        elif shape == "dangling-link":
            (tmp_path / "model.safetensors").symlink_to(tmp_path / "nowhere")
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert (est.device_bytes, est.shared_bytes) == (RESERVE, RESERVE)

    def test_an_ir_folder_with_only_a_variant_counts_the_variant_bin(self, tmp_path: Path) -> None:
        _write(tmp_path, "openvino_model_qint8_quantized.xml", 10)
        _write(tmp_path, "openvino_model_qint8_quantized.bin", 400)
        _write_small_config(tmp_path)
        est = mm.estimate_memory(ModelFormat.OPENVINO, _request(tmp_path))
        assert est.shared_bytes == 2 * 400 + SMALL_KV_BYTES * 4096 * 2 + 928 * 1024**2

    def test_the_reserve_is_eight_gibibytes(self) -> None:
        assert RESERVE == 8 * 1024**3


def _footprint(**fields: object) -> GgufVramEstimate:
    return GgufVramEstimate(**fields)  # type: ignore[arg-type]


class TestGgufEstimate:
    @pytest.mark.parametrize(
        ("parsed", "extra"),
        [
            (_footprint(vram_bytes=1, ram_bytes=2, unified_bytes=3), {}),
            (
                _footprint(
                    vram_bytes=10,
                    ram_bytes=20,
                    unified_bytes=30,
                    per_device_vram=(4, 6),
                    per_device_unified=(14, 16),
                ),
                {"tensor_split": (1, 1), "batch_size": 512, "expert_offload": ("exps",)},
            ),
        ],
    )
    def test_equals_the_parsers_figures_and_passes_every_request_field(
        self, monkeypatch, tmp_path: Path, parsed: GgufVramEstimate, extra: dict
    ) -> None:
        seen: dict[str, object] = {}

        def _est(model_path: Path, **kwargs: object) -> GgufVramEstimate:
            seen.update(model_path=model_path, **kwargs)
            return parsed

        monkeypatch.setattr(mm, "estimate_instance_footprint", _est)
        mmproj = tmp_path / "proj.gguf"
        est = LLAMA.estimate_memory(_request(tmp_path / "m.gguf", mmproj_path=mmproj, **extra))
        assert est == MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=parsed.vram_bytes,
            host_bytes=parsed.ram_bytes,
            shared_bytes=parsed.unified_bytes,
            per_device=parsed.per_device_vram,
            per_device_shared=parsed.per_device_unified,
        )
        assert seen == {
            "model_path": tmp_path / "m.gguf",
            "ctx": 4096,
            "slots": 2,
            "gpu_layers": 7,
            "flash_attn": True,
            "kv_cache_type": KvCacheType.F16,
            "mmproj_path": mmproj,
            "tensor_split": extra.get("tensor_split", ()),
            "batch_size": extra.get("batch_size"),
            "expert_offload": extra.get("expert_offload", ()),
        }

    def test_the_llama_adapter_asks_the_table_for_the_gguf_estimator(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        sentinel = MemoryEstimate(MemoryKind.SIZED, 9, 8, 7)
        monkeypatch.setitem(mm.FORMAT_ESTIMATORS, ModelFormat.GGUF, lambda request: sentinel)
        assert LLAMA.estimate_memory(_request(tmp_path / "m.gguf")) is sentinel


class TestTable:
    def test_every_format_has_one_estimator(self) -> None:
        assert set(mm.FORMAT_ESTIMATORS) == set(ModelFormat)
        assert len(mm.FORMAT_ESTIMATORS) > 0

    def test_every_format_an_adapter_names_has_an_estimator(self) -> None:
        assert LLAMA.formats
        assert LLAMA.formats <= set(mm.FORMAT_ESTIMATORS)


_WINDOWS_CHILD_VARIABLES = frozenset(
    {"SYSTEMROOT", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA"}
)


class TestColdImport:
    @pytest.mark.parametrize(
        "module",
        ["lilbee.providers.fleet.model_memory", "lilbee.providers.fleet.engines"],
    )
    def test_a_fresh_interpreter_imports_the_module(self, module: str) -> None:
        done = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=120,
            env={
                "PYTHONPATH": child_pythonpath(Path(mm.__file__).parents[3]),
                # A Windows child needs its system root for the network stack, and its
                # profile variables for the home folder that the data directory names.
                **{k: v for k, v in os.environ.items() if k in _WINDOWS_CHILD_VARIABLES},
            },
        )
        assert done.returncode == 0, done.stderr
