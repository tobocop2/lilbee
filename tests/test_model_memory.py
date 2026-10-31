"""The one memory estimator for each model format."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import GgufVramEstimate

RESERVE = mm.UNKNOWN_SIZE_RESERVE_BYTES


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
    target.write_bytes(b"\0" * size)


class TestWeightsFromHeader:
    def test_sums_each_dtype_at_its_own_width(self) -> None:
        assert mm.weights_from_header({"BF16": 10, "U32": 5, "F8_E4M3": 3}) == 20 + 20 + 3

    def test_unknown_dtype_counts_at_the_widest_width(self) -> None:
        assert mm.weights_from_header({"F4_NEW": 3}) == 24

    def test_empty_header_is_the_reserve(self) -> None:
        assert mm.weights_from_header({}) == RESERVE

    @pytest.mark.parametrize(
        "count", [-5, 2.5, "7", None, 2**62, True], ids=lambda value: repr(value)
    )
    def test_a_malformed_count_is_the_reserve(self, count: object) -> None:
        assert mm.weights_from_header({"BF16": 4, "F32": count}) == RESERVE  # type: ignore[dict-item]

    def test_a_whole_number_float_is_a_count(self) -> None:
        assert mm.weights_from_header({"BF16": 6.0}) == 12  # type: ignore[dict-item]

    def test_a_header_of_zero_tensors_is_the_reserve(self) -> None:
        assert mm.weights_from_header({"BF16": 0}) == RESERVE


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
        def _fail(self: Path, pattern: str) -> None:
            raise PermissionError("denied")

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
        est = mm.estimate_memory(fmt, _request(tmp_path))
        assert est == MemoryEstimate(
            kind=MemoryKind.FIXED_RESERVE, device_bytes=500, host_bytes=0, shared_bytes=500
        )

    def test_a_single_weight_file_is_sized_by_its_own_size(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.safetensors", 123)
        est = mm.estimate_memory(ModelFormat.SAFETENSORS, _request(tmp_path / "model.safetensors"))
        assert est.shared_bytes == 123

    def test_openvino_counts_the_ir_files_of_the_folder(self, tmp_path: Path) -> None:
        _write(tmp_path, "openvino_model.xml", 10)
        _write(tmp_path, "openvino_model.bin", 400)
        _write(tmp_path, "pytorch_model.bin", 9000)
        est = mm.estimate_memory(ModelFormat.OPENVINO, _request(tmp_path))
        assert (est.kind, est.device_bytes, est.shared_bytes) == (
            MemoryKind.FIXED_RESERVE,
            410,
            410,
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
        est = mm.estimate_memory(ModelFormat.OPENVINO, _request(tmp_path))
        assert est.shared_bytes == 400

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
            env={"PYTHONPATH": str(Path(mm.__file__).parents[3])},
        )
        assert done.returncode == 0, done.stderr
