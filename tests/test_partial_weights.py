"""A cut-off GGUF, a cut-off OpenVINO weights file and a split GGUF missing a shard: no size."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

from lilbee.catalog import header_probe
from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import AttentionShape, GgufVramEstimate
from lilbee.providers.roles import WorkerRole
from tests._gguf_fixture import GGUF_MAGIC, GGUF_VERSION, cut_file_of

_F32 = 0
_UINT32 = 4
_TENSOR_FLOATS = 4
_TENSOR_BYTES = _TENSOR_FLOATS * 4
_PARSED = GgufVramEstimate(vram_bytes=11, ram_bytes=22, unified_bytes=33)


def _string(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _kv_u32(key: bytes, value: int) -> bytes:
    return _string(key) + struct.pack("<II", _UINT32, value)


def _gguf(
    *,
    tensors: int = 2,
    alignment: int | None = None,
    version: int = GGUF_VERSION,
    tensor_type: int = _F32,
    extra_kv: bytes = b"",
    extra_kv_count: int = 0,
) -> bytes:
    """A complete GGUF file: header, key-value table, tensor table, aligned tensor data."""
    kv = extra_kv
    count = extra_kv_count
    if alignment is not None:
        kv += _kv_u32(b"general.alignment", alignment)
        count += 1
    infos = b"".join(
        _string(f"t{i}".encode())
        + struct.pack("<I", 1)
        + struct.pack("<Q", _TENSOR_FLOATS)
        + struct.pack("<I", tensor_type)
        + struct.pack("<Q", i * _TENSOR_BYTES)
        for i in range(tensors)
    )
    head = GGUF_MAGIC + struct.pack("<IQQ", version, tensors, count) + kv + infos
    step = alignment or 32
    padded = -(-len(head) // step) * step if tensors else len(head)
    return head + b"\0" * (padded - len(head)) + b"\0" * (tensors * _TENSOR_BYTES)


def _write(path: Path, blob: bytes) -> Path:
    path.write_bytes(blob)
    return path


def _request(path: Path) -> MemoryRequest:
    return MemoryRequest(
        model_path=path,
        ctx=4096,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


@pytest.fixture
def parser_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The paths the GGUF parser is asked about; it answers with a fixed figure."""
    calls: list[Path] = []

    def _parse(model_path: Path, **_kwargs: object) -> GgufVramEstimate:
        calls.append(model_path)
        return _PARSED

    monkeypatch.setattr(mm, "estimate_instance_footprint", _parse)
    return calls


def _is_unknown(est: MemoryEstimate) -> bool:
    return (est.kind, est.size_known, est.device_bytes) == (
        MemoryKind.FIXED_RESERVE,
        False,
        mm.UNKNOWN_SIZE_RESERVE_BYTES,
    )


class TestAWholeGgufIsSized:
    @pytest.mark.parametrize(
        "blob",
        [
            pytest.param(_gguf(), id="two-tensors"),
            pytest.param(_gguf(tensors=0), id="no-tensors"),
            pytest.param(_gguf(alignment=256), id="aligned-256"),
            pytest.param(
                _gguf(
                    extra_kv=_string(b"tokens")
                    + struct.pack("<IIQ", 9, 8, 2)
                    + _string(b"ab")
                    + _string(b"c"),
                    extra_kv_count=1,
                ),
                id="string-array",
            ),
        ],
    )
    def test_the_parser_sizes_it(
        self, tmp_path: Path, parser_calls: list[Path], blob: bytes
    ) -> None:
        model = _write(tmp_path / "m.gguf", blob)
        est = mm.estimate_gguf(_request(model))
        assert (est.kind, est.device_bytes, parser_calls) == (MemoryKind.SIZED, 11, [model])

    def test_a_path_that_is_not_there_goes_to_the_parser(
        self, tmp_path: Path, parser_calls: list[Path]
    ) -> None:
        mm.estimate_gguf(_request(tmp_path / "gone.gguf"))
        assert parser_calls == [tmp_path / "gone.gguf"]


# Rows that make each tensor a multiple of 64 bytes, so the last tensor leaves no padding.
_ROWS_BY_TYPE = {
    GGMLQuantizationType.F32: 32,
    GGMLQuantizationType.Q4_0: 32,
    GGMLQuantizationType.Q4_K: 4,
    GGMLQuantizationType.Q8_0: 32,
}


def _written_gguf(
    path: Path, types: list[GGMLQuantizationType], alignment: int | None = None
) -> Path:
    """A GGUF file written by the gguf package, one tensor of each type in *types*."""
    writer = GGUFWriter(path, "llama")
    if alignment is not None:
        writer.add_custom_alignment(alignment)
    for index, tensor_type in enumerate(types):
        rows = _ROWS_BY_TYPE[tensor_type]
        if tensor_type is GGMLQuantizationType.F32:
            writer.add_tensor(f"t{index}", np.zeros((rows, 8), dtype=np.float32))
        else:
            _block, type_size = GGML_QUANT_SIZES[tensor_type]
            writer.add_tensor(
                f"t{index}", np.zeros((rows, type_size), dtype=np.uint8), raw_dtype=tensor_type
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestAFileTheGgufPackageWroteIsJudgedExactly:
    """The writer is the oracle: a whole file is not cut, and one byte less is."""

    @pytest.mark.parametrize(
        ("types", "alignment"),
        [
            pytest.param([GGMLQuantizationType.F32], None, id="f32"),
            pytest.param([GGMLQuantizationType.Q4_0], None, id="q4_0"),
            pytest.param([GGMLQuantizationType.Q4_K], None, id="q4_k"),
            pytest.param([GGMLQuantizationType.Q8_0], None, id="q8_0"),
            pytest.param(
                [
                    GGMLQuantizationType.F32,
                    GGMLQuantizationType.Q4_0,
                    GGMLQuantizationType.Q4_K,
                    GGMLQuantizationType.Q8_0,
                ],
                None,
                id="mixed",
            ),
            pytest.param(
                [GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q8_0], 64, id="aligned-64"
            ),
        ],
    )
    def test_a_whole_file_is_whole_and_one_byte_less_is_cut(
        self, tmp_path: Path, types: list[GGMLQuantizationType], alignment: int | None
    ) -> None:
        whole = _written_gguf(tmp_path / "whole.gguf", types, alignment)
        cut = _write(tmp_path / "cut.gguf", whole.read_bytes()[:-1])
        assert not header_probe.gguf_is_cut_off(whole)
        assert header_probe.gguf_is_cut_off(cut)


class TestACutGgufIsUnknown:
    @pytest.mark.parametrize(
        "blob",
        [
            pytest.param(_gguf()[:-1], id="cut-in-the-tensor-data"),
            pytest.param(_gguf()[:60], id="cut-in-the-tensor-table"),
            pytest.param(
                _gguf(
                    extra_kv=_string(b"tokens")
                    + struct.pack("<IIQ", 9, 8, 2)
                    + _string(b"ab")
                    + _string(b"c"),
                    extra_kv_count=1,
                )[:50],
                id="cut-in-the-key-value-table",
            ),
            pytest.param(_gguf(alignment=256)[:-1], id="cut-aligned-file"),
        ],
    )
    def test_the_estimate_states_unknown_and_skips_the_parser(
        self, tmp_path: Path, parser_calls: list[Path], blob: bytes
    ) -> None:
        est = mm.estimate_gguf(_request(_write(tmp_path / "m.gguf", blob)))
        assert _is_unknown(est)
        assert parser_calls == []

    def test_the_cut_is_judged_again_when_the_file_changes(
        self, tmp_path: Path, parser_calls: list[Path]
    ) -> None:
        blob = _gguf()
        model = _write(tmp_path / "m.gguf", blob[:-1])
        assert _is_unknown(mm.estimate_gguf(_request(model)))
        _write(model, blob)
        assert mm.estimate_gguf(_request(model)).kind is MemoryKind.SIZED


class TestAFileTheReaderCannotSizeGoesToTheParser:
    @pytest.mark.parametrize(
        "blob",
        [
            pytest.param(b"x" * 100, id="no-magic"),
            pytest.param(b"GGUF", id="the-magic-alone"),
            pytest.param(_gguf()[:23], id="shorter-than-the-fixed-header"),
            pytest.param(_gguf(version=1)[:-1], id="version-1"),
            pytest.param(_gguf(tensor_type=9999)[:-1], id="unknown-tensor-type"),
            pytest.param(
                _gguf(
                    extra_kv=_string(b"odd") + struct.pack("<I", 99) + b"\0" * 8, extra_kv_count=1
                )[:-1],
                id="unknown-kv-type",
            ),
            pytest.param(_gguf(alignment=0)[:-1], id="zero-alignment"),
        ],
    )
    def test_the_parser_is_asked(
        self, tmp_path: Path, parser_calls: list[Path], blob: bytes
    ) -> None:
        model = _write(tmp_path / "m.gguf", blob)
        assert mm.estimate_gguf(_request(model)).kind is MemoryKind.SIZED
        assert parser_calls == [model]


class TestAFileThatVanishes:
    def test_a_split_name_in_a_folder_that_is_gone_goes_to_the_parser(
        self, tmp_path: Path, parser_calls: list[Path]
    ) -> None:
        model = tmp_path / "gone" / "big-00001-of-00002.gguf"
        assert mm.estimate_gguf(_request(model)).kind is MemoryKind.SIZED
        assert parser_calls == [model]

    def test_a_file_that_goes_between_the_stat_and_the_read_is_not_cut_off(
        self, tmp_path: Path
    ) -> None:
        assert header_probe._cut_off(str(tmp_path / "gone.gguf"), 100, 0) is False

    def test_an_empty_file_is_not_cut_off(self, tmp_path: Path) -> None:
        empty = _write(tmp_path / "empty.gguf", b"")
        assert header_probe.gguf_is_cut_off(empty) is False

    def test_a_file_emptied_between_the_stat_and_the_read_is_not_cut_off(
        self, tmp_path: Path
    ) -> None:
        empty = _write(tmp_path / "empty.gguf", b"")
        assert header_probe._cut_off(str(empty), 100, 0) is False

    def test_a_file_that_shrinks_during_the_read_is_a_truncation(self, tmp_path: Path) -> None:
        model = _write(tmp_path / "m.gguf", b"GGUF" + b"\0" * 8)
        with model.open("rb") as handle:
            view = header_probe._FileView(handle, 1000)
            with pytest.raises(header_probe._TruncatedHeaderError):
                _chunk = view[0:50]


_GIB = 1024**3


@pytest.mark.usefixtures("stated_sizes")
class TestACutFileLargerThanTheReserve:
    def test_it_is_charged_at_least_its_size_on_disk(self, tmp_path: Path) -> None:
        size = mm.UNKNOWN_SIZE_RESERVE_BYTES + _GIB
        model = cut_file_of(tmp_path / "big.gguf", declared_bytes=size + 4 * _GIB, size=size)
        est = mm.estimate_gguf(_request(model))
        assert (est.kind, est.size_known, est.device_bytes) == (
            MemoryKind.FIXED_RESERVE,
            False,
            size,
        )

    def test_a_cut_file_under_the_reserve_is_charged_the_reserve(self, tmp_path: Path) -> None:
        model = cut_file_of(tmp_path / "small.gguf", declared_bytes=4096, size=60)
        assert _is_unknown(mm.estimate_gguf(_request(model)))

    def test_a_split_set_missing_a_shard_is_charged_its_shards_on_disk(
        self, tmp_path: Path
    ) -> None:
        first = cut_file_of(
            tmp_path / "big-00001-of-00002.gguf",
            declared_bytes=mm.UNKNOWN_SIZE_RESERVE_BYTES + 4 * _GIB,
            size=mm.UNKNOWN_SIZE_RESERVE_BYTES + _GIB,
        )
        est = mm.estimate_gguf(_request(first))
        assert (est.size_known, est.device_bytes) == (False, first.stat().st_size)


@pytest.mark.usefixtures("stated_sizes")
class TestACutFileOnTheCpuDecodePath:
    """The CPU chat figure keeps the unknown charge and never reads the cut file's header."""

    @pytest.fixture
    def shape_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        calls: list[Path] = []

        def _shape(model_path: Path) -> AttentionShape:
            calls.append(model_path)
            return AttentionShape(heads=28, kv_heads=4, vocab=151_936)

        monkeypatch.setattr(mm, "attention_shape", _shape)
        return calls

    @pytest.mark.parametrize("flash_attn", [True, False], ids=["flash-on", "flash-off"])
    @pytest.mark.parametrize(
        "size", [60, mm.UNKNOWN_SIZE_RESERVE_BYTES + _GIB], ids=["under-reserve", "over-reserve"]
    )
    def test_the_charge_is_the_unknown_charge_and_the_header_is_not_read(
        self,
        tmp_path: Path,
        parser_calls: list[Path],
        shape_calls: list[Path],
        flash_attn: bool,
        size: int,
    ) -> None:
        model = cut_file_of(tmp_path / "m.gguf", declared_bytes=size + 4 * _GIB, size=size)
        request = replace(_request(model), on_cpu=True, flash_attn=flash_attn)
        est = LLAMA.estimate_memory(request)
        charge = max(mm.UNKNOWN_SIZE_RESERVE_BYTES, size)
        assert (est.kind, est.size_known, est.resident_footprint(unified=True)) == (
            MemoryKind.FIXED_RESERVE,
            False,
            charge,
        )
        assert (parser_calls, shape_calls) == ([], [])

    def test_a_whole_file_still_gets_the_cpu_figure(
        self,
        tmp_path: Path,
        parser_calls: list[Path],
        shape_calls: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repacked = 5
        monkeypatch.setattr(mm, "repacked_weight_bytes", lambda _path: repacked)
        model = _write(tmp_path / "m.gguf", _gguf())
        request = replace(_request(model), on_cpu=True, flash_attn=True)
        est = LLAMA.estimate_memory(request)
        unused_logits = (512 - 1) * 151_936 * 8
        assert est.mapped_weights_bytes == _PARSED.mapped_weights_bytes + repacked
        assert est.shared_bytes == (
            _PARSED.unified_bytes - unused_logits + mm.CPU_RUNTIME_ALLOWANCE_BYTES
        )
        assert (parser_calls, shape_calls) == ([model], [model])


class TestTheUnknownSizeText:
    def test_it_names_the_subject_and_the_reserve(self) -> None:
        assert mm.unknown_size_text("this model") == (
            "The size of this model is unknown, so 8.0 GiB is reserved for it."
        )

    def test_it_states_the_charge_of_a_cut_file_larger_than_the_reserve(self) -> None:
        charge = mm.UNKNOWN_SIZE_RESERVE_BYTES + _GIB
        assert mm.unknown_size_text("this model", charge) == (
            "The size of this model is unknown, so 9.0 GiB is reserved for it."
        )


class TestBytesOnDisk:
    def test_a_single_file_is_its_size(self, tmp_path: Path) -> None:
        model = _write(tmp_path / "m.gguf", b"x" * 7)
        assert header_probe.gguf_bytes_on_disk(model) == 7

    def test_a_split_set_is_the_sum_of_its_present_shards(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "m-00001-of-00003.gguf", b"x" * 5)
        _write(tmp_path / "m-00003-of-00003.gguf", b"x" * 6)
        _write(tmp_path / "other-00002-of-00003.gguf", b"x" * 100)
        assert header_probe.gguf_bytes_on_disk(first) == 11

    def test_a_file_that_is_gone_is_zero_bytes(self, tmp_path: Path) -> None:
        assert header_probe.gguf_bytes_on_disk(tmp_path / "gone.gguf") == 0

    def test_a_split_name_in_a_folder_that_is_gone_is_zero_bytes(self, tmp_path: Path) -> None:
        assert header_probe.gguf_bytes_on_disk(tmp_path / "gone" / "m-00001-of-00002.gguf") == 0


class _CountingBlob:
    """A blob that records every slice read from it."""

    def __init__(self, size: int) -> None:
        self._size = size
        self.reads: list[slice] = []

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, span: slice) -> bytes:
        self.reads.append(span)
        return b"\0" * (span.stop - span.start)


class TestTheCursorSkipsWithoutReading:
    def test_a_skip_reads_no_bytes_and_moves_the_position(self) -> None:
        blob = _CountingBlob(100)
        cursor = header_probe._HeaderCursor(blob)
        cursor.skip(40)
        assert (blob.reads, cursor.position) == ([], 40)

    def test_a_skip_past_the_end_is_a_truncation(self) -> None:
        cursor = header_probe._HeaderCursor(_CountingBlob(10))
        with pytest.raises(header_probe._TruncatedHeaderError):
            cursor.skip(11)

    def test_a_negative_skip_is_a_truncation(self) -> None:
        cursor = header_probe._HeaderCursor(_CountingBlob(10))
        with pytest.raises(header_probe._TruncatedHeaderError):
            cursor.skip(-1)

    def test_skipping_a_string_reads_only_its_length(self) -> None:
        blob = _CountingBlob(100)
        cursor = header_probe._HeaderCursor(blob)
        cursor.skip_string()
        assert [(span.start, span.stop) for span in blob.reads] == [(0, 8)]

    def test_a_take_still_reads_its_bytes(self) -> None:
        blob = _CountingBlob(100)
        header_probe._HeaderCursor(blob).take(4)
        assert [(span.start, span.stop) for span in blob.reads] == [(0, 4)]


class TestASplitGguf:
    def _shards(self, folder: Path, total: int, present: list[int]) -> list[Path]:
        blob = _gguf()
        return [_write(folder / f"big-{index:05d}-of-{total:05d}.gguf", blob) for index in present]

    def test_a_whole_set_goes_to_the_parser(self, tmp_path: Path, parser_calls: list[Path]) -> None:
        shards = self._shards(tmp_path, 3, [1, 2, 3])
        assert mm.estimate_gguf(_request(shards[0])).kind is MemoryKind.SIZED
        assert parser_calls == [shards[0]]

    @pytest.mark.parametrize(
        ("present", "asked"),
        [
            pytest.param([1, 3], 0, id="middle-shard-missing"),
            pytest.param([1, 2], 0, id="last-shard-missing"),
            pytest.param([2, 3], 0, id="first-shard-missing"),
        ],
    )
    def test_a_missing_shard_states_unknown(
        self, tmp_path: Path, parser_calls: list[Path], present: list[int], asked: int
    ) -> None:
        shards = self._shards(tmp_path, 3, present)
        assert _is_unknown(mm.estimate_gguf(_request(shards[asked])))
        assert parser_calls == []

    def test_a_cut_shard_states_unknown(self, tmp_path: Path, parser_calls: list[Path]) -> None:
        shards = self._shards(tmp_path, 2, [1, 2])
        _write(shards[1], _gguf()[:-1])
        assert _is_unknown(mm.estimate_gguf(_request(shards[0])))
        assert parser_calls == []

    def test_a_shard_of_another_model_does_not_complete_the_set(
        self, tmp_path: Path, parser_calls: list[Path]
    ) -> None:
        shards = self._shards(tmp_path, 2, [1])
        _write(tmp_path / "other-00002-of-00002.gguf", _gguf())
        assert _is_unknown(mm.estimate_gguf(_request(shards[0])))


class TestThePlannerKeepsTheOnDiskSumOfACutFile:
    """A cut file never reads as 0 to a planner caller: no skipped role, no free charge."""

    @pytest.fixture
    def cut_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        model = _write(tmp_path / "m.gguf", _gguf()[:-1])
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: model)
        return model

    def test_a_cut_file_reads_as_its_bytes(self, cut_model: Path) -> None:
        assert planning_mod._weights_bytes(cut_model) == cut_model.stat().st_size > 0

    def test_a_set_missing_a_shard_reads_as_the_shards_present(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "big-00001-of-00003.gguf", _gguf())
        _write(tmp_path / "big-00003-of-00003.gguf", _gguf())
        assert planning_mod._weights_bytes(first) == 2 * len(_gguf())

    def test_the_role_weights_are_the_file_bytes(self, cut_model: Path) -> None:
        size = planning_mod._role_weights_bytes(WorkerRole.CHAT, "org/repo/m.gguf")
        assert size == cut_model.stat().st_size

    def test_a_role_the_estimator_cannot_size_is_not_skipped(self, cut_model: Path) -> None:
        placed = planning_mod._sizing_failure_fallback(
            WorkerRole.CHAT,
            "org/repo/m.gguf",
            RuntimeError("cut"),
            device_count=1,
            total_vram=24 * 1024**3,
            engine=LLAMA,
        )
        assert placed is not None

    def test_a_cpu_role_the_estimator_cannot_size_is_charged_its_bytes(
        self, cut_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail(*_a: object, **_k: object) -> None:
            raise OSError("cut")

        monkeypatch.setattr(planning_mod, "_estimate_role", _fail)
        sized = planning_mod._estimate_cpu_role(
            WorkerRole.CHAT, "org/repo/m.gguf", engine=LLAMA, skipped_not_installed={}
        )
        assert sized is not None
        assert (sized.est_vram_bytes, sized.est_ram_bytes) == (0, cut_model.stat().st_size)

    def test_a_cut_file_is_placeable(
        self, cut_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "_vision_without_mmproj", lambda _r, _ref: False)
        assert planning_mod.role_model_placeable(
            WorkerRole.CHAT, "org/repo/m.gguf", 80 * 1024**3, engine=LLAMA
        )


_IR_XML = '<net><layer><data element_type="f16" shape="8" offset="0" size="100"/></layer></net>'


def _openvino_folder(folder: Path, *, bin_bytes: int, xml: str = _IR_XML) -> MemoryRequest:
    (folder / "openvino_model.xml").write_text(xml, encoding="utf-8")
    (folder / "openvino_model.bin").write_bytes(b"\0" * bin_bytes)
    return _request(folder)


_VISION_PARTS = (
    "openvino_language_model",
    "openvino_text_embeddings_model",
    "openvino_vision_embeddings_model",
)


def _vision_folder(folder: Path, *, cut_part: str | None) -> MemoryRequest:
    for part in _VISION_PARTS:
        (folder / f"{part}.xml").write_text(_IR_XML, encoding="utf-8")
        (folder / f"{part}.bin").write_bytes(b"\0" * (99 if part == cut_part else 100))
    return _request(folder)


class TestACutOpenvinoWeightsFile:
    def test_a_whole_bin_is_charged_its_size(self, tmp_path: Path) -> None:
        request = _openvino_folder(tmp_path, bin_bytes=100)
        est = mm.estimate_memory(ModelFormat.OPENVINO, request)
        assert est.device_bytes == len(_IR_XML) + 100

    def test_a_bin_cut_short_of_the_listed_weights_is_charged_the_reserve(
        self, tmp_path: Path
    ) -> None:
        request = _openvino_folder(tmp_path, bin_bytes=99)
        assert _is_unknown(mm.estimate_memory(ModelFormat.OPENVINO, request))

    def test_the_last_listed_weights_set_the_bound(self, tmp_path: Path) -> None:
        xml = (
            '<net><data offset="0" size="100"/><data element_type="f16" offset="100" size="50"/>'
            '<data shape="4"/></net>'
        )
        whole = mm.estimate_memory(
            ModelFormat.OPENVINO, _openvino_folder(tmp_path, bin_bytes=150, xml=xml)
        )
        cut = mm.estimate_memory(
            ModelFormat.OPENVINO, _openvino_folder(tmp_path, bin_bytes=149, xml=xml)
        )
        assert (whole.device_bytes, whole.size_known) == (len(xml) + 150, True)
        assert _is_unknown(cut)

    def test_an_xml_that_lists_no_weights_keeps_the_bin_size(self, tmp_path: Path) -> None:
        request = _openvino_folder(tmp_path, bin_bytes=7, xml="<net/>")
        assert mm.estimate_memory(ModelFormat.OPENVINO, request).device_bytes == 6 + 7

    def test_an_xml_that_is_not_text_keeps_the_bin_size(self, tmp_path: Path) -> None:
        request = _openvino_folder(tmp_path, bin_bytes=7)
        (tmp_path / "openvino_model.xml").write_bytes(b"\xff\xfe" * 4)
        assert mm.estimate_memory(ModelFormat.OPENVINO, request).device_bytes == 8 + 7

    def test_a_vision_ir_with_one_cut_part_is_charged_the_reserve(self, tmp_path: Path) -> None:
        whole = _vision_folder(tmp_path, cut_part=None)
        est = mm.estimate_memory(ModelFormat.OPENVINO, whole)
        assert (est.device_bytes, est.size_known) == (3 * (len(_IR_XML) + 100), True)
        cut = _vision_folder(tmp_path, cut_part="openvino_vision_embeddings_model")
        assert _is_unknown(mm.estimate_memory(ModelFormat.OPENVINO, cut))

    def test_a_folder_with_no_ir_pair_has_no_bound(self, tmp_path: Path) -> None:
        (tmp_path / "openvino_model.bin").write_bytes(b"\0" * 5)
        est = mm.estimate_memory(ModelFormat.OPENVINO, _request(tmp_path))
        assert est.device_bytes == 5
