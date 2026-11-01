"""One memory estimator for each model format; the adapters name formats and own no estimate."""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import estimate_instance_footprint

_SAFETENSORS_SUFFIX = ".safetensors"
_IR_MODEL_STEM = "openvino_model"
_IR_SUFFIXES = (".xml", ".bin")
_IR_WEIGHTS_SUFFIX = ".bin"
_GGUF_KEY_METADATA = "metadata"
_GGUF_KEY_FILE_SIZE = "fileSize"

_DTYPE_BYTES: Mapping[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_WIDEST_DTYPE_BYTES = 8
"""A dtype name this table lacks counts at the widest width, so a new dtype over-reserves."""
_MAX_PARAMETER_COUNT = 10**13
"""A tensor count past this is a bad header, not a model."""

UNKNOWN_SIZE_RESERVE_BYTES = 8 * 1024**3
"""What a non-GGUF model is charged when neither its header nor its files give a size."""


def _known_or_reserve(weight_bytes: int | None) -> int:
    """*weight_bytes* when it is a known positive size, else the unknown-size reserve."""
    if weight_bytes is None or weight_bytes <= 0:
        return UNKNOWN_SIZE_RESERVE_BYTES
    return weight_bytes


def _whole_number(value: object) -> int | None:
    """*value* as an int when it is a whole number, else None; a bool is not a count."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def weights_from_header(parameter_count: Mapping[str, int]) -> int:
    """Weight bytes from a safetensors header: the tensor count of each dtype times its width.

    An empty header, or a count that is negative, not a whole number, or larger than any
    real model, gives the unknown-size reserve.
    """
    total = 0
    for dtype, raw in parameter_count.items():
        count = _whole_number(raw)
        if count is None or not 0 <= count <= _MAX_PARAMETER_COUNT:
            return UNKNOWN_SIZE_RESERVE_BYTES
        total += count * _DTYPE_BYTES.get(dtype, _WIDEST_DTYPE_BYTES)
    return _known_or_reserve(total)


def _ir_pair_bytes(sizes: Mapping[str, int]) -> int:
    """Bytes of ``openvino_model.xml`` plus ``.bin`` in the shallowest folder that holds both."""
    folders = sorted(
        (
            posixpath.dirname(name)
            for name in sizes
            if posixpath.basename(name) == _IR_MODEL_STEM + ".xml"
            and posixpath.join(posixpath.dirname(name), _IR_MODEL_STEM + ".bin") in sizes
        ),
        key=lambda folder: (folder.count("/"), folder),
    )
    if not folders:
        return 0
    return sum(
        sizes[posixpath.join(folders[0], _IR_MODEL_STEM + suffix)] for suffix in _IR_SUFFIXES
    )


def weights_from_files(sizes: Mapping[str, int]) -> int:
    """Weight bytes of an OpenVINO IR model from a ``name -> size`` listing.

    Only ``openvino_model.xml`` and ``openvino_model.bin`` in the shallowest folder
    that holds both count: a pickled ``pytorch_model.bin``, ONNX files and tokenizer
    IR files are not the model's weights. A repo with no such pair counts the total of
    its ``openvino_model*.bin`` files, and a repo with none gives the unknown-size reserve.
    """
    pair = _ir_pair_bytes(sizes)
    if pair > 0:
        return pair
    return _known_or_reserve(
        sum(
            size
            for name, size in sizes.items()
            if posixpath.basename(name).startswith(_IR_MODEL_STEM)
            and name.endswith(_IR_WEIGHTS_SUFFIX)
        )
    )


def weights_from_gguf_report(report: Mapping[str, Any]) -> int:
    """The file size a GGUF parser report states, or the unknown-size reserve."""
    metadata = report.get(_GGUF_KEY_METADATA)
    if not isinstance(metadata, Mapping):
        return UNKNOWN_SIZE_RESERVE_BYTES
    return _known_or_reserve(_whole_number(metadata.get(_GGUF_KEY_FILE_SIZE)))


def _installed_sizes(model_path: Path) -> dict[str, int]:
    """``posix name -> size`` of the files of an installed model, a folder or a single file."""
    try:
        if model_path.is_file():
            return {model_path.name: model_path.stat().st_size}
        return {
            path.relative_to(model_path).as_posix(): path.stat().st_size
            for path in model_path.rglob("*")
            if path.is_file()
        }
    except OSError:
        return {}


def _sum_suffix(sizes: Mapping[str, int], suffix: str) -> int:
    return sum(size for name, size in sizes.items() if name.endswith(suffix))


def _weights_estimate(weights: int) -> MemoryEstimate:
    """A weights-only figure, charged whatever the placement, until KV is measured per engine."""
    return MemoryEstimate(
        kind=MemoryKind.FIXED_RESERVE,
        device_bytes=weights,
        host_bytes=0,
        shared_bytes=weights,
    )


def installed_safetensors_bytes(model_path: Path) -> int:
    """Bytes of the ``.safetensors`` files of an installed model; 0 when it holds none."""
    return _sum_suffix(_installed_sizes(model_path), _SAFETENSORS_SUFFIX)


def _estimate_safetensors(request: MemoryRequest) -> MemoryEstimate:
    return _weights_estimate(_known_or_reserve(installed_safetensors_bytes(request.model_path)))


def _estimate_openvino(request: MemoryRequest) -> MemoryEstimate:
    return _weights_estimate(weights_from_files(_installed_sizes(request.model_path)))


def estimate_gguf(request: MemoryRequest) -> MemoryEstimate:
    """The GGUF parser's footprint for one instance, with its engine and mtime caches."""
    parsed = estimate_instance_footprint(
        request.model_path,
        ctx=request.ctx,
        slots=request.slots,
        gpu_layers=request.gpu_layers,
        flash_attn=request.flash_attn,
        kv_cache_type=request.kv_cache_type,
        mmproj_path=request.mmproj_path,
        tensor_split=request.tensor_split,
        batch_size=request.batch_size,
        expert_offload=request.expert_offload,
    )
    return MemoryEstimate(
        kind=MemoryKind.SIZED,
        device_bytes=parsed.vram_bytes,
        host_bytes=parsed.ram_bytes,
        shared_bytes=parsed.unified_bytes,
        per_device=parsed.per_device_vram,
        per_device_shared=parsed.per_device_unified,
    )


FORMAT_ESTIMATORS: Mapping[ModelFormat, Callable[[MemoryRequest], MemoryEstimate]] = {
    ModelFormat.GGUF: estimate_gguf,
    ModelFormat.MLX: _estimate_safetensors,
    ModelFormat.SAFETENSORS: _estimate_safetensors,
    ModelFormat.OPENVINO: _estimate_openvino,
}


def estimate_memory(fmt: ModelFormat, request: MemoryRequest) -> MemoryEstimate:
    """The estimate for *request* under the one estimator that *fmt* names."""
    return FORMAT_ESTIMATORS[fmt](request)
