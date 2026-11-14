"""One memory estimator for each model format; the adapters name formats and own no estimate."""

from __future__ import annotations

import json
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import estimate_instance_footprint
from lilbee.providers.model_cache import kv_bytes_per_token

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

_NARROWED_F32_BYTES = 2
"""A float32 tensor counts at this width when the engine runs it as float16."""
_F32 = "F32"
_CONFIG_NAME = "config.json"
_CONFIG_TEXT_KEY = "text_config"
_CONFIG_DTYPE_KEYS = ("dtype", "torch_dtype")
_CONFIG_FLOAT32 = "float32"
_KV_ELEMENT_BYTES = 2
"""The chat engines for safetensors hold each cache element at the model's 2 byte dtype."""
_KV_MLA_KEY = "kv_lora_rank"
_KV_WINDOW_KEY = "sliding_window"
_KV_LAYER_TYPES_KEY = "layer_types"
_KV_FULL_LAYER = "full_attention"
_MIB = 1024**2
CHAT_RUNTIME_BYTES = 512 * _MIB
"""What a safetensors chat engine holds beyond weights and cache: device context and workspace."""
ENCODER_RUNTIME_BYTES = 500 * _MIB
"""What a safetensors encoder server holds beyond its weights (500 MiB measured on CUDA)."""

UNKNOWN_SIZE_RESERVE_BYTES = 8 * 1024**3
"""What a non-GGUF model is charged when neither its header nor its files give a size."""


@dataclass(frozen=True)
class KvPerToken:
    """The cache bytes one token holds, and whether the figure is a ceiling."""

    bytes_per_token: int
    upper_bound: bool
    """True when some layers hold a shorter window than the context: the real cache is smaller."""


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


def weights_from_header(
    parameter_count: Mapping[str, int], *, narrow_f32: bool = False
) -> int | None:
    """Weight bytes from a safetensors header: the tensor count of each dtype times its width.

    *narrow_f32* counts float32 tensors at 2 bytes, for an engine that runs them as float16.
    An empty header, or a count that is negative, not a whole number, or larger than any
    real model, gives None: the size is unknown.
    """
    total = 0
    for dtype, raw in parameter_count.items():
        count = _whole_number(raw)
        if count is None or not 0 <= count <= _MAX_PARAMETER_COUNT:
            return None
        narrowed = narrow_f32 and dtype == _F32
        width = _NARROWED_F32_BYTES if narrowed else _DTYPE_BYTES.get(dtype, _WIDEST_DTYPE_BYTES)
        total += count * width
    return total if total > 0 else None


def _config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The text model's keys: ``text_config`` of a multimodal config, else the config itself."""
    section = config.get(_CONFIG_TEXT_KEY)
    return section if isinstance(section, Mapping) else config


def _is_upper_bound(section: Mapping[str, Any]) -> bool:
    """Whether some layer holds a shorter window than the context."""
    if section.get(_KV_WINDOW_KEY) is not None:
        return True
    layer_types = section.get(_KV_LAYER_TYPES_KEY)
    return isinstance(layer_types, list) and any(kind != _KV_FULL_LAYER for kind in layer_types)


def _positive_int(value: object) -> int | None:
    """*value* as a whole number above zero, else None."""
    count = _whole_number(value)
    return count if count is not None and count > 0 else None


def kv_per_token_from_config(config: Mapping[str, Any]) -> KvPerToken | None:
    """The cache bytes per token a model's ``config.json`` implies, or None when it cannot say.

    The cache holds a key and a value per layer for each key-value head. A config with
    no layer, head or width keys gives None, and so does a latent-attention model
    (``kv_lora_rank``), whose cache holds neither shape. A sliding window or a layer
    list that is not all full attention marks the figure as an upper bound.
    """
    section = _config_section(config)
    layers = _positive_int(section.get("num_hidden_layers"))
    heads = _positive_int(section.get("num_attention_heads"))
    if layers is None or heads is None or _KV_MLA_KEY in section:
        return None
    kv_heads = _positive_int(section.get("num_key_value_heads")) or heads
    head_dim = _positive_int(section.get("head_dim"))
    if head_dim is None:
        hidden = _positive_int(section.get("hidden_size"))
        head_dim = hidden // heads if hidden is not None else None
    if not head_dim:
        return None
    meta = {
        "block_count": str(layers),
        "head_count_kv": str(kv_heads),
        "key_length": str(head_dim),
        "value_length": str(head_dim),
    }
    return KvPerToken(
        bytes_per_token=kv_bytes_per_token(meta, k_elem_bytes=_KV_ELEMENT_BYTES),
        upper_bound=_is_upper_bound(section),
    )


def _unknown_estimate(weights: int | None) -> MemoryEstimate:
    """The stated reserve, or the weights when they are larger; the size is flagged unknown."""
    charge = max(UNKNOWN_SIZE_RESERVE_BYTES, weights or 0)
    return MemoryEstimate(
        kind=MemoryKind.FIXED_RESERVE,
        device_bytes=charge,
        host_bytes=0,
        shared_bytes=charge,
        size_known=False,
    )


def _sized_estimate(charge: int) -> MemoryEstimate:
    return MemoryEstimate(
        kind=MemoryKind.FIXED_RESERVE, device_bytes=charge, host_bytes=0, shared_bytes=charge
    )


def fixed_weights_estimate(weights: int | None) -> MemoryEstimate:
    """The weights as a fixed charge, or the unknown-size reserve when *weights* is not a size."""
    if weights is None or weights <= 0:
        return _unknown_estimate(None)
    return _sized_estimate(weights)


def safetensors_chat_estimate(
    weights: int | None, kv: KvPerToken | None, ctx: int, slots: int
) -> MemoryEstimate:
    """Weights, one cache of *ctx* tokens per slot, and the runtime allowance.

    Unknown weights or an unknown cache give the unknown-size reserve.
    """
    if weights is None or kv is None:
        return _unknown_estimate(weights)
    return _sized_estimate(weights + kv.bytes_per_token * ctx * slots + CHAT_RUNTIME_BYTES)


def encoder_estimate(weights: int | None) -> MemoryEstimate:
    """The weights plus the encoder server's runtime allowance; an encoder keeps no cache."""
    if weights is None:
        return _unknown_estimate(None)
    return _sized_estimate(weights + ENCODER_RUNTIME_BYTES)


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


def installed_safetensors_bytes(model_path: Path) -> int:
    """Bytes of the ``.safetensors`` files of an installed model; 0 when it holds none."""
    return _sum_suffix(_installed_sizes(model_path), _SAFETENSORS_SUFFIX)


def installed_config(model_path: Path) -> dict[str, Any] | None:
    """The ``config.json`` beside an installed model's weights, or None when it is not a mapping."""
    folder = model_path if model_path.is_dir() else model_path.parent
    try:
        config = json.loads((folder / _CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _is_float32(config: Mapping[str, Any] | None) -> bool:
    """Whether a config declares float32 weights, at its top level or in ``text_config``."""
    if config is None:
        return False
    return any(
        section.get(key) == _CONFIG_FLOAT32
        for section in (config, _config_section(config))
        for key in _CONFIG_DTYPE_KEYS
    )


def estimate_installed_encoder(model_path: Path, *, narrow_f32: bool) -> MemoryEstimate:
    """The encoder figure of an installed folder, its float32 weights halved when *narrow_f32*."""
    weights = installed_safetensors_bytes(model_path)
    if narrow_f32 and _is_float32(installed_config(model_path)):
        weights //= 2
    return encoder_estimate(weights or None)


def _estimate_installed_chat(request: MemoryRequest) -> MemoryEstimate:
    """The chat figure of an installed folder: its weight files and the cache its config implies."""
    config = installed_config(request.model_path)
    return safetensors_chat_estimate(
        installed_safetensors_bytes(request.model_path) or None,
        kv_per_token_from_config(config) if config is not None else None,
        request.ctx,
        request.slots,
    )


def _estimate_openvino(request: MemoryRequest) -> MemoryEstimate:
    return _sized_estimate(weights_from_files(_installed_sizes(request.model_path)))


def _estimate_xdna(request: MemoryRequest) -> MemoryEstimate:
    return _weights_estimate(UNKNOWN_SIZE_RESERVE_BYTES)


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
    ModelFormat.MLX: _estimate_installed_chat,
    ModelFormat.SAFETENSORS: _estimate_installed_chat,
    ModelFormat.OPENVINO: _estimate_openvino,
    ModelFormat.XDNA: _estimate_xdna,
}


def estimate_memory(fmt: ModelFormat, request: MemoryRequest) -> MemoryEstimate:
    """The estimate for *request* under the one estimator that *fmt* names."""
    return FORMAT_ESTIMATORS[fmt](request)
