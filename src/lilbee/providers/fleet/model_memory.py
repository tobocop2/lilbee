"""One memory estimator for each model format; the adapters name formats and own no estimate."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from lilbee.catalog.formats import ir_parts, ir_weights_file
from lilbee.catalog.header_probe import gguf_bytes_on_disk, gguf_is_incomplete
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.catalog.xdna import read_footprint, read_task
from lilbee.core.config import cfg
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.vram import (
    GgufVramEstimate,
    attention_shape,
    estimate_instance_footprint,
    repacked_weight_bytes,
)
from lilbee.providers.folder_embed_width import read_folder_config
from lilbee.providers.model_cache import kv_bytes_per_token
from lilbee.providers.roles import WorkerRole

_SAFETENSORS_SUFFIX = ".safetensors"
_SAFETENSORS_PREFIX_BYTES = 8
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_SAFETENSORS_METADATA_KEY = "__metadata__"
_SAFETENSORS_OFFSETS_KEY = "data_offsets"
_SAFETENSORS_OFFSET_COUNT = 2
_SAFETENSORS_DTYPE_KEY = "dtype"
_IR_MODEL_STEM = "openvino_model"
_IR_WEIGHTS_SUFFIX = ".bin"
_IR_WEIGHT_SPAN = re.compile(r'<data\b[^>]*?\boffset="(\d+)"[^>]*?\bsize="(\d+)"')
"""A constant's ``offset`` and ``size`` in the ``.bin``, in the order the IR writer emits them."""
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
_CONFIG_TEXT_KEY = "text_config"
_HALF_DTYPES = ("F16", "BF16")
"""The header dtypes a float32 launch holds at twice their stored width."""
_CONFIG_VOCAB_KEY = "vocab_size"
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
"""What a safetensors encoder server holds beyond its weights on CUDA and XDNA (500 MiB measured on
CUDA). A CPU build is charged by vocabulary size instead."""
_CPU_ENCODER_VOCAB_CUT = 100_000
"""A CPU encoder with a vocabulary of this size or more, or an unstated one, peaks far above its
weights at load."""
_CPU_SMALL_VOCAB_WEIGHTS_FACTOR = (3, 2)
_CPU_SMALL_VOCAB_BYTES = 64 * _MIB
_CPU_LARGE_VOCAB_WEIGHTS_FACTOR = (5, 4)
_CPU_LARGE_VOCAB_BYTES = 1280 * _MIB

CPU_RUNTIME_ALLOWANCE_BYTES = 96 * _MIB
"""What a CPU chat server holds beyond the priced parts: heap, libraries and the compute the
parser under-prices on a small model. The measured maximum is 92.8 MB."""
_CPU_UBATCH = 512
"""The ubatch of a chat launch, which the parser's default figure also uses."""
_LOGITS_ROW_BYTES_PER_TOKEN = 8
"""The parser's logits cost for one row, per vocabulary entry."""
_SCORE_BYTES_PER_QUERY_HEAD = 4
"""The f32 attention score matrix of a flash-off launch, per query head."""
_SCORE_BYTES_PER_KV_HEAD = 6
"""The CPU workspace beside the score matrix, per KV head (fitted to three models)."""

# The OVMS constants below are fitted to readings of the OVMS process (VmHWM) on CPU, OVMS
# 2026.4.1, x86 without AVX-512. They do not cover GPU, NPU or ARM hosts.
_OVMS_CPU_KV_ELEMENT_BYTES = 1
"""The CPU plugin holds each cache element as one byte (u8)."""
_OVMS_KV_ELEMENT_BYTES = 2
"""Off the CPU the cache default is f16. Charged until GPU and NPU hosts are read."""
_OVMS_CHAT_RUNTIME_BYTES = 928 * _MIB
"""What a chat role holds beyond its two copies of the weights and its cache."""
_OVMS_ENCODER_RUNTIME_BYTES = 576 * _MIB
"""What an embed or rerank role holds beyond its weights and its request activations."""
_OVMS_EMBED_BYTES_PER_TOKEN_WIDTH = 64
"""Activation bytes an embed role keeps for each token of its largest request, per hidden unit."""
_OVMS_RERANK_BYTES_PER_TOKEN_WIDTH = 120
"""Activation bytes a rerank role keeps for each token of its largest request, per hidden unit."""
OVMS_RERANK_BATCH = 100
"""The most documents lilbee sends in one rerank request. The server allows 10000 by default."""

UNKNOWN_SIZE_RESERVE_BYTES = 8 * 1024**3
"""What a non-GGUF model is charged when neither its header nor its files give a size."""


def unknown_size_text(subject: str, reserve_bytes: int = UNKNOWN_SIZE_RESERVE_BYTES) -> str:
    """The sentence that says *subject* has no known size and what is reserved for it."""
    gib = reserve_bytes / 1024**3
    return f"The size of {subject} is unknown, so {gib:.1f} GiB is reserved for it."


def memory_size_text(num_bytes: float) -> str:
    """*num_bytes* as GiB with one decimal, or whole MiB below one GiB.

    A size above zero never reads as zero: below one MiB it reads "under 1 MiB".
    """
    if 0 < num_bytes < _MIB:
        return "under 1 MiB"
    if num_bytes < 1024**3:
        return f"{round(num_bytes / _MIB)} MiB"
    return f"{num_bytes / 1024**3:.1f} GiB"


def tight_memory_text(subject: str, shortfall_bytes: int) -> str:
    """The sentence that says *subject* needs more GPU memory than is free, and what to do."""
    return (
        f"Memory is tight for {subject}: it is estimated to need "
        # A sub-0.05 GiB shortfall would render as "0.0 GiB more".
        f"{max(shortfall_bytes / 1024**3, 0.1):.1f} GiB more GPU memory than is available. "
        "It will still load on demand, keeping the layers that fit on the GPU and the rest "
        "in system memory; if it runs slowly, free up GPU memory or use a smaller model."
    )


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


def kv_per_token_from_config(
    config: Mapping[str, Any], *, element_bytes: int = _KV_ELEMENT_BYTES
) -> KvPerToken | None:
    """The cache bytes per token a model's ``config.json`` implies, or None when it cannot say.

    The cache holds a key and a value per layer for each key-value head. A config with
    no layer, head or width keys gives None, and so does a latent-attention model
    (``kv_lora_rank``), whose cache holds neither shape. A sliding window or a layer
    list that is not all full attention marks the figure as an upper bound. Each cache
    element takes *element_bytes*.
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
        bytes_per_token=kv_bytes_per_token(meta, k_elem_bytes=element_bytes),
        upper_bound=_is_upper_bound(section),
    )


def _unknown_estimate(weights: int | None) -> MemoryEstimate:
    """The stated reserve, or the weights when they are larger, on the card and in host memory."""
    charge = max(UNKNOWN_SIZE_RESERVE_BYTES, weights or 0)
    return MemoryEstimate(
        kind=MemoryKind.FIXED_RESERVE,
        device_bytes=charge,
        host_bytes=charge,
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


def fixed_file_estimate(model_path: Path) -> MemoryEstimate:
    """The size of the weights at *model_path*, every shard of a split set, as a fixed charge.

    A cut-off GGUF, or a split set with a missing shard, has no known size: it is charged the
    reserve, or the bytes on disk when those are larger. An unreadable file is unknown too.
    """
    weights = gguf_bytes_on_disk(model_path)
    if gguf_is_incomplete(model_path):
        return _unknown_estimate(weights)
    return fixed_weights_estimate(weights)


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


def cpu_encoder_estimate(weights: int, vocab_size: int | None) -> MemoryEstimate:
    """The load peak of a CPU encoder server, which follows the vocabulary and not the task.

    A vocabulary of 1 to 99 999 charges 1.5 times the weights plus 64 MiB. A larger one, or one
    that is unstated or below 1, charges 1.25 times the weights plus 1280 MiB.
    """
    if vocab_size is not None and 0 < vocab_size < _CPU_ENCODER_VOCAB_CUT:
        numerator, denominator = _CPU_SMALL_VOCAB_WEIGHTS_FACTOR
        extra = _CPU_SMALL_VOCAB_BYTES
    else:
        numerator, denominator = _CPU_LARGE_VOCAB_WEIGHTS_FACTOR
        extra = _CPU_LARGE_VOCAB_BYTES
    return _sized_estimate(weights * numerator // denominator + extra)


def _ir_pairs(sizes: Mapping[str, int]) -> list[tuple[str, str]]:
    """The ``(xml, bin)`` pair of each IR part in the listing, or none if any part lacks its bin."""
    pairs = [(part, ir_weights_file(part)) for part in ir_parts(sizes)]
    if any(weights not in sizes for _, weights in pairs):
        return []
    return pairs


def _ir_pair_bytes(sizes: Mapping[str, int]) -> int:
    """Bytes of every IR part's ``.xml`` plus ``.bin`` in the IR folder."""
    return sum(sizes[name] for pair in _ir_pairs(sizes) for name in pair)


def _declared_weight_bytes(xml_path: Path) -> int | None:
    """The weight bytes an IR ``.xml`` lists, or None when it cannot be read."""
    try:
        xml_text = xml_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return max(
        (int(offset) + int(size) for offset, size in _IR_WEIGHT_SPAN.findall(xml_text)), default=0
    )


def _ir_bin_is_cut_off(model_path: Path, sizes: Mapping[str, int]) -> bool:
    """Whether the ``.bin`` of any IR part is shorter than the weight offsets its ``.xml`` lists."""
    for xml_name, bin_name in _ir_pairs(sizes):
        declared = _declared_weight_bytes(model_path / xml_name)
        if declared is not None and sizes[bin_name] < declared:
            return True
    return False


def _ir_weight_bytes(sizes: Mapping[str, int]) -> int | None:
    """Weight bytes of an OpenVINO IR listing, or None when it holds no IR weights."""
    pair = _ir_pair_bytes(sizes)
    if pair > 0:
        return pair
    total = sum(
        size
        for name, size in sizes.items()
        if posixpath.basename(name).startswith(_IR_MODEL_STEM) and name.endswith(_IR_WEIGHTS_SUFFIX)
    )
    return total if total > 0 else None


def weights_from_files(sizes: Mapping[str, int]) -> int:
    """Weight bytes of an OpenVINO IR model from a ``name -> size`` listing.

    Only the ``.xml`` and ``.bin`` of each IR part in the shallowest IR folder count (see
    ``ir_parts``): a pickled ``pytorch_model.bin``, ONNX files and tokenizer IR files are
    not the model's weights. A repo with no complete IR counts the total of its
    ``openvino_model*.bin`` files, and a repo with none gives the unknown-size reserve.
    """
    return _known_or_reserve(_ir_weight_bytes(sizes))


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


def installed_folder_weight_bytes(model_path: Path) -> int:
    """Weight bytes of an installed folder: safetensors, else the OpenVINO IR pair; 0 for none."""
    sizes = _installed_sizes(model_path)
    return _sum_suffix(sizes, _SAFETENSORS_SUFFIX) or _ir_pair_bytes(sizes)


def _read_safetensors_header(path: Path, size: int) -> bytes | None:
    """The JSON header text of the file of *size* bytes, or None when it cannot be read."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(_SAFETENSORS_PREFIX_BYTES)
            if len(prefix) < _SAFETENSORS_PREFIX_BYTES:
                return None
            header_bytes = int.from_bytes(prefix, "little")
            if header_bytes > min(size - _SAFETENSORS_PREFIX_BYTES, _MAX_SAFETENSORS_HEADER_BYTES):
                return None
            return stream.read(header_bytes)
    except OSError:
        return None


def _safetensors_dtype_bytes(path: Path, size: int) -> dict[str, int] | None:
    """The data bytes of each dtype in the file of *size* bytes; None unless the file is whole.

    The file is a u64 header length, the JSON header, then the data the header's
    ``data_offsets`` span; the sizes add up to the file size exactly.
    """
    raw_header = _read_safetensors_header(path, size)
    if raw_header is None:
        return None
    try:
        header = json.loads(raw_header)
    except (ValueError, RecursionError):
        return None
    layout = _header_layout(header)
    if layout is None:
        return None
    data_end, dtype_bytes = layout
    if size != _SAFETENSORS_PREFIX_BYTES + len(raw_header) + data_end:
        return None
    return dtype_bytes


def _header_layout(header: object) -> tuple[int, dict[str, int]] | None:
    """The data end and the bytes of each dtype in a safetensors header, or None when malformed.

    The data end is the largest ``data_offsets`` end. A tensor that names no dtype counts
    under the empty name.
    """
    if not isinstance(header, Mapping):
        return None
    data_end = 0
    dtype_bytes: dict[str, int] = {}
    for name, entry in header.items():
        if name == _SAFETENSORS_METADATA_KEY:
            continue
        offsets = entry.get(_SAFETENSORS_OFFSETS_KEY) if isinstance(entry, Mapping) else None
        if not isinstance(offsets, list) or len(offsets) != _SAFETENSORS_OFFSET_COUNT:
            return None
        start, end = _whole_number(offsets[0]), _whole_number(offsets[1])
        if start is None or end is None or not 0 <= start <= end:
            return None
        data_end = max(data_end, end)
        dtype = entry.get(_SAFETENSORS_DTYPE_KEY)
        name_of_dtype = dtype if isinstance(dtype, str) else ""
        dtype_bytes[name_of_dtype] = dtype_bytes.get(name_of_dtype, 0) + end - start
    return data_end, dtype_bytes


def _whole_safetensors(model_path: Path) -> tuple[int, dict[str, int]]:
    """The file bytes and the data bytes of each dtype of the installed ``.safetensors`` files.

    Both are empty when any file is cut short.
    """
    sizes = _installed_sizes(model_path)
    folder = model_path.parent if model_path.is_file() else model_path
    shards = {
        folder / name: size for name, size in sizes.items() if name.endswith(_SAFETENSORS_SUFFIX)
    }
    dtype_bytes: dict[str, int] = {}
    for path, size in shards.items():
        found = _safetensors_dtype_bytes(path, size)
        if found is None:
            return 0, {}
        for dtype, count in found.items():
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + count
    return sum(shards.values()), dtype_bytes


def whole_safetensors_bytes(model_path: Path) -> int:
    """Bytes of the ``.safetensors`` files of an installed model; 0 when any is cut short.

    A cut-off file makes the sum of the sizes smaller than the weights, so the size is unknown.
    """
    return _whole_safetensors(model_path)[0]


def installed_config(model_path: Path) -> dict[str, Any] | None:
    """The ``config.json`` of an installed model, or None when it is empty or not a mapping."""
    folder = model_path if model_path.is_dir() else model_path.parent
    return read_folder_config(folder) or None


class EncoderCharge(StrEnum):
    """How an engine build holds the weights of an encoder, which sets its memory charge."""

    CUDA_FLOAT16 = "cuda-float16"
    """A CUDA build that loads float32 weights as float16."""
    CUDA_FLOAT32 = "cuda-float32"
    """A CUDA build that loads every weight as float32."""
    VOCABULARY = "vocabulary"
    """Any other build: the charge follows the vocabulary in ``config.json``."""


def estimate_installed_encoder(model_path: Path, *, charge: EncoderCharge) -> MemoryEstimate:
    """The encoder figure of an installed folder.

    A CUDA build is charged the weights it holds plus the runtime allowance. A build that
    loads float32 weights holds each F16 and BF16 tensor at twice its stored size. A build
    that loads float16 holds each F32 tensor at half its stored size. The file header names
    the dtypes, and the config is not read for either.
    """
    weights, dtype_bytes = _whole_safetensors(model_path)
    if not weights:
        return _unknown_estimate(installed_safetensors_bytes(model_path))
    if charge is EncoderCharge.CUDA_FLOAT16:
        return encoder_estimate(weights - dtype_bytes.get(_F32, 0) // 2)
    if charge is EncoderCharge.CUDA_FLOAT32:
        return encoder_estimate(weights + sum(dtype_bytes.get(d, 0) for d in _HALF_DTYPES))
    config = installed_config(model_path)
    vocab_size = _whole_number((config or {}).get(_CONFIG_VOCAB_KEY))
    return cpu_encoder_estimate(weights, vocab_size)


def _estimate_installed_chat(request: MemoryRequest) -> MemoryEstimate:
    """The chat figure of an installed folder: its weight files and the cache its config implies."""
    weights = whole_safetensors_bytes(request.model_path)
    if not weights:
        return _unknown_estimate(installed_safetensors_bytes(request.model_path))
    config = installed_config(request.model_path)
    return safetensors_chat_estimate(
        weights,
        kv_per_token_from_config(config) if config is not None else None,
        request.ctx,
        request.slots,
    )


def _ovms_encoder_estimate(
    weights: int,
    config: Mapping[str, Any] | None,
    request_tokens: int,
    bytes_per_token_width: int,
) -> MemoryEstimate:
    """The weights, the runtime, and the activations of the largest request an encoder serves."""
    hidden = _positive_int(_config_section(config or {}).get("hidden_size"))
    if hidden is None:
        return _unknown_estimate(weights)
    return _sized_estimate(
        weights + _OVMS_ENCODER_RUNTIME_BYTES + request_tokens * hidden * bytes_per_token_width
    )


def _estimate_openvino(request: MemoryRequest) -> MemoryEstimate:
    """The charge of an OVMS role by its task: chat holds its weights twice, plus its cache.

    A request with no role is charged as chat, the larger figure for the same weights. An
    encoder request is sized by the tokens of the largest request lilbee sends: a batch of
    sequences, each no longer than the window or the chunk size.
    """
    sizes = _installed_sizes(request.model_path)
    if _ir_bin_is_cut_off(request.model_path, sizes):
        return _unknown_estimate(None)
    weights = _ir_weight_bytes(sizes)
    if weights is None:
        return _unknown_estimate(None)
    config = installed_config(request.model_path)
    if request.role is WorkerRole.EMBED:
        tokens = cfg.embed_batch_sequences * min(request.ctx, cfg.chunk_size)
        return _ovms_encoder_estimate(weights, config, tokens, _OVMS_EMBED_BYTES_PER_TOKEN_WIDTH)
    if request.role is WorkerRole.RERANK:
        tokens = OVMS_RERANK_BATCH * min(request.ctx, cfg.chunk_size)
        return _ovms_encoder_estimate(weights, config, tokens, _OVMS_RERANK_BYTES_PER_TOKEN_WIDTH)
    element_bytes = _OVMS_CPU_KV_ELEMENT_BYTES if request.on_cpu else _OVMS_KV_ELEMENT_BYTES
    kv = (
        kv_per_token_from_config(config, element_bytes=element_bytes)
        if config is not None
        else None
    )
    if kv is None:
        return _unknown_estimate(weights)
    return _sized_estimate(
        2 * weights + kv.bytes_per_token * request.ctx * request.slots + _OVMS_CHAT_RUNTIME_BYTES
    )


def _estimate_xdna(request: MemoryRequest) -> MemoryEstimate:
    """The charge of the NPU model that a ticket names, from the footprint the ticket records.

    An embedding model is charged as an encoder: the footprint and the runtime allowance. A
    chat model takes the unknown-size reserve, because nothing states what its cache costs
    per token. A ticket with no footprint or no task takes the reserve too.
    """
    footprint = read_footprint(request.model_path)
    try:
        task = read_task(request.model_path)
    except ValueError:
        return _unknown_estimate(None)
    if task is ModelTask.EMBEDDING:
        return encoder_estimate(footprint)
    return safetensors_chat_estimate(footprint, None, request.ctx, request.slots)


def _parse(request: MemoryRequest, *, gpu_layers: int, with_projector: bool) -> GgufVramEstimate:
    return estimate_instance_footprint(
        request.model_path,
        ctx=request.ctx,
        slots=request.slots,
        gpu_layers=gpu_layers,
        flash_attn=request.flash_attn,
        kv_cache_type=request.kv_cache_type,
        mmproj_path=request.mmproj_path if with_projector else None,
        tensor_split=request.tensor_split,
        batch_size=request.batch_size,
        expert_offload=request.expert_offload,
    )


def _device_booked_buffer_bytes(parsed: GgufVramEstimate) -> int:
    """Bytes the parser books to a device that its unified figure leaves out."""
    return sum(
        max(device - unified, 0)
        for device, unified in zip(parsed.per_device_vram, parsed.per_device_unified, strict=True)
    )


def estimate_gguf(request: MemoryRequest) -> MemoryEstimate:
    """The GGUF parser's footprint for one instance, with its engine and mtime caches.

    A pooled CPU load, which launches with a batch raised to its context, is sized with no
    layer offloaded: the parser then books only the batch compute buffer to the device, and
    that buffer lives in host memory. The weights are already charged as the mapped file.
    A CPU chat load keeps the parser figure, which its own estimator corrects. A CPU load
    also holds the repacked copy of the weights when the request says the engine keeps one.

    A cut-off file, or a split set with a missing shard, has no known size; it is charged the
    reserve, or the bytes on disk when those are larger.
    """
    if gguf_is_incomplete(request.model_path):
        return _unknown_estimate(gguf_bytes_on_disk(request.model_path))
    pooled_on_cpu = request.on_cpu and request.batch_size is not None
    gpu_layers = 0 if pooled_on_cpu else request.gpu_layers
    parsed = _parse(request, gpu_layers=gpu_layers, with_projector=True)
    compute_buffer = (
        _device_booked_buffer_bytes(_parse(request, gpu_layers=0, with_projector=False))
        if pooled_on_cpu
        else 0
    )
    return MemoryEstimate(
        kind=MemoryKind.SIZED,
        device_bytes=parsed.vram_bytes,
        host_bytes=parsed.ram_bytes,
        shared_bytes=parsed.unified_bytes + compute_buffer,
        mapped_weights_bytes=parsed.mapped_weights_bytes
        + (
            repacked_weight_bytes(request.model_path)
            if request.on_cpu and request.cpu_repack
            else 0
        ),
        per_device=parsed.per_device_vram,
        per_device_shared=parsed.per_device_unified,
        checkpoint_bytes=_checkpoint_or_slot_share(parsed, request.slots),
    )


def _checkpoint_or_slot_share(parsed: GgufVramEstimate, slots: int) -> int:
    """One checkpoint's bytes; for a model whose header cannot size it, one slot's share.

    One slot's share of the footprint without the weights holds that slot's whole KV and
    state, so it never under-counts what a checkpoint copies.
    """
    if parsed.checkpoint_bytes is not None:
        return parsed.checkpoint_bytes
    return parsed.unified_bytes // max(1, slots)


def estimate_gguf_cpu_decode(request: MemoryRequest) -> MemoryEstimate:
    """The CPU chat figure: one logits row per slot, scores when flash is off.

    A header whose head counts are absent, or that has no vocabulary, keeps the parser's own
    figure. A header with per-layer head counts (a hybrid recurrent model) gets only the score
    term, sized by its largest layer: the logits and runtime terms are fitted to dense models
    and undercharge a hybrid one. An estimate of unknown size is returned unchanged.
    """
    estimate = estimate_gguf(request)
    if not estimate.size_known:
        return estimate
    shape = attention_shape(request.model_path)
    if shape is None:
        return estimate
    scores = (
        0
        if request.flash_attn
        else _CPU_UBATCH
        * request.ctx
        * (_SCORE_BYTES_PER_QUERY_HEAD * shape.heads + _SCORE_BYTES_PER_KV_HEAD * shape.kv_heads)
    )
    if shape.per_layer:
        return replace(estimate, shared_bytes=estimate.shared_bytes + scores)
    unused_logits = (_CPU_UBATCH - request.slots) * shape.vocab * _LOGITS_ROW_BYTES_PER_TOKEN
    return replace(
        estimate,
        shared_bytes=estimate.shared_bytes - unused_logits + scores + CPU_RUNTIME_ALLOWANCE_BYTES,
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
