"""The memory charge of an OVMS role follows the role: chat, embed or rerank.

The rule is the fit to readings of the OVMS process (CPU, OVMS 2026.4.1, x86 without AVX-512):
a chat role holds twice its weights plus its cache, and an encoder role holds its weights plus
memory in proportion to the tokens of its largest request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import model_memory
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.roles import WorkerRole

MIB = 1024**2
XML_BYTES = 6
BIN_BYTES = 4096
WEIGHTS = XML_BYTES + BIN_BYTES
KV_PER_TOKEN = 16
"""On the CPU: 2 layers x 1 key-value head x head width 4 x 2 (key and value) x 1 byte (u8)."""
KV_PER_TOKEN_OFF_CPU = 32
"""Off the CPU the cache default is f16: the same shape at 2 bytes an element."""
HIDDEN = 8
CTX = 1000
SLOTS = 2
BATCH = 7
CHUNK = 300
RERANK_BATCH = 100


def _folder(tmp_path: Path, *, config: dict[str, int] | None) -> Path:
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "openvino_model.bin").write_bytes(b"\0" * BIN_BYTES)
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


_FULL_CONFIG = {
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "hidden_size": HIDDEN,
}


def _request(
    path: Path, role: WorkerRole | None, ctx: int = CTX, *, on_cpu: bool = True
) -> MemoryRequest:
    fields: dict[str, object] = {
        "model_path": path,
        "ctx": ctx,
        "slots": SLOTS,
        "gpu_layers": 0,
        "flash_attn": False,
        "kv_cache_type": KvCacheType.F16,
        "on_cpu": on_cpu,
    }
    if role is not None:
        fields["role"] = role
    return MemoryRequest(**fields)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _batch_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "embed_batch_sequences", BATCH)
    monkeypatch.setattr(cfg, "chunk_size", CHUNK)


def _charge(path: Path, role: WorkerRole | None, ctx: int = CTX, *, on_cpu: bool = True) -> int:
    return OVMS.estimate_memory(_request(path, role, ctx, on_cpu=on_cpu)).shared_bytes


def test_a_chat_role_holds_its_weights_twice_plus_the_cache_and_the_runtime(
    tmp_path: Path,
) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    assert _charge(path, WorkerRole.CHAT) == (2 * WEIGHTS + KV_PER_TOKEN * CTX * SLOTS + 928 * MIB)


def test_a_chat_role_off_the_cpu_holds_a_two_byte_cache(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    assert _charge(path, WorkerRole.CHAT, on_cpu=False) == (
        2 * WEIGHTS + KV_PER_TOKEN_OFF_CPU * CTX * SLOTS + 928 * MIB
    )


def test_the_cache_bytes_do_not_change_the_charge_of_an_encoder(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    assert _charge(path, WorkerRole.EMBED, on_cpu=False) == _charge(path, WorkerRole.EMBED)


def test_the_charge_reads_the_config_at_module_level() -> None:
    assert model_memory.cfg is cfg


def test_a_request_with_no_role_is_charged_as_chat(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    assert _charge(path, None) == _charge(path, WorkerRole.CHAT)


def test_an_embed_role_holds_its_weights_plus_the_tokens_of_its_largest_request(
    tmp_path: Path,
) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    tokens = BATCH * min(CTX, CHUNK)
    assert _charge(path, WorkerRole.EMBED) == WEIGHTS + 576 * MIB + tokens * HIDDEN * 64


def test_a_rerank_role_holds_its_weights_plus_a_hundred_documents(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    tokens = RERANK_BATCH * min(CTX, CHUNK)
    assert _charge(path, WorkerRole.RERANK) == WEIGHTS + 576 * MIB + tokens * HIDDEN * 120


def test_a_window_below_the_chunk_size_bounds_the_encoder_tokens(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=_FULL_CONFIG)
    assert _charge(path, WorkerRole.EMBED, ctx=100) == (
        WEIGHTS + 576 * MIB + BATCH * 100 * HIDDEN * 64
    )


def test_a_chat_folder_with_no_cache_shape_is_the_unknown_size_reserve(tmp_path: Path) -> None:
    path = _folder(tmp_path, config=None)
    estimate = OVMS.estimate_memory(_request(path, WorkerRole.CHAT))
    assert (estimate.shared_bytes, estimate.size_known) == (UNKNOWN_SIZE_RESERVE_BYTES, False)


@pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
def test_an_encoder_folder_with_no_hidden_width_is_the_unknown_size_reserve(
    tmp_path: Path, role: WorkerRole
) -> None:
    path = _folder(tmp_path, config={})
    estimate = OVMS.estimate_memory(_request(path, role))
    assert (estimate.shared_bytes, estimate.size_known) == (UNKNOWN_SIZE_RESERVE_BYTES, False)
