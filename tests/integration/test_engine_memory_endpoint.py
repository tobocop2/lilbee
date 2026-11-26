"""The engine the integration lane runs serves ``GET /memory``, the readback the product uses.

The product ships the tobocop2 llama.cpp fork because upstream llama-server has
no ``GET /memory``. Everything else in this lane drives whichever llama-server
resolution finds, so an upstream binary on PATH keeps the lane green while the
``/memory`` path never runs. These tests start the resolved binary with the flag
the product launches it with and read the endpoint, so an engine without it, or
with a changed response, turns them red.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from lilbee.providers.fleet.binary import resolve_llama_server
from lilbee.providers.fleet.readback import MEMORY_FLAG, parse_memory_rows, supports_memory_readback
from lilbee.providers.fleet.swap_manager import _pick_free_ports

pytestmark = pytest.mark.slow

_READY_SECONDS = 90.0
_CTX = 256


@pytest.fixture(scope="module")
def engine() -> Path:
    return resolve_llama_server()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    gguf = pytest.importorskip("gguf")
    numpy = pytest.importorskip("numpy")
    path = tmp_path_factory.mktemp("memory-endpoint") / "tiny.gguf"
    embd, ffn, n_layer, vocab = 64, 128, 2, 3
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(n_layer)
    writer.add_context_length(_CTX)
    writer.add_embedding_length(embd)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<unk>", "<s>", "</s>"])
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([2, 3, 3])
    writer.add_tensor("token_embd.weight", numpy.zeros((vocab, embd), dtype=numpy.float32))
    writer.add_tensor("output_norm.weight", numpy.zeros((embd,), dtype=numpy.float32))
    writer.add_tensor("output.weight", numpy.zeros((vocab, embd), dtype=numpy.float32))
    for i in range(n_layer):
        for name, shape in (
            ("attn_norm.weight", (embd,)),
            ("attn_q.weight", (embd, embd)),
            ("attn_k.weight", (embd, embd)),
            ("attn_v.weight", (embd, embd)),
            ("attn_output.weight", (embd, embd)),
            ("ffn_norm.weight", (embd,)),
            ("ffn_gate.weight", (ffn, embd)),
            ("ffn_up.weight", (ffn, embd)),
            ("ffn_down.weight", (embd, ffn)),
        ):
            writer.add_tensor(f"blk.{i}.{name}", numpy.zeros(shape, dtype=numpy.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture
def served(engine: Path, tiny_model: Path, tmp_path: Path) -> Iterator[str]:
    """The base URL of the real engine, started with the readback flag and ready."""
    (port,) = _pick_free_ports(1)
    log_path = tmp_path / "engine.log"
    log = log_path.open("wb")
    proc = subprocess.Popen(
        [
            str(engine),
            "--model",
            str(tiny_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(_CTX),
            "--n-gpu-layers",
            "0",
            MEMORY_FLAG,
        ],
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    base = f"http://127.0.0.1:{port}"

    def engine_log() -> str:
        log.flush()
        return log_path.read_text(encoding="utf-8", errors="replace")[-2000:]

    try:
        deadline = time.monotonic() + _READY_SECONDS
        while time.monotonic() < deadline:
            assert proc.poll() is None, (
                f"{engine} exited with {proc.returncode} before it was ready:\n{engine_log()}"
            )
            try:
                if httpx.get(f"{base}/health", timeout=2.0).status_code == httpx.codes.OK:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise AssertionError(
                f"{engine} was not ready after {_READY_SECONDS:.0f} seconds:\n{engine_log()}"
            )
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=30.0)
        log.close()


def test_the_engine_advertises_the_memory_flag(engine: Path) -> None:
    assert supports_memory_readback(engine), (
        f"{engine} does not list {MEMORY_FLAG} in its --help, so it is not the llama.cpp "
        "fork the product ships and the lane would only run the log readback"
    )


def test_the_engine_answers_memory_with_per_device_rows(served: str) -> None:
    resp = httpx.get(f"{served}/memory", timeout=10.0)
    assert resp.status_code == httpx.codes.OK
    rows = parse_memory_rows(resp.json())
    assert rows, f"/memory answered {resp.text[:300]!r}, which holds no device row"
    assert sum(rows.values()) > 0
