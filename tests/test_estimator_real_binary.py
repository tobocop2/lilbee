"""One test that runs the real gguf-parser against a real GGUF.

Every other estimator test monkeypatches the binary, so the suite pins the shape
of the JSON and nothing about the numbers in it. gguf-parser ignores --parallel
today, which is why lilbee folds slot count into --ctx-size; a version that
starts honouring it would return well-formed JSON with doubled KV, shift every
placement decision, and leave the whole suite green.
"""

from __future__ import annotations

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet.binary import EngineTool, resolve_engine_tool
from lilbee.providers.fleet.vram import estimate_instance_footprint

# Opts out of the engine-binary seal on purpose: this is the one test whose
# subject is the real parser, so blocking host resolution would leave it
# asserting against a stub, which is what every other estimator test already
# does. It skips cleanly where no parser is installed.
pytestmark = [pytest.mark.slow, pytest.mark.real_engine_resolution]


@pytest.fixture(scope="module")
def parser() -> object:
    try:
        return resolve_engine_tool(EngineTool.GGUF_PARSER)
    except Exception as exc:
        pytest.skip(f"gguf-parser not installed: {exc}")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory) -> object:
    gguf = pytest.importorskip("gguf")
    path = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(2)
    writer.add_context_length(4096)
    writer.add_embedding_length(64)
    writer.add_feed_forward_length(128)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<unk>", "<s>", "</s>"])
    writer.add_token_scores([0.0, 0.0, 0.0])
    writer.add_token_types([2, 3, 3])
    # Real tensors, not just metadata: the parser sizes weights and returns
    # nothing useful for a header-only file.
    numpy = pytest.importorskip("numpy")
    embd, n_layer = 64, 2
    writer.add_tensor("token_embd.weight", numpy.zeros((3, embd), dtype=numpy.float32))
    writer.add_tensor("output_norm.weight", numpy.zeros((embd,), dtype=numpy.float32))
    writer.add_tensor("output.weight", numpy.zeros((3, embd), dtype=numpy.float32))
    for i in range(n_layer):
        for name, shape in (
            ("attn_norm.weight", (embd,)),
            ("attn_q.weight", (embd, embd)),
            ("attn_k.weight", (embd, embd)),
            ("attn_v.weight", (embd, embd)),
            ("attn_output.weight", (embd, embd)),
            ("ffn_norm.weight", (embd,)),
            ("ffn_gate.weight", (128, embd)),
            ("ffn_up.weight", (128, embd)),
            ("ffn_down.weight", (embd, 128)),
        ):
            writer.add_tensor(f"blk.{i}.{name}", numpy.zeros(shape, dtype=numpy.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _estimate(model, *, ctx: int, slots: int):
    return estimate_instance_footprint(
        model,
        ctx=ctx,
        slots=slots,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


def test_the_kv_cache_scales_with_the_context(parser, tiny_gguf) -> None:
    # The property lilbee depends on: KV is linear in ctx, which is what makes
    # folding slots into --ctx-size equivalent to asking for that many slots.
    small = _estimate(tiny_gguf, ctx=1024, slots=1)
    large = _estimate(tiny_gguf, ctx=4096, slots=1)
    assert large.vram_bytes > small.vram_bytes, (small, large)


def test_slots_are_folded_into_the_context_not_passed_through(parser, tiny_gguf) -> None:
    # If a future gguf-parser starts honouring --parallel, these stop matching
    # and every placement decision shifts under a green suite.
    folded = _estimate(tiny_gguf, ctx=2048, slots=2)
    doubled = _estimate(tiny_gguf, ctx=4096, slots=1)
    assert folded.vram_bytes == doubled.vram_bytes


def test_the_single_device_fit_lands_on_the_last_window_that_fits(parser, tiny_gguf) -> None:
    # The bisection against the real parser, not a stub of it: the window it
    # returns fits the budget and the next step up does not.
    from lilbee.providers.fleet.ctx import fit_single_ctx
    from lilbee.providers.model_cache import _DYNAMIC_CTX_FLOOR, _DYNAMIC_CTX_QUANTUM

    budget = _estimate(tiny_gguf, ctx=2048, slots=1).vram_bytes
    fitted = fit_single_ctx(
        tiny_gguf,
        meta=None,
        slots=1,
        available_bytes=budget,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
        kv_cache_type_v=KvCacheType.F16,
        unified=False,
        ctx_ceiling=4096,
    )
    assert (fitted - _DYNAMIC_CTX_FLOOR) % _DYNAMIC_CTX_QUANTUM == 0
    assert _estimate(tiny_gguf, ctx=fitted, slots=1).vram_bytes <= budget
    assert _estimate(tiny_gguf, ctx=fitted + _DYNAMIC_CTX_QUANTUM, slots=1).vram_bytes > budget
