"""One test that runs the real gguf-parser against a real GGUF.

Every other estimator test monkeypatches the binary, so the suite pins the shape
of the JSON and nothing about the numbers in it. The parser does not divide the
context by --parallel the way llama-server does, which is why lilbee folds the
slot count into --ctx-size; a version that started dividing it would return
well-formed JSON with doubled KV, shift every placement decision, and leave the
whole suite green. --parallel still reaches the recurrent state, which is sized
per sequence rather than per token, so both flags carry weight.
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
    # Dense attention has no per-sequence term, so folding is exact here. If a
    # future gguf-parser starts dividing the context by --parallel, these stop
    # matching and every placement decision shifts under a green suite. The
    # hybrid case below is deliberately not folded: recurrence is per sequence.
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
        expert_offload=(),
    )
    assert (fitted - _DYNAMIC_CTX_FLOOR) % _DYNAMIC_CTX_QUANTUM == 0
    assert _estimate(tiny_gguf, ctx=fitted, slots=1).vram_bytes <= budget
    assert _estimate(tiny_gguf, ctx=fitted + _DYNAMIC_CTX_QUANTUM, slots=1).vram_bytes > budget


@pytest.fixture(scope="module")
def hybrid_gguf(tmp_path_factory) -> object:
    """A GGUF shaped like the hybrid Qwen3.x family: interleaved attention and
    recurrent layers, declared by ``full_attention_interval`` plus the SSM fields
    the recurrent state is sized from."""
    gguf = pytest.importorskip("gguf")
    numpy = pytest.importorskip("numpy")
    path = tmp_path_factory.mktemp("gguf") / "hybrid.gguf"
    embd, n_layer, ffn = 256, 8, 512
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_block_count(n_layer)
    w.add_context_length(4096)
    w.add_embedding_length(embd)
    w.add_feed_forward_length(ffn)
    w.add_head_count(8)
    w.add_head_count_kv(2)
    w.add_key_length(32)
    w.add_value_length(32)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_uint32("qwen35moe.full_attention_interval", 4)
    w.add_ssm_conv_kernel(4)
    w.add_ssm_inner_size(512)
    w.add_ssm_group_count(1)
    w.add_ssm_state_size(128)
    w.add_tokenizer_model("llama")
    w.add_token_list(["<unk>", "<s>", "</s>"])
    w.add_token_scores([0.0, 0.0, 0.0])
    w.add_token_types([2, 3, 3])
    zeros = lambda *shape: numpy.zeros(shape, dtype=numpy.float32)  # noqa: E731
    w.add_tensor("token_embd.weight", zeros(3, embd))
    w.add_tensor("output_norm.weight", zeros(embd))
    w.add_tensor("output.weight", zeros(3, embd))
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
            w.add_tensor(f"blk.{i}.{name}", zeros(*shape))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_a_hybrid_model_is_charged_per_slot_for_its_recurrent_state(parser, hybrid_gguf) -> None:
    """Folding slots into the context is exact for attention and not for recurrence.

    llama.cpp sizes the attention half by ``n_ctx / n_seq_max`` and the recurrent
    half by ``n_seq_max`` alone, so the slot count has to reach the parser through
    ``--parallel`` as well as the context multiply. Holding the total context
    fixed, the estimate must therefore rise with the slot count, by the same
    amount for each added sequence.
    """
    one = _estimate(hybrid_gguf, ctx=4096, slots=1).vram_bytes
    two = _estimate(hybrid_gguf, ctx=2048, slots=2).vram_bytes
    four = _estimate(hybrid_gguf, ctx=1024, slots=4).vram_bytes

    per_seq = two - one
    if per_seq == 0:
        pytest.skip(
            "this gguf-parser does not model hybrid recurrent state; the engine pin "
            "(ENGINE_GGUF_PARSER_REF) predates it"
        )
    assert per_seq > 0, (one, two)
    # Linear in the sequence count: three more sequences cost three times one.
    assert four - one == 3 * per_seq, (one, two, four)
