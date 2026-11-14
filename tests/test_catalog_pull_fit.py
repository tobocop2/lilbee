"""The memory a safetensors chat model needs, worked out from the Hub before it downloads."""

from __future__ import annotations

import pytest

from lilbee.catalog import hf_client
from lilbee.catalog.pull_fit import PullFit, chat_pull_fit
from lilbee.providers.fleet.model_memory import CHAT_RUNTIME_BYTES, UNKNOWN_SIZE_RESERVE_BYTES

GIB = 1024**3
QWEN3_4B = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2560,
    "max_position_embeddings": 40960,
    "sliding_window": None,
}
KV_QWEN3_4B = 147456
WEIGHTS = 8044982000


def _hub(
    monkeypatch: pytest.MonkeyPatch, config: dict | None, counts: dict[str, int]
) -> list[tuple[str, object]]:
    """Stand in for the two Hub reads; the list records each call and its token."""
    calls: list[tuple[str, object]] = []

    def read_config(repo: str, *, token: object = None) -> dict | None:
        calls.append(("config", token))
        return config

    def read_counts(repo: str, *, token: object = None) -> dict[str, int]:
        calls.append(("counts", token))
        return counts

    monkeypatch.setattr(hf_client, "read_model_config", read_config)
    monkeypatch.setattr(hf_client, "read_parameter_count", read_counts)
    return calls


class TestChatPullFit:
    def test_charges_weights_cache_and_runtime_at_the_configured_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _hub(monkeypatch, QWEN3_4B, {"BF16": WEIGHTS // 2})
        fit = chat_pull_fit("org/model", 8192, token="t")
        assert fit == PullFit(
            device_bytes=WEIGHTS + KV_QWEN3_4B * 8192 + CHAT_RUNTIME_BYTES,
            ctx=8192,
            size_known=True,
            upper_bound=False,
        )
        assert sorted(calls) == [("config", "t"), ("counts", "t")]

    def test_the_context_is_capped_by_the_trained_context_of_the_fetched_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "max_position_embeddings": 4096}, {"BF16": 100})
        fit = chat_pull_fit("org/model", 8192)
        assert fit.ctx == 4096
        assert fit.device_bytes == 200 + KV_QWEN3_4B * 4096 + CHAT_RUNTIME_BYTES

    def test_a_trained_context_in_the_text_section_caps_the_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = {**QWEN3_4B, "max_position_embeddings": 2048}
        _hub(monkeypatch, {"text_config": text}, {"BF16": 100})
        assert chat_pull_fit("org/model", 8192).ctx == 2048

    @pytest.mark.parametrize("trained", [None, 0, -5, "long", True])
    def test_a_trained_context_that_is_unusable_leaves_the_context_alone(
        self, monkeypatch: pytest.MonkeyPatch, trained: object
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "max_position_embeddings": trained}, {"BF16": 100})
        assert chat_pull_fit("org/model", 8192).ctx == 8192

    def test_a_context_below_the_trained_context_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        assert chat_pull_fit("org/model", 2048).ctx == 2048

    def test_each_slot_holds_its_own_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        one = chat_pull_fit("org/model", 1000, slots=1)
        two = chat_pull_fit("org/model", 1000, slots=2)
        assert two.device_bytes - one.device_bytes == KV_QWEN3_4B * 1000

    def test_a_sliding_window_makes_the_figure_an_upper_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "sliding_window": 1024}, {"BF16": 100})
        assert chat_pull_fit("org/model", 8192).upper_bound is True

    @pytest.mark.parametrize(
        ("config", "counts"),
        [
            (None, {"BF16": 100}),
            (QWEN3_4B, {}),
            ({**QWEN3_4B, "kv_lora_rank": 512}, {"BF16": 100}),
            ({"hidden_size": 8}, {"BF16": 100}),
        ],
        ids=["no config", "no counts", "latent attention", "no layer keys"],
    )
    def test_a_missing_part_gives_an_unknown_size_at_the_reserve(
        self, monkeypatch: pytest.MonkeyPatch, config: dict | None, counts: dict[str, int]
    ) -> None:
        _hub(monkeypatch, config, counts)
        fit = chat_pull_fit("org/model", 8192)
        assert (fit.device_bytes, fit.size_known, fit.upper_bound) == (
            UNKNOWN_SIZE_RESERVE_BYTES,
            False,
            False,
        )

    def test_the_figure_matches_the_estimate_of_the_installed_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import json

        from lilbee.core.config.enums import KvCacheType
        from lilbee.providers.fleet.engines.mistral import MISTRAL
        from lilbee.providers.fleet.memory_types import MemoryRequest

        folder = tmp_path / "model"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(QWEN3_4B), encoding="utf-8")
        (folder / "model.safetensors").write_bytes(b"\0" * 1_000_000)
        _hub(monkeypatch, QWEN3_4B, {"BF16": 500_000})
        installed = MISTRAL.estimate_memory(
            MemoryRequest(folder, 8192, 1, -1, True, KvCacheType.F16)
        )
        assert chat_pull_fit("org/model", 8192).device_bytes == installed.device_bytes


class TestPullFitText:
    def test_a_known_size_names_the_figure_and_the_context(self) -> None:
        fit = PullFit(device_bytes=int(9.5 * GIB), ctx=8192, size_known=True, upper_bound=False)
        assert fit.text() == "This model needs 9.5 GiB at a context of 8192 tokens."

    def test_an_upper_bound_says_at_most(self) -> None:
        fit = PullFit(device_bytes=10 * GIB, ctx=8192, size_known=True, upper_bound=True)
        assert fit.text() == "This model needs at most 10.0 GiB at a context of 8192 tokens."

    def test_an_unknown_size_says_so_and_names_the_reserve(self) -> None:
        fit = PullFit(device_bytes=8 * GIB, ctx=8192, size_known=False, upper_bound=False)
        assert fit.text() == "The size of this model is unknown, so 8.0 GiB is reserved for it."
