"""The memory a safetensors chat model needs, worked out from the Hub before it downloads."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import httpx2
import pytest
from huggingface_hub import constants as hub_constants

from lilbee.catalog import hf_client
from lilbee.catalog.pull_fit import PullFit, chat_pull_fit
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.model_memory import CHAT_RUNTIME_BYTES, UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.roles import WorkerRole
from tests._safetensors import safetensors_bytes

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
    def test_charges_weights_cache_and_runtime_at_the_launch_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _hub(monkeypatch, QWEN3_4B, {"BF16": WEIGHTS // 2})
        fit = chat_pull_fit("org/model", MISTRAL, token="t")
        assert fit == PullFit(
            device_bytes=WEIGHTS + KV_QWEN3_4B * 8192 + CHAT_RUNTIME_BYTES,
            ctx=8192,
            size_known=True,
            upper_bound=False,
        )
        assert sorted(calls) == [("config", "t"), ("counts", "t")]

    def test_mistral_at_default_settings_says_the_context_the_placement_estimate_uses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One installed model: the pull verdict and the planner work out the same context."""
        assert cfg.num_ctx is None
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 16384)
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        folder = tmp_path / "model"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(QWEN3_4B), encoding="utf-8")
        planned = planning._placement_estimate_ctx(
            WorkerRole.CHAT, folder, MISTRAL.model_meta(folder), engine=MISTRAL
        )
        assert chat_pull_fit("org/model", MISTRAL).ctx == planned == 8192

    def test_an_engine_that_reads_the_trained_window_caps_the_context_with_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "max_position_embeddings": 4096}, {"BF16": 100})
        fit = chat_pull_fit("org/model", MLX)
        assert fit.ctx == 4096
        assert fit.device_bytes == 200 + KV_QWEN3_4B * 4096 + CHAT_RUNTIME_BYTES

    def test_a_window_past_the_target_leaves_the_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_n_ctx_target", 16384)
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        assert chat_pull_fit("org/model", MLX).ctx == 16384

    def test_a_trained_context_in_the_text_section_caps_the_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = {**QWEN3_4B, "max_position_embeddings": 2048}
        _hub(monkeypatch, {"text_config": text}, {"BF16": 100})
        assert chat_pull_fit("org/model", MLX).ctx == 2048

    @pytest.mark.parametrize("trained", [None, 0, -5, "long", True])
    def test_a_trained_context_that_is_unusable_leaves_the_engine_default(
        self, monkeypatch: pytest.MonkeyPatch, trained: object
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "max_position_embeddings": trained}, {"BF16": 100})
        assert chat_pull_fit("org/model", MLX).ctx == 8192

    def test_a_pinned_context_is_the_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "num_ctx", 2048)
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        assert chat_pull_fit("org/model", MISTRAL).ctx == 2048

    def test_a_pin_past_the_trained_window_is_sized_at_the_window_and_logs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The preview of a candidate repo reports the window and stays silent: the launch warns."""
        monkeypatch.setattr(cfg, "num_ctx", 8192)
        _hub(monkeypatch, {**QWEN3_4B, "max_position_embeddings": 4096}, {"BF16": 100})
        with caplog.at_level(logging.DEBUG):
            fit = chat_pull_fit("org/model", MLX)
        assert fit.ctx == 4096
        assert fit.device_bytes == 200 + KV_QWEN3_4B * 4096 + CHAT_RUNTIME_BYTES
        assert "trained" not in fit.text()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_each_slot_holds_its_own_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "num_ctx", 1000)
        _hub(monkeypatch, QWEN3_4B, {"BF16": 100})
        one = chat_pull_fit("org/model", MISTRAL, slots=1)
        two = chat_pull_fit("org/model", MISTRAL, slots=2)
        assert two.device_bytes - one.device_bytes == KV_QWEN3_4B * 1000

    def test_a_sliding_window_makes_the_figure_an_upper_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hub(monkeypatch, {**QWEN3_4B, "sliding_window": 1024}, {"BF16": 100})
        assert chat_pull_fit("org/model", MISTRAL).upper_bound is True

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
        fit = chat_pull_fit("org/model", MISTRAL)
        assert (fit.device_bytes, fit.size_known, fit.upper_bound) == (
            UNKNOWN_SIZE_RESERVE_BYTES,
            False,
            False,
        )

    def test_the_figure_matches_the_estimate_of_the_installed_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from lilbee.core.config.enums import KvCacheType
        from lilbee.providers.fleet.memory_types import MemoryRequest

        folder = tmp_path / "model"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(QWEN3_4B), encoding="utf-8")
        (folder / "model.safetensors").write_bytes(safetensors_bytes(1_000_000))
        _hub(monkeypatch, QWEN3_4B, {"BF16": 500_000})
        installed = MISTRAL.estimate_memory(
            MemoryRequest(folder, 8192, 1, -1, True, KvCacheType.F16)
        )
        assert chat_pull_fit("org/model", MISTRAL).device_bytes == installed.device_bytes


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


class _FailingSession:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def get(self, *_args: object, **_kwargs: object) -> object:
        raise self._error


class _FailingApi:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def model_info(self, *_args: object, **_kwargs: object) -> object:
        raise self._error


class TestAHubReadThatFails:
    @pytest.mark.parametrize(
        "error",
        [
            httpx2.ReadTimeout("slow"),
            httpx2.ConnectError("down"),
            json.JSONDecodeError("not json", "<html>", 0),
            KeyError("content-length"),
            OSError("offline"),
        ],
        ids=["timeout", "connect", "json", "header", "os"],
    )
    def test_gives_an_unknown_size_whatever_the_read_raised(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        monkeypatch.setattr(hf_client, "get_session", lambda: _FailingSession(error))
        monkeypatch.setattr(hf_client, "HfApi", lambda: _FailingApi(error))
        fit = chat_pull_fit("org/model", MISTRAL)
        assert fit.size_known is False
        assert fit.text() == "The size of this model is unknown, so 8.0 GiB is reserved for it."


class TestARefusedConnection:
    @pytest.fixture
    def closed_port_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
        monkeypatch.setattr(hub_constants, "ENDPOINT", endpoint)
        monkeypatch.setattr(
            hub_constants,
            "HUGGINGFACE_CO_URL_TEMPLATE",
            endpoint + "/{repo_id}/resolve/{revision}/{filename}",
        )

    def test_the_config_read_gives_none(self, closed_port_hub: None) -> None:
        assert hf_client.read_model_config("org/model") is None

    def test_the_figures_read_gives_nothing(self, closed_port_hub: None) -> None:
        assert hf_client.read_parameter_count("org/model") == {}

    def test_the_pull_fit_reserves_the_unknown_size(self, closed_port_hub: None) -> None:
        fit = chat_pull_fit("org/model", MISTRAL)
        assert fit.size_known is False
