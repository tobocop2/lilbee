"""Pins on the sizing fallbacks that the first tests left open: the pin exemption and the inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.catalog.pull_fit import _launch_ctx
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole

_TARGET = 16384


@pytest.fixture
def clean_downshift() -> Iterator[None]:
    planning.clear_ctx_downshift()
    yield
    planning.clear_ctx_downshift()


@pytest.fixture
def unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "chat_n_ctx_target", _TARGET)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)


def test_a_pinned_chat_is_priced_at_the_pin_whatever_step_is_recorded(
    monkeypatch: pytest.MonkeyPatch, clean_downshift: None, unpinned: None
) -> None:
    planning.apply_ctx_downshift(WorkerRole.CHAT, _TARGET)
    assert planning.record_ctx_downshift(WorkerRole.CHAT)
    assert planning.downshifted_ctx(WorkerRole.CHAT, _TARGET) < _TARGET
    monkeypatch.setattr(cfg, "num_ctx", 3000)
    assert planning.downshifted_ctx(WorkerRole.CHAT, _TARGET) == _TARGET


def test_a_pinned_chat_launch_leaves_the_downshift_base_unrecorded(
    monkeypatch: pytest.MonkeyPatch, clean_downshift: None, unpinned: None
) -> None:
    monkeypatch.setattr(cfg, "num_ctx", 3000)
    assert planning.apply_ctx_downshift(WorkerRole.CHAT, _TARGET) == _TARGET
    assert planning._ctx_downshift_store.base(WorkerRole.CHAT) is None


def test_the_verdict_names_its_scratch_folder_for_the_last_segment_of_the_repo(
    clean_downshift: None, unpinned: None
) -> None:
    seen: list[Path] = []

    def model_meta(folder: Path) -> dict[str, str]:
        seen.append(folder)
        return {"context_length": "not a number"}

    engine = SimpleNamespace(model_meta=model_meta)
    ctx = _launch_ctx("org/sub/name", None, engine)  # type: ignore[arg-type]
    assert [folder.name for folder in seen] == ["name"]
    assert ctx == engine_params.DEFAULT_NUM_CTX


def _no_fit(*_args: object, **_kwargs: object) -> int:
    raise ProviderError("no fit")


def test_the_header_math_aims_at_the_target_when_memory_allows_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unpinned: None
) -> None:
    monkeypatch.setattr(planning, "fit_chat_ctx", _no_fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 1024)
    meta = {"context_length": "40960"}
    resolved = engine_params.resolve_chat_ctx(model, meta, engine=LLAMA, available_bytes=1 << 40)
    assert resolved == _TARGET


def test_the_static_cap_answers_when_the_window_in_the_header_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unpinned: None
) -> None:
    monkeypatch.setattr(planning, "fit_chat_ctx", _no_fit)
    meta = {"context_length": "not a number"}
    resolved = engine_params.resolve_chat_ctx(
        tmp_path / "absent.gguf", meta, engine=LLAMA, available_bytes=1 << 40
    )
    assert resolved == engine_params.DEFAULT_NUM_CTX
