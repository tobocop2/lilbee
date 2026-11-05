"""A settings change reaches cfg in one step: a reader sees the old values or the new."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from lilbee.app import settings as appset
from lilbee.core import settings as persistent_settings
from lilbee.data.extract import chunk

OLD = {"chunk_size": 1024, "chunk_overlap": 128}
NEW = {"chunk_size": 256, "chunk_overlap": 32}


def _budget(state: Mapping[str, int]) -> tuple[int, int]:
    """The pair the chunker reads, worked out from *state* the way the chunker does."""
    max_chars = state["chunk_size"] * chunk.CHARS_PER_TOKEN
    return max_chars, min(state["chunk_overlap"] * chunk.CHARS_PER_TOKEN, max_chars // 2)


def _pair() -> tuple[int, int]:
    return appset.cfg.chunk_size, appset.cfg.chunk_overlap


@pytest.fixture(autouse=True)
def _old_pair(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """cfg holds OLD in a fresh data root; caches and the index check stay out of the way."""
    monkeypatch.setattr(appset, "_invalidate_caches", lambda keys: None)
    monkeypatch.setattr(appset, "index_has_chunks", lambda: False)
    monkeypatch.setattr(appset.cfg, "data_root", tmp_path)
    for key, value in OLD.items():
        monkeypatch.setattr(appset.cfg, key, value)
    yield


class _PausingValues(Mapping[str, Any]):
    """The values a reload reads; on the second key it lets a reader in and waits for it."""

    def __init__(self, data: dict[str, Any], first_taken: threading.Event, read: threading.Event):
        self._data, self._first_taken, self._read, self._gets = data, first_taken, read, 0

    def __getitem__(self, key: str) -> Any:
        self._gets += 1
        if self._gets == 2:
            self._first_taken.set()
            assert self._read.wait(10)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_a_reader_inside_a_reload_sees_the_old_pair() -> None:
    first_taken, read = threading.Event(), threading.Event()
    seen: list[tuple[int, int]] = []

    def reader() -> None:
        assert first_taken.wait(10)
        seen.append(chunk._char_budget())
        read.set()

    thread = threading.Thread(target=reader)
    thread.start()
    appset._settle(
        set(NEW), embed_in_batch=False, values=_PausingValues(dict(NEW), first_taken, read)
    )
    thread.join(10)
    assert seen == [_budget(OLD)]
    assert _pair() == (NEW["chunk_size"], NEW["chunk_overlap"])


def test_a_reload_from_the_file_publishes_both_keys_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    persistent_settings.update_values(tmp_path, dict(NEW))
    real_sync = persistent_settings.sync_from_resolver
    seen: list[tuple[int, int]] = []

    def sync_then_look(config, keys):
        real_sync(config, keys)
        seen.append(_pair())

    monkeypatch.setattr(appset.persistent_settings, "sync_from_resolver", sync_then_look)
    appset._settle(set(NEW), embed_in_batch=False)
    assert seen == [(OLD["chunk_size"], OLD["chunk_overlap"])]
    assert _pair() == (NEW["chunk_size"], NEW["chunk_overlap"])


def test_a_refused_update_never_changes_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    top_k = appset.cfg.top_k
    real_coerce = appset._coerce_value
    seen: list[tuple[int, int]] = []

    def coerce_and_look(key: str, value: Any) -> Any:
        if key == "top_k":
            seen.append(_pair())
        return real_coerce(key, value)

    monkeypatch.setattr(appset, "_coerce_value", coerce_and_look)
    with pytest.raises(ValueError, match="top_k"):
        appset.apply_settings_update({"chunk_overlap": NEW["chunk_overlap"], "top_k": "many"})
    assert seen == [(OLD["chunk_size"], OLD["chunk_overlap"])]
    assert _pair() == (OLD["chunk_size"], OLD["chunk_overlap"])
    assert appset.cfg.top_k == top_k
    assert "chunk_overlap" not in persistent_settings.load(appset.cfg.data_root)


def _write_from_another_thread(name: str, value: Any) -> None:
    """A writer outside the settings lock (the TUI toggle, placement) sets *name* now."""
    go, done = threading.Event(), threading.Event()

    def writer() -> None:
        assert go.wait(10)
        setattr(appset.cfg, name, value)
        done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    go.set()
    assert done.wait(10)
    thread.join(10)


def test_an_update_keeps_a_write_another_thread_made_during_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appset.cfg, "markdown_rendering", True)
    real_coerce = appset._coerce_value

    def coerce_while_another_thread_writes(key: str, value: Any) -> Any:
        _write_from_another_thread("markdown_rendering", False)
        return real_coerce(key, value)

    monkeypatch.setattr(appset, "_coerce_value", coerce_while_another_thread_writes)
    appset.apply_settings_update({"top_k": 5})
    assert appset.cfg.top_k == 5
    assert appset.cfg.markdown_rendering is False


def test_a_reload_keeps_a_write_another_thread_made_during_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appset.cfg, "top_k", 3)

    class WriteOnRead(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            _write_from_another_thread("top_k", 9)
            return super().__getitem__(key)

    appset._settle({"chunk_size"}, embed_in_batch=False, values=WriteOnRead(chunk_size=512))
    assert appset.cfg.chunk_size == 512
    assert appset.cfg.top_k == 9


def test_a_reader_never_sees_the_new_embedder_with_the_old_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appset.cfg, "embedding_model", "old/Embed-GGUF")
    monkeypatch.setattr(appset.cfg, "embedding_dim", 768)
    asked, read = threading.Event(), threading.Event()
    seen: list[tuple[str, int]] = []

    def reader() -> None:
        assert asked.wait(10)
        seen.append((appset.cfg.embedding_model, appset.cfg.embedding_dim))
        read.set()

    def dim_while_a_reader_looks(ref: str, registry: object = None) -> int:
        asked.set()
        assert read.wait(10)
        return 1024

    monkeypatch.setattr(appset, "_embedder_dim_from_gguf", dim_while_a_reader_looks)
    thread = threading.Thread(target=reader)
    thread.start()
    appset._settle(
        {"embedding_model"}, embed_in_batch=False, values={"embedding_model": "new/Embed-GGUF"}
    )
    thread.join(10)
    assert seen == [("old/Embed-GGUF", 768)]
    assert (appset.cfg.embedding_model, appset.cfg.embedding_dim) == ("new/Embed-GGUF", 1024)


def test_an_update_writes_the_file_after_the_new_embedder_and_its_dim_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appset.cfg, "embedding_model", "old/Embed-GGUF")
    monkeypatch.setattr(appset.cfg, "embedding_dim", 768)
    monkeypatch.setattr(appset, "_pin_legacy_store_meta", lambda: None)
    monkeypatch.setattr(appset, "_coerce_value", lambda key, value: value)
    monkeypatch.setattr(appset, "_embedder_dim_from_gguf", lambda ref, registry=None: 1024)
    monkeypatch.setattr(appset, "_run_deferred_provider_reset", lambda *a, **k: None)
    real_update = persistent_settings.update_values
    seen: list[tuple[tuple[str, int], dict[str, Any]]] = []

    def write_and_look(root: Any, updates: dict[str, Any]) -> None:
        live = (appset.cfg.embedding_model, appset.cfg.embedding_dim)
        real_update(root, updates)
        seen.append((live, persistent_settings.load(root)))

    monkeypatch.setattr(persistent_settings, "update_values", write_and_look)
    appset.apply_settings_update({"embedding_model": "new/Embed-GGUF"})
    assert seen == [(("new/Embed-GGUF", 1024), {"embedding_model": "new/Embed-GGUF"})]
    assert (appset.cfg.embedding_model, appset.cfg.embedding_dim) == ("new/Embed-GGUF", 1024)


def test_a_commit_leaves_a_re_derived_equal_field_in_place() -> None:
    root = appset.cfg.data_root
    appset._restore_values({"chunk_size": NEW["chunk_size"]})
    assert appset.cfg.chunk_size == NEW["chunk_size"]
    assert appset.cfg.data_root is root
