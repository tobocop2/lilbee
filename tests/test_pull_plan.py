"""The pull plan: the one selection of the files a pull fetches, and its pre-flight.

The download and the TUI check before the enqueue both run the plan and then
the pre-flight, so the check charges exactly the files the download fetches.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask

_REPO = "acme/plan-GGUF"


def _entry(fmt: ModelFormat = ModelFormat.GGUF, filename: str = "plan-Q4_K_M.gguf") -> CatalogModel:
    return CatalogModel(
        hf_repo=_REPO,
        gguf_filename=filename,
        size_gb=1.0,
        min_ram_gb=1.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        format=fmt,
        formats=frozenset({fmt}),
    )


def _usage(free: int) -> Any:
    return shutil._ntuple_diskusage(total=free * 4, used=free * 3, free=free)


class _Hub:
    """A Hub that names each file's size and blob, and records every question."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, sizes: dict[str, int]) -> None:
        self.sizes = sizes
        self.asked: list[str] = []
        self.fetched: list[str] = []
        monkeypatch.setattr(dl, "fetch_remote_file", self._remote)
        monkeypatch.setattr(dl, "_repo_sibling_files", lambda _repo, **_kw: sorted(sizes))
        monkeypatch.setattr(dl, "resolve_filename", lambda entry, _engine: entry.gguf_filename)
        monkeypatch.setattr(dl, "_download_with_retry", self._download)

    def _remote(self, _repo: str, name: str) -> dl.RemoteFile:
        self.asked.append(name)
        return dl.RemoteFile(self.sizes[name], f"blob-{name}")

    def _download(
        self, _entry: CatalogModel, config: dl.DownloadConfig, *_a: Any, **_kw: Any
    ) -> Path:
        self.fetched.append(config.filename)
        path = Path(config.cache_dir or "") / "fetched" / config.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * self.sizes[config.filename])
        return path


def _no_cache(monkeypatch: pytest.MonkeyPatch, held: dict[str, Path] | None = None) -> None:
    files = held or {}
    monkeypatch.setattr(dl, "cached_file_path", lambda _repo, name, _dir: files.get(name))


def test_an_xdna_plan_is_empty_and_asks_the_hub_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An XDNA name is an engine tag; no hub function may see it."""

    def _refuse(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("an XDNA plan asked the hub")

    monkeypatch.setattr(dl, "_repo_sibling_files", _refuse)
    monkeypatch.setattr(dl, "fetch_remote_file", _refuse)
    monkeypatch.setattr(dl, "cached_file_path", _refuse)

    plan = dl.plan_pull(_entry(ModelFormat.XDNA, ""), tmp_path, None, None)

    assert plan == dl.PullPlan(_REPO)
    assert plan.missing == []


def test_the_preflight_of_an_empty_plan_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("the pre-flight touched the cache")

    monkeypatch.setattr(dl, "discard_partial_blobs", _refuse)
    monkeypatch.setattr(dl.shutil, "disk_usage", _refuse)

    assert dl.preflight(dl.PullPlan(_REPO), tmp_path) is None


def test_the_preflight_reclaims_the_plan_partials_before_it_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover partial of a planned blob is deleted, so its bytes count as free."""
    blobs = tmp_path / "models--acme--plan-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    partial = blobs / "blob-a.123.incomplete"
    partial.write_bytes(b"p" * 50)
    other = blobs / "blob-other.456.incomplete"
    other.write_bytes(b"p" * 50)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(60 if partial.exists() else 110))
    plan = dl.PullPlan(_REPO, (dl.PlannedFile("a", 100, "blob-a", None),))

    assert dl.preflight(plan, tmp_path) is None
    assert not partial.exists()
    assert other.exists(), "the pre-flight deleted a partial outside the plan"


def test_the_preflight_charges_only_the_files_the_disk_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = tmp_path / "held"
    held.write_bytes(b"h")
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(100))
    plan = dl.PullPlan(
        _REPO,
        (dl.PlannedFile("a", 10**9, None, held), dl.PlannedFile("b", 101, None, None)),
    )

    message = dl.preflight(plan, tmp_path)

    assert message is not None and "Not enough disk space" in message
    assert dl.preflight(dl.PullPlan(_REPO, (plan.files[0],)), tmp_path) is None


def test_an_unknown_size_leaves_the_preflight_without_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(0))
    plan = dl.PullPlan(
        _REPO,
        (dl.PlannedFile("a", 10, None, None), dl.PlannedFile("b", dl._SIZE_UNKNOWN, None, None)),
    )

    assert dl.preflight(plan, tmp_path) is None


def test_a_gguf_plan_charges_the_projector_the_repo_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _Hub(monkeypatch, {"plan-Q4_K_M.gguf": 500, "mmproj-plan-f16.gguf": 300})
    _no_cache(monkeypatch)

    plan = dl.plan_pull(_entry(), tmp_path, object(), None)  # type: ignore[arg-type]

    assert [file.name for file in plan.files] == ["plan-Q4_K_M.gguf"]
    assert plan.projector == dl.PlannedFile(
        "mmproj-plan-f16.gguf", 300, "blob-mmproj-plan-f16.gguf", None
    )
    assert [file.size for file in plan.missing] == [500, 300]
    assert hub.asked == ["plan-Q4_K_M.gguf", "mmproj-plan-f16.gguf"]


def test_a_gguf_pull_needs_an_engine(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="needs an engine"):
        dl.plan_pull(_entry(), tmp_path, None, None)


def test_a_split_gguf_is_held_only_when_every_shard_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set with one shard missing fetches every shard, as one folder."""
    shards = ["plan-00001-of-00002.gguf", "plan-00002-of-00002.gguf"]
    _Hub(monkeypatch, {shards[0]: 4, shards[1]: 4})
    monkeypatch.setattr(dl, "_resolve_mmproj_filename", lambda *_a: None)
    first = tmp_path / shards[0]
    first.write_bytes(b"1234")
    _no_cache(monkeypatch)

    plan = dl.plan_pull(_entry(filename=shards[0]), tmp_path, object(), None)  # type: ignore[arg-type]

    assert [file.held for file in plan.files] == [None, None]
    (tmp_path / shards[1]).write_bytes(b"1234")
    plan = dl.plan_pull(_entry(filename=shards[0]), tmp_path, object(), None)  # type: ignore[arg-type]
    assert [file.held for file in plan.files] == [first, tmp_path / shards[1]]
    assert plan.missing == []


def test_a_projector_beside_a_held_model_skips_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Hub(monkeypatch, {"plan-Q4_K_M.gguf": 4})
    (tmp_path / "plan-Q4_K_M.gguf").write_bytes(b"1234")
    _no_cache(monkeypatch)
    monkeypatch.setattr(dl, "_projector_on_disk", lambda _path: True)

    def _refuse(*_a: Any) -> Any:
        raise AssertionError("listed the repo for a projector already on disk")

    monkeypatch.setattr(dl, "_resolve_mmproj_filename", _refuse)

    plan = dl.plan_pull(_entry(), tmp_path, object(), None)  # type: ignore[arg-type]

    assert plan.projector is None
    assert plan.missing == []


def test_an_unreadable_projector_listing_leaves_the_projector_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Hub(monkeypatch, {"plan-Q4_K_M.gguf": 4})
    _no_cache(monkeypatch)

    def _unreadable(*_a: Any) -> Any:
        raise dl.ProjectorListingError("listing timed out")

    monkeypatch.setattr(dl, "_resolve_mmproj_filename", _unreadable)

    plan = dl.plan_pull(_entry(), tmp_path, object(), None)  # type: ignore[arg-type]

    assert plan.projector is None
    assert [file.name for file in plan.missing] == ["plan-Q4_K_M.gguf"]


def test_a_vision_pull_refuses_before_any_byte_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Room for the model but not the projector: nothing is transferred."""
    hub = _Hub(monkeypatch, {"plan-Q4_K_M.gguf": 500, "mmproj-plan-f16.gguf": 300})
    _no_cache(monkeypatch)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(799))

    with pytest.raises(RuntimeError, match="Not enough disk space"):
        dl.fetch_model_files(_entry(), tmp_path, None, object())  # type: ignore[arg-type]

    assert hub.fetched == []


def test_a_held_projector_is_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _Hub(monkeypatch, {"plan-Q4_K_M.gguf": 5, "mmproj-plan-f16.gguf": 3})
    projector = tmp_path / "mmproj-plan-f16.gguf"
    projector.write_bytes(b"123")
    _no_cache(monkeypatch, {"mmproj-plan-f16.gguf": projector})
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(10**6))
    progress: list[tuple[int, int]] = []

    dest = dl.fetch_model_files(
        _entry(),
        tmp_path,
        None,
        object(),
        on_progress=lambda d, t: progress.append((d, t)),  # type: ignore[arg-type]
    )

    assert dest.name == "plan-Q4_K_M.gguf"
    assert hub.fetched == ["plan-Q4_K_M.gguf"]
    assert progress[-1] == (5, 5)


def test_a_format_plan_does_not_ask_the_hub_about_held_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _Hub(monkeypatch, {"model.safetensors": 40, "config.json": 2})
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    _no_cache(monkeypatch, {"config.json": config})

    plan = dl.plan_pull(_entry(ModelFormat.SAFETENSORS, ""), tmp_path, None, None)

    assert hub.asked == ["model.safetensors"]
    assert dl.PlannedFile("config.json", 2, None, config) in plan.files
    assert [file.name for file in plan.missing] == ["model.safetensors"]


def test_an_index_that_does_not_fit_is_refused_before_it_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = dl.SAFETENSORS_INDEX
    names = {index: 50, "model-00001-of-00001.safetensors": 10, "config.json": 2}
    hub = _Hub(monkeypatch, names)
    _no_cache(monkeypatch)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(49))

    with pytest.raises(RuntimeError, match="Not enough disk space"):
        dl.plan_pull(_entry(ModelFormat.SAFETENSORS, ""), tmp_path, None, None)

    assert hub.fetched == []


def test_a_held_index_is_read_without_a_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = dl.SAFETENSORS_INDEX
    shard = "model-00001-of-00001.safetensors"
    hub = _Hub(monkeypatch, {index: 50, shard: 10, "config.json": 2})
    held = tmp_path / index
    held.write_text(json.dumps({"weight_map": {"w": shard}}), encoding="utf-8")
    _no_cache(monkeypatch, {index: held})

    plan = dl.plan_pull(_entry(ModelFormat.SAFETENSORS, ""), tmp_path, None, None)

    assert hub.fetched == []
    assert index not in [file.name for file in plan.missing]


def test_a_gated_file_raises_the_login_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated answer is definitive: the transfer would be refused, so the size is no guess."""
    from huggingface_hub.errors import GatedRepoError

    def _gated(*_a: Any) -> Any:
        raise GatedRepoError("restricted", response=MagicMock())

    monkeypatch.setattr(dl, "_hf_file_metadata", _gated)

    with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
        dl.fetch_remote_file(_REPO, "plan-Q4_K_M.gguf")


def test_download_mmproj_returns_none_when_the_repo_lists_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(dl, "_resolve_mmproj_filename", lambda *_a: None)

    assert dl.download_mmproj(_entry()) is None


def test_download_mmproj_refuses_a_projector_that_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _Hub(monkeypatch, {"mmproj-plan-f16.gguf": 300})
    _no_cache(monkeypatch)
    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(dl, "hf_token", lambda: None)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(299))

    with pytest.raises(RuntimeError, match="Not enough disk space"):
        dl.download_mmproj(_entry())

    assert hub.fetched == []
