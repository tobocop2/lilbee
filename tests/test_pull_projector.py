"""A pull leaves a repo's projector beside the model, or fails and names it.

The Hub listing says whether a repo ships a projector; the engine launch judge
(``find_mmproj_for_model``) says whether one sits beside the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.gguf_meta import find_mmproj_for_model

_REPO = "acme/SmolVLM2-GGUF"
_MODEL = "SmolVLM2-Q4_K_M.gguf"
_LISTING = [_MODEL, "mmproj-SmolVLM2-Q8_0.gguf", "mmproj-SmolVLM2-f16.gguf"]


def _entry(task: ModelTask = ModelTask.CHAT) -> CatalogModel:
    return CatalogModel(
        hf_repo=_REPO,
        gguf_filename=_MODEL,
        size_gb=1.0,
        min_ram_gb=2,
        description="d",
        featured=False,
        downloads=0,
        task=task,
    )


class _Hub:
    """The Hub as the fetch sees it: a listing that may be down, files written flat."""

    def __init__(self, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.models_dir = models_dir
        self.listing: list[str] | None = list(_LISTING)
        self.listing_reads = 0
        self.model_fetches = 0
        self.fetched: list[str] = []
        self.fail_transfer: Exception | None = None
        monkeypatch.setattr(dl, "_repo_sibling_files", self._list)
        monkeypatch.setattr(dl, "resolve_filename", lambda entry, engine: _MODEL)
        monkeypatch.setattr(
            dl, "_shard_is_cached", lambda entry, d, shard, probe: (d / shard).exists()
        )
        monkeypatch.setattr(dl, "_fetch_files", self._fetch_model)
        monkeypatch.setattr(dl, "fetch_remote_file", lambda *a: dl.RemoteFile(size=1, blob="b"))
        monkeypatch.setattr(dl, "_require_disk_space", lambda *a: None)
        monkeypatch.setattr(dl, "_download_with_retry", self._transfer)

    def _list(self, hf_repo: str) -> list[str]:
        self.listing_reads += 1
        if self.listing is None:
            raise RuntimeError(f"Cannot query files for {hf_repo}: 503")
        return self.listing

    def _fetch_model(self, entry, models_dir, token, shards, on_progress, on_probe):
        self.model_fetches += 1
        paths = [models_dir / shard for shard in shards]
        for path in paths:
            path.write_bytes(b"model")
        return paths

    def _transfer(self, entry: CatalogModel, config: dl.DownloadConfig) -> Path:
        if self.fail_transfer is not None:
            raise self.fail_transfer
        self.fetched.append(config.filename)
        path = Path(config.cache_dir) / config.filename
        path.write_bytes(b"projector")
        return path

    def pull(self, entry: CatalogModel) -> Path:
        return dl.fetch_model_files(entry, self.models_dir, None, LLAMA)

    def launch_judge(self) -> str | None:
        try:
            return find_mmproj_for_model(self.models_dir / _MODEL).name
        except ProviderError:
            return None


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Hub:
    return _Hub(tmp_path, monkeypatch)


class TestFreshPull:
    def test_chat_classified_repo_listing_a_projector_gets_the_unquantized_one(
        self, hub: _Hub
    ) -> None:
        hub.pull(_entry(ModelTask.CHAT))
        assert hub.fetched == ["mmproj-SmolVLM2-f16.gguf"]
        assert hub.launch_judge() == "mmproj-SmolVLM2-f16.gguf"

    def test_vision_named_repo_listing_a_projector_gets_it(self, hub: _Hub) -> None:
        hub.pull(_entry(ModelTask.VISION))
        assert hub.launch_judge() == "mmproj-SmolVLM2-f16.gguf"

    def test_text_only_repo_fetches_no_projector_and_does_not_fail(self, hub: _Hub) -> None:
        hub.listing = [_MODEL]
        hub.pull(_entry())
        assert hub.fetched == []
        assert hub.launch_judge() is None

    def test_unreadable_listing_fails_and_names_the_projector(self, hub: _Hub) -> None:
        hub.listing = None
        with pytest.raises(RuntimeError, match=f"vision projector of {_REPO}"):
            hub.pull(_entry())
        assert hub.model_fetches == 1

    def test_failed_projector_transfer_fails_the_pull(self, hub: _Hub) -> None:
        hub.fail_transfer = RuntimeError("projector transfer stalled")
        with pytest.raises(RuntimeError, match="projector transfer stalled"):
            hub.pull(_entry())
        assert hub.launch_judge() is None

    def test_next_pull_after_a_failed_listing_fetches_only_the_projector(self, hub: _Hub) -> None:
        hub.listing = None
        with pytest.raises(RuntimeError, match="vision projector"):
            hub.pull(_entry())
        assert (hub.models_dir / _MODEL).exists()
        hub.listing = list(_LISTING)
        hub.pull(_entry())
        assert hub.model_fetches == 1
        assert hub.launch_judge() == "mmproj-SmolVLM2-f16.gguf"

    def test_projector_already_beside_the_model_needs_no_listing(self, hub: _Hub) -> None:
        hub.pull(_entry())
        hub.listing = None
        reads = hub.listing_reads
        hub.pull(_entry())
        assert hub.listing_reads == reads
        assert hub.fetched == ["mmproj-SmolVLM2-f16.gguf"]
