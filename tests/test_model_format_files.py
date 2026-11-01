"""Per-format file selection, listing parameters, download routing and manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.app.services import get_services
from lilbee.catalog import download
from lilbee.catalog.formats import formats_from_tags, hf_listing_params, select_files
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat, ModelFormat, ModelTask
from lilbee.core.config.model import cfg
from lilbee.modelhub.registry import ModelRegistry, register_downloaded_model

_TOKENIZER = ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
_JUNK = [".gitattributes", "README.md", "LICENSE"]
_REV = "0" * 40
_QWEN3_EMBEDDING_LISTING = [
    ".gitattributes",
    "1_Pooling/config.json",
    "LICENSE",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
_DENSE_LISTING = [
    *_QWEN3_EMBEDDING_LISTING,
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "2_Dense/pytorch_model.bin",
    "3_Normalize/config.json",
    "docs/config.json",
    "9_x/other.json",
]
_MODULE_FILES = {"modules.json", "1_Pooling/config.json"}


class TestFormatsFromTags:
    def test_mlx_removes_safetensors(self) -> None:
        assert formats_from_tags(["mlx", "safetensors", "transformers"]) == {ModelFormat.MLX}

    def test_openvino_and_safetensors_are_two_formats(self) -> None:
        got = formats_from_tags(["openvino", "safetensors"])
        assert got == {ModelFormat.OPENVINO, ModelFormat.SAFETENSORS}

    def test_unknown_tags_name_no_format(self) -> None:
        assert formats_from_tags(["transformers", "text-generation"]) == frozenset()


class TestHfListingParams:
    @pytest.mark.parametrize("fmt", list(ModelFormat))
    def test_filter_is_the_format(self, fmt: ModelFormat) -> None:
        assert hf_listing_params(fmt, "qwen 3")["filter"] == fmt.value

    def test_only_gguf_keeps_the_search_word(self) -> None:
        assert hf_listing_params(ModelFormat.GGUF, "qwen/3-b")["search"] == "GGUF qwen 3 b"
        assert hf_listing_params(ModelFormat.MLX, "qwen/3-b")["search"] == "qwen 3 b"

    def test_empty_search_for_a_non_gguf_format(self) -> None:
        assert hf_listing_params(ModelFormat.OPENVINO, "  ")["search"] == ""


class TestSelectFiles:
    def test_gguf_picks_the_best_quant_only(self) -> None:
        names = [*_JUNK, "params", "m-Q4_K_M.gguf", "m-Q8_0.gguf"]
        assert select_files(ModelFormat.GGUF, names) == ["m-Q4_K_M.gguf"]

    def test_gguf_split_set_keeps_every_shard(self) -> None:
        names = ["m-Q4_K_M-00001-of-00002.gguf", "m-Q4_K_M-00002-of-00002.gguf"]
        assert select_files(ModelFormat.GGUF, names) == names

    def test_mlx_takes_weights_config_and_tokenizer(self) -> None:
        names = [*_JUNK, "config.json", "model.safetensors", "generation_config.json", *_TOKENIZER]
        got = select_files(ModelFormat.MLX, names)
        assert got[0] == "model.safetensors"
        assert sorted(got) == sorted(["config.json", "model.safetensors", *_TOKENIZER])

    def test_sharded_safetensors_starts_at_the_index(self) -> None:
        shards = [f"model-0000{i}-of-00002.safetensors" for i in (1, 2)]
        names = [
            *_JUNK,
            *shards,
            "model.safetensors.index.json",
            "config.json",
            "pytorch_model.bin",
        ]
        got = select_files(ModelFormat.SAFETENSORS, names)
        assert got[0] == "model.safetensors.index.json"
        assert set(got) == {*shards, "model.safetensors.index.json", "config.json"}

    def test_single_safetensors_takes_generation_config(self) -> None:
        names = ["config.json", "generation_config.json", "model.safetensors", "other.safetensors"]
        assert select_files(ModelFormat.SAFETENSORS, names) == [
            "model.safetensors",
            "config.json",
            "generation_config.json",
        ]

    def test_an_embedding_repo_takes_the_module_files(self) -> None:
        got = select_files(ModelFormat.SAFETENSORS, _QWEN3_EMBEDDING_LISTING)
        assert got[0] == "model.safetensors"
        assert set(got) >= _MODULE_FILES
        assert "config_sentence_transformers.json" not in got
        assert "README.md" not in got

    def test_each_module_folder_gives_its_config_and_weights_and_no_other_format(self) -> None:
        got = set(select_files(ModelFormat.SAFETENSORS, _DENSE_LISTING))
        assert got >= {*_MODULE_FILES, "2_Dense/config.json", "2_Dense/model.safetensors"}
        assert "3_Normalize/config.json" in got
        assert not got & {"2_Dense/pytorch_model.bin", "docs/config.json", "9_x/other.json"}

    def test_a_numbered_folder_without_modules_json_is_not_taken(self) -> None:
        names = ["model.safetensors", "config.json", "2_Dense/config.json"]
        assert select_files(ModelFormat.SAFETENSORS, names) == ["model.safetensors", "config.json"]

    def test_a_gguf_pull_never_takes_the_module_files(self) -> None:
        names = [*_DENSE_LISTING, "Qwen3-Embedding-0.6B-Q8_0.gguf"]
        assert select_files(ModelFormat.GGUF, names) == ["Qwen3-Embedding-0.6B-Q8_0.gguf"]

    def test_an_mlx_pull_does_not_take_the_module_files(self) -> None:
        got = select_files(ModelFormat.MLX, _DENSE_LISTING)
        assert not (_MODULE_FILES | {"2_Dense/config.json"}) & set(got)

    def test_four_of_five_shards_is_refused_naming_the_fifth(self) -> None:
        names = [f"model-0000{i}-of-00005.safetensors" for i in (1, 2, 3, 4)]
        with pytest.raises(ValueError, match=r"model-00005-of-00005\.safetensors is missing"):
            select_files(ModelFormat.SAFETENSORS, [*names, "model.safetensors.index.json"])

    def test_a_shard_count_the_listing_does_not_meet_is_refused(self) -> None:
        names = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2)]
        with pytest.raises(ValueError, match=r"model-00003-of-00003\.safetensors is missing"):
            select_files(ModelFormat.MLX, [*names, "model.safetensors.index.json"])

    @pytest.mark.parametrize("name", ["../openvino_model.xml", "a/../openvino_model.xml"])
    def test_a_name_with_a_parent_part_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="leaves the model folder"):
            select_files(ModelFormat.OPENVINO, [name])

    def test_every_format_goes_through_the_path_check(self) -> None:
        with pytest.raises(ValueError, match="leaves the model folder"):
            select_files(ModelFormat.GGUF, ["../m-Q4_K_M.gguf"])

    def test_safetensors_in_a_subfolder_is_not_a_weight(self) -> None:
        assert select_files(ModelFormat.SAFETENSORS, ["config.json", "sub/model.safetensors"]) == []

    def test_openvino_reads_the_folder_that_holds_the_ir(self) -> None:
        names = [
            "pytorch_model.bin",
            "model.safetensors",
            "onnx/model.onnx",
            "openvino/openvino_model.xml",
            "openvino/openvino_model.bin",
            "openvino/openvino_model_qint8_quantized.xml",
            "openvino/openvino_model_qint8_quantized.bin",
            "modules.json",
            "1_Pooling/config.json",
        ]
        assert select_files(ModelFormat.OPENVINO, names) == [
            "openvino/openvino_model.xml",
            "openvino/openvino_model.bin",
            "1_Pooling/config.json",
            "modules.json",
        ]

    def test_openvino_prefers_the_shallowest_folder(self) -> None:
        names = ["a/b/openvino_model.xml", "a/b/openvino_model.bin", "openvino_model.xml"]
        assert select_files(ModelFormat.OPENVINO, names) == ["openvino_model.xml"]

    @pytest.mark.parametrize("fmt", list(ModelFormat))
    def test_a_repo_of_another_format_selects_nothing(self, fmt: ModelFormat) -> None:
        blobs = [hashlib.sha256(bytes([i])).hexdigest() for i in range(5)]
        other = {
            ModelFormat.GGUF: ["config.json", "model.safetensors"],
            ModelFormat.MLX: ["m-Q4_K_M.gguf", "config.json"],
            ModelFormat.SAFETENSORS: ["m-Q4_K_M.gguf", "openvino_model.xml"],
            ModelFormat.OPENVINO: ["m-Q4_K_M.gguf", "model.safetensors"],
        }[fmt]
        assert select_files(fmt, [*blobs, *other, "params"]) == []


def _hub_row(repo: str, files: list[str], tags: list[str]) -> dict:
    return {
        "id": repo,
        "downloads": 1,
        "pipeline_tag": "text-generation",
        "tags": tags,
        "siblings": [{"rfilename": f} for f in files],
    }


@pytest.mark.real_hf_client
class TestHfClientFormat:
    def _fetch(self, monkeypatch: pytest.MonkeyPatch, rows: list[dict], fmt: ModelFormat):
        seen: list[httpx.QueryParams] = []

        def fake_get(_url: str, params: httpx.QueryParams, **_kw: object) -> httpx.Response:
            seen.append(params)
            return httpx.Response(200, json=rows)

        monkeypatch.setattr(httpx, "get", fake_get)
        return get_services().hf_client.fetch_models(fmt=fmt), seen[0]

    def test_mlx_listing_filters_by_tag_and_names_the_entry_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _hub_row("o/m-4bit", ["config.json", "model.safetensors"], ["mlx", "safetensors"])
        page, params = self._fetch(monkeypatch, [row], ModelFormat.MLX)
        assert params["filter"] == "mlx"
        assert "GGUF" not in params["search"]
        (model,) = page.models
        assert (model.format, model.gguf_filename) == (ModelFormat.MLX, "model.safetensors")
        assert model.compat is ModelCompat.UNKNOWN

    def test_a_repo_with_no_file_of_the_format_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _hub_row("o/m", ["README.md", "pytorch_model.bin"], ["safetensors"])
        page, _params = self._fetch(monkeypatch, [row], ModelFormat.SAFETENSORS)
        assert page.models == []

    def test_a_repo_with_a_partial_shard_set_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = ["model-00001-of-00002.safetensors", "model.safetensors.index.json"]
        row = _hub_row("o/m", files, ["safetensors"])
        page, _params = self._fetch(monkeypatch, [row], ModelFormat.SAFETENSORS)
        assert page.models == []

    def test_gguf_listing_stays_gguf_with_the_search_word(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _hub_row("o/m-gguf", ["m-Q4_K_M.gguf"], ["gguf"])
        page, params = self._fetch(monkeypatch, [row], ModelFormat.GGUF)
        assert params["filter"] == "gguf"
        assert params["search"] == "GGUF"
        assert page.models[0].format is ModelFormat.GGUF
        assert page.models[0].gguf_filename == "m-Q4_K_M.gguf"


def _entry(fmt: ModelFormat, filename: str = "model.safetensors") -> CatalogModel:
    return CatalogModel(
        hf_repo="org/repo",
        gguf_filename=filename,
        size_gb=0.0,
        min_ram_gb=1.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        format=fmt,
    )


class TestDownloadRouting:
    def test_non_gguf_entry_fetches_the_selected_file_set(self, tmp_path: Path) -> None:
        listing = ["README.md", "config.json", "model.safetensors", "tokenizer.json"]
        fetched: list[list[str]] = []

        def fake_fetch(_entry, _dir, _token, files, _progress, _probe, _cached) -> list[Path]:
            fetched.append(files)
            return [tmp_path / name for name in files]

        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=listing),
            mock.patch.object(download, "_fetch_files", side_effect=fake_fetch),
            mock.patch.object(download, "resolve_filename", side_effect=AssertionError("GGUF")),
        ):
            path = download.fetch_model_files(
                _entry(ModelFormat.MLX), tmp_path, None, engine=mock.Mock()
            )
        assert fetched == [["model.safetensors", "config.json", "tokenizer.json"]]
        assert path == tmp_path / "model.safetensors"

    def test_an_embedding_pull_fetches_the_module_files(self, tmp_path: Path) -> None:
        fetched: list[list[str]] = []

        def fake_fetch(_entry, _dir, _token, files, _progress, _probe, _cached) -> list[Path]:
            fetched.append(files)
            return [tmp_path / name for name in files]

        with (
            mock.patch.object(
                download, "_repo_sibling_files", return_value=_QWEN3_EMBEDDING_LISTING
            ),
            mock.patch.object(download, "_fetch_files", side_effect=fake_fetch),
        ):
            download.fetch_format_files(_entry(ModelFormat.SAFETENSORS), tmp_path, None)
        assert set(fetched[0]) >= _MODULE_FILES

    def test_a_model_pulled_before_the_fix_fetches_only_the_missing_files(
        self, tmp_path: Path
    ) -> None:
        old = ["model.safetensors", "config.json", "merges.txt"]
        old += ["tokenizer.json", "tokenizer_config.json", "vocab.json", "generation_config.json"]
        seen: list[tuple[list[str], frozenset[str]]] = []

        def fake_fetch(_entry, _dir, _token, files, _progress, _probe, cached) -> list[Path]:
            seen.append((files, cached))
            return [tmp_path / name for name in files]

        with (
            mock.patch.object(
                download, "_repo_sibling_files", return_value=_QWEN3_EMBEDDING_LISTING
            ),
            mock.patch.object(
                download,
                "_cached_path",
                side_effect=lambda _repo, name, _dir: tmp_path / name if name in old else None,
            ),
            mock.patch.object(download, "_fetch_files", side_effect=fake_fetch),
        ):
            download.fetch_format_files(_entry(ModelFormat.SAFETENSORS), tmp_path, None)
        assert seen[0][1] == frozenset(old)
        assert set(seen[0][0]) - seen[0][1] == _MODULE_FILES

    def test_a_repo_without_the_format_is_refused_by_name(self, tmp_path: Path) -> None:
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=["README.md"]),
            pytest.raises(RuntimeError, match="No openvino model files found in org/repo"),
        ):
            download.fetch_format_files(_entry(ModelFormat.OPENVINO), tmp_path, None)

    def test_a_name_that_leaves_the_snapshot_is_refused_before_any_fetch(
        self, tmp_path: Path
    ) -> None:
        with (
            mock.patch.object(
                download, "_repo_sibling_files", return_value=["../openvino_model.xml"]
            ),
            mock.patch.object(download, "_fetch_files") as fetch,
            pytest.raises(ValueError, match="leaves the model folder"),
        ):
            download.fetch_format_files(_entry(ModelFormat.OPENVINO), tmp_path, None)
        fetch.assert_not_called()

    def test_a_partial_shard_set_is_refused_before_any_fetch(self, tmp_path: Path) -> None:
        listing = [f"model-0000{i}-of-00005.safetensors" for i in (1, 2, 3, 4)]
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=listing),
            mock.patch.object(download, "_fetch_files") as fetch,
            pytest.raises(ValueError, match=r"model-00005-of-00005\.safetensors is missing"),
        ):
            download.fetch_format_files(_entry(ModelFormat.SAFETENSORS), tmp_path, None)
        fetch.assert_not_called()

    def test_a_fully_cached_set_probes_nothing(self, tmp_path: Path) -> None:
        names = ["model.safetensors", "config.json"]
        (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
        (tmp_path / "config.json").write_bytes(b"y" * 5)
        seen: list[tuple[int, int]] = []
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=names),
            mock.patch.object(download, "_cached_path", side_effect=lambda _r, n, d: d / n),
            mock.patch.object(download, "fetch_remote_file") as probe,
        ):
            path = download.fetch_format_files(
                _entry(ModelFormat.MLX),
                tmp_path,
                None,
                on_progress=lambda a, b: seen.append((a, b)),
            )
        assert path == tmp_path / "model.safetensors"
        assert seen == [(15, 15)]
        probe.assert_not_called()

    def test_the_space_check_counts_only_the_files_still_to_fetch(self, tmp_path: Path) -> None:
        names = ["model.safetensors", "config.json", "tokenizer.json"]
        sizes = dict(zip(names, (15, 5, 10), strict=True))
        cached = {"model.safetensors", "config.json"}  # 20 of 30 bytes are on disk
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=names),
            mock.patch.object(
                download,
                "_cached_path",
                side_effect=lambda _r, n, d: d / n if n in cached else None,
            ),
            mock.patch.object(
                download,
                "fetch_remote_file",
                side_effect=lambda _r, n: download.RemoteFile(sizes[n], n),
            ),
            mock.patch.object(download, "discard_partial_blobs"),
            mock.patch.object(download, "_free_bytes", return_value=15),
            mock.patch.object(
                download, "_download_with_retry", side_effect=lambda _e, c: tmp_path / c.filename
            ) as fetch,
            mock.patch.object(Path, "stat", return_value=mock.Mock(st_size=1)),
        ):
            download.fetch_format_files(_entry(ModelFormat.MLX), tmp_path, None)
        assert fetch.call_count == 3

    def test_the_space_check_still_refuses_when_the_missing_files_do_not_fit(
        self, tmp_path: Path
    ) -> None:
        names = ["model.safetensors", "config.json"]
        sizes = {"model.safetensors": 15, "config.json": 5}
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=names),
            mock.patch.object(
                download,
                "_cached_path",
                side_effect=lambda _r, n, d: d / n if n == "config.json" else None,
            ),
            mock.patch.object(
                download,
                "fetch_remote_file",
                side_effect=lambda _r, n: download.RemoteFile(sizes[n], n),
            ),
            mock.patch.object(download, "discard_partial_blobs"),
            mock.patch.object(download, "_free_bytes", return_value=10),
            pytest.raises(RuntimeError, match="Not enough disk space"),
        ):
            download.fetch_format_files(_entry(ModelFormat.MLX), tmp_path, None)

    def test_progress_reports_the_whole_set(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
        (tmp_path / "config.json").write_bytes(b"y" * 5)
        names = ["model.safetensors", "config.json"]
        seen: list[tuple[int, int]] = []
        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=names),
            mock.patch.object(
                download, "_fetch_files", return_value=[tmp_path / name for name in names]
            ),
        ):
            download.fetch_format_files(
                _entry(ModelFormat.MLX),
                tmp_path,
                None,
                on_progress=lambda a, b: seen.append((a, b)),
            )
        assert seen == [(15, 15)]


def _seed_snapshot(models_dir: Path, files: dict[str, bytes]) -> dict[str, Path]:
    cache = models_dir / "models--org--repo"
    (cache / "blobs").mkdir(parents=True)
    out: dict[str, Path] = {}
    for name, content in files.items():
        blob = cache / "blobs" / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        link = cache / "snapshots" / _REV / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        out[name] = link
    return out


class TestNonGgufManifest:
    def test_install_records_format_total_and_the_other_blobs(self, tmp_path: Path) -> None:
        files = {
            "model.safetensors": b"w" * 100,
            "config.json": b"{}",
            "tokenizer.json": b"{ }",
            "README.md": b"not selected",
        }
        paths = _seed_snapshot(tmp_path, files)
        with mock.patch.object(cfg, "models_dir", tmp_path):
            register_downloaded_model(_entry(ModelFormat.MLX), paths["model.safetensors"])
        (manifest,) = ModelRegistry(tmp_path).list_installed()
        assert manifest.format is ModelFormat.MLX
        assert manifest.gguf_filename == "model.safetensors"
        assert manifest.total_size_bytes == 100 + 2 + 3
        assert manifest.blob == hashlib.sha256(files["model.safetensors"]).hexdigest()
        assert sorted(manifest.shard_blobs) == sorted(
            hashlib.sha256(files[n]).hexdigest() for n in ("config.json", "tokenizer.json")
        )

    def test_gguf_install_still_records_gguf(self, tmp_path: Path) -> None:
        paths = _seed_snapshot(tmp_path, {"m-Q4_K_M.gguf": b"GGUF" + b"\x00" * 8})
        entry = _entry(ModelFormat.GGUF, "m-Q4_K_M.gguf")
        with mock.patch.object(cfg, "models_dir", tmp_path):
            register_downloaded_model(entry, paths["m-Q4_K_M.gguf"])
        (manifest,) = ModelRegistry(tmp_path).list_installed()
        assert (manifest.format, manifest.total_size_bytes) == (ModelFormat.GGUF, None)


class _PullingManager:
    """An installed manager whose pull runs the real file-set fetch over a recorded listing."""

    def __init__(self, tmp_path: Path, cached: set[str]) -> None:
        self.tmp_path = tmp_path
        self.cached = cached
        self.fetched: list[list[str]] = []

    def is_installed(self, _ref: str, _source: object = None) -> bool:
        return True

    def pull(self, _ref: str, _source: object, *, cancel: object = None) -> None:
        def fake_fetch(_entry, _dir, _token, files, _progress, _probe, _cached) -> list[Path]:
            self.fetched.append(files)
            return [self.tmp_path / name for name in files]

        with (
            mock.patch.object(download, "_repo_sibling_files", return_value=_DENSE_LISTING),
            mock.patch.object(
                download,
                "_cached_path",
                side_effect=lambda _r, name, _d: (
                    self.tmp_path / name if name in self.cached else None
                ),
            ),
            mock.patch.object(download, "_fetch_files", side_effect=fake_fetch),
            mock.patch.object(Path, "stat", return_value=mock.Mock(st_size=1)),
        ):
            download.fetch_format_files(_entry(ModelFormat.SAFETENSORS), self.tmp_path, None)


class TestPullOfAnInstalledModel:
    _REF = "org/repo#safetensors"

    def _pull(self, manager: _PullingManager):
        from lilbee.app import models as models_mod
        from lilbee.catalog.types import ModelSource

        with mock.patch.object(
            models_mod, "get_services", return_value=mock.Mock(model_manager=manager)
        ):
            return models_mod.pull_model_data(self._REF, ModelSource.NATIVE)

    def test_a_model_pulled_before_the_fix_gets_its_module_files(self, tmp_path: Path) -> None:
        old = {"model.safetensors", "config.json", "merges.txt", "tokenizer.json"}
        manager = _PullingManager(tmp_path, old | {"tokenizer_config.json", "vocab.json"})
        from lilbee.app.models import PullStatus

        assert self._pull(manager).status == PullStatus.ALREADY_INSTALLED
        assert len(manager.fetched) == 1
        assert {"modules.json", "2_Dense/model.safetensors"} <= set(manager.fetched[0])

    @staticmethod
    def _pull_offline(ref: str, error: Exception) -> tuple[str, int]:
        from lilbee.app import models as models_mod
        from lilbee.catalog.types import ModelSource

        manager = mock.Mock()
        manager.is_installed.return_value = True
        manager.pull.side_effect = error
        with (
            mock.patch.object(
                models_mod, "get_services", return_value=mock.Mock(model_manager=manager)
            ),
            mock.patch.object(models_mod, "_ensure_vision_projector"),
        ):
            result = models_mod.pull_model_data(ref, ModelSource.NATIVE)
        return result.status, manager.pull.call_count

    @pytest.mark.parametrize(
        ("ref", "calls"),
        [
            ("org/repo#safetensors", 1),
            ("org/repo#mlx", 0),
            ("org/repo#openvino", 0),
            ("org/repo/model.gguf", 0),
            ("org/repo", 0),
        ],
    )
    def test_every_ref_kind_stays_installed_offline(self, ref: str, calls: int) -> None:
        from lilbee.app.models import PullStatus

        error = RuntimeError("Cannot query files for org/repo: offline")
        assert self._pull_offline(ref, error) == (PullStatus.ALREADY_INSTALLED, calls)

    def test_a_safetensors_pull_without_an_engine_stays_installed(self) -> None:
        from lilbee.app.models import PullStatus
        from lilbee.providers.fleet.engines.choice import NoEngineError

        status, calls = self._pull_offline("org/repo#safetensors", NoEngineError("none"))
        assert (status, calls) == (PullStatus.ALREADY_INSTALLED, 1)

    def test_a_bug_in_the_fetch_still_surfaces(self) -> None:
        with pytest.raises(KeyError, match="bug"):
            self._pull_offline("org/repo#safetensors", KeyError("bug"))

    def test_a_complete_model_downloads_nothing(self, tmp_path: Path) -> None:
        manager = _PullingManager(tmp_path, set(_DENSE_LISTING))
        self._pull(manager)
        assert manager.fetched == []
