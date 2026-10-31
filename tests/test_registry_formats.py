"""Install, list, resolve, get_manifest and remove of a model by its ``org/repo#format`` ref."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

import pytest

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelSource, ModelTask
from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager import ModelManager
from lilbee.modelhub.registry import ModelManifest, ModelRegistry, register_downloaded_model
from lilbee.providers.fleet.engines import LLAMA

_REV = "0" * 40
_GGUF = "m-Q4_K_M.gguf"
_FILES = {
    _GGUF: b"GGUF" + b"\x00" * 8,
    "model.safetensors": b"w" * 100,
    "config.json": b"{}",
    "tokenizer.json": b"{ }",
}


def _digest(name: str) -> str:
    return hashlib.sha256(_FILES[name]).hexdigest()


def _entry(fmt: ModelFormat, filename: str) -> CatalogModel:
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """One repo cache holding a GGUF and the files of two non-GGUF formats; no download."""
    cache = tmp_path / "models--org--repo"
    (cache / "blobs").mkdir(parents=True)
    for name, content in _FILES.items():
        blob = cache / "blobs" / _digest(name)
        blob.write_bytes(content)
        link = cache / "snapshots" / _REV / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return tmp_path


def _install(models_dir: Path, fmt: ModelFormat) -> None:
    name = _GGUF if fmt == ModelFormat.GGUF else "model.safetensors"
    link = models_dir / "models--org--repo" / "snapshots" / _REV / name
    with mock.patch.object(cfg, "models_dir", models_dir):
        register_downloaded_model(_entry(fmt, name), link)


def _snapshot_names(models_dir: Path) -> set[str]:
    snapshot = models_dir / "models--org--repo" / "snapshots" / _REV
    return {p.name for p in snapshot.iterdir()}


def _blob_names(models_dir: Path) -> set[str]:
    return {p.name for p in (models_dir / "models--org--repo" / "blobs").iterdir()}


class TestResolveByFormatRef:
    def test_resolves_to_the_snapshot_directory(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        got = ModelRegistry(repo).resolve("org/repo#mlx")
        assert got == repo / "models--org--repo" / "snapshots" / _REV
        assert (got / "config.json").is_file()

    def test_get_manifest_by_ref(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        manifest = ModelRegistry(repo).get_manifest("org/repo#mlx")
        assert manifest is not None
        assert (manifest.format, manifest.ref) == (ModelFormat.MLX, "org/repo#mlx")

    def test_listing_names_the_ref_that_resolves(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        (manifest,) = registry.list_installed()
        assert registry.resolve(manifest.ref).is_dir()

    def test_is_installed_by_ref(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        assert registry.is_installed("org/repo#mlx")
        assert not registry.is_installed("org/repo#openvino")

    def test_a_format_that_is_not_installed_is_not_found(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        with pytest.raises(KeyError, match="org/repo#safetensors not installed"):
            registry.resolve("org/repo#safetensors")
        assert registry.get_manifest("org/repo#safetensors") is None

    def test_snapshot_links_gone_is_a_named_error(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        (repo / "models--org--repo" / "snapshots" / _REV / "model.safetensors").unlink()
        with pytest.raises(KeyError, match="org/repo#mlx"):
            ModelRegistry(repo).resolve("org/repo#mlx")

    def test_an_unknown_format_is_refused_by_name(self, repo: Path) -> None:
        registry = ModelRegistry(repo)
        with pytest.raises(ValueError, match="'bogus'"):
            registry.resolve("org/repo#bogus")
        assert registry.get_manifest("org/repo#bogus") is None
        assert registry.remove("org/repo#bogus") is False


class TestTwoFormatsOfOneRepo:
    def test_gguf_and_mlx_are_two_models(self, repo: Path) -> None:
        _install(repo, ModelFormat.GGUF)
        _install(repo, ModelFormat.MLX)
        refs = sorted(m.ref for m in ModelRegistry(repo).list_installed())
        assert refs == ["org/repo#mlx", f"org/repo/{_GGUF}"]

    def test_mlx_and_safetensors_are_two_models(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        _install(repo, ModelFormat.SAFETENSORS)
        refs = sorted(m.ref for m in ModelRegistry(repo).list_installed())
        assert refs == ["org/repo#mlx", "org/repo#safetensors"]

    def test_a_bare_repo_still_reads_the_gguf(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        _install(repo, ModelFormat.GGUF)
        registry = ModelRegistry(repo)
        assert registry.resolve("org/repo").name == _digest(_GGUF)
        assert registry.installed_ref_for_repo("org/repo") == f"org/repo/{_GGUF}"

    def test_a_bare_repo_does_not_resolve_to_a_non_gguf_model(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        assert registry.installed_ref_for_repo("org/repo") is None
        with (
            mock.patch.object(ModelRegistry, "_cached_gguf_names", return_value=set()),
            pytest.raises(KeyError, match="org/repo not installed"),
        ):
            registry.resolve("org/repo")


@pytest.fixture
def copy_repo(tmp_path: Path) -> Path:
    """The repo cache of a host with no symlinks: snapshot entries are real files."""
    cache = tmp_path / "models--org--repo"
    (cache / "blobs").mkdir(parents=True)
    for name, content in _FILES.items():
        snapshot_file = cache / "snapshots" / _REV / name
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file.write_bytes(content)
    return tmp_path


class TestRemoveWithoutSymlinks:
    def test_the_removed_models_files_go_and_the_siblings_stay(self, copy_repo: Path) -> None:
        _install(copy_repo, ModelFormat.GGUF)
        _install(copy_repo, ModelFormat.MLX)
        assert ModelRegistry(copy_repo).remove("org/repo#mlx") is True
        assert _snapshot_names(copy_repo) == {_GGUF}

    def test_a_file_a_remaining_model_names_stays(self, copy_repo: Path) -> None:
        _install(copy_repo, ModelFormat.MLX)
        _install(copy_repo, ModelFormat.SAFETENSORS)
        assert ModelRegistry(copy_repo).remove("org/repo#mlx") is True
        kept = _snapshot_names(copy_repo)
        assert {"model.safetensors", "config.json", "tokenizer.json"} <= kept

    def test_removing_the_gguf_keeps_the_other_formats_files(self, copy_repo: Path) -> None:
        _install(copy_repo, ModelFormat.GGUF)
        _install(copy_repo, ModelFormat.MLX)
        assert ModelRegistry(copy_repo).remove(f"org/repo/{_GGUF}") is True
        assert _snapshot_names(copy_repo) == {"model.safetensors", "config.json", "tokenizer.json"}

    def test_a_sibling_whose_files_cannot_be_named_keeps_every_file(self, copy_repo: Path) -> None:
        _install(copy_repo, ModelFormat.MLX)
        _install(copy_repo, ModelFormat.SAFETENSORS)
        with mock.patch(
            "lilbee.modelhub.registry.select_files", side_effect=ValueError("unusable")
        ):
            assert ModelRegistry(copy_repo).remove("org/repo#mlx") is True
        assert _snapshot_names(copy_repo) == set(_FILES)


class TestRemoveByFormatRef:
    def test_removing_mlx_keeps_the_gguf(self, repo: Path) -> None:
        _install(repo, ModelFormat.GGUF)
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        assert registry.remove("org/repo#mlx") is True
        assert [m.ref for m in registry.list_installed()] == [f"org/repo/{_GGUF}"]
        assert registry.get_manifest("org/repo#mlx") is None
        assert registry.resolve(f"org/repo/{_GGUF}").name == _digest(_GGUF)
        assert _blob_names(repo) == {_digest(_GGUF)}
        assert _snapshot_names(repo) == {_GGUF}

    def test_removing_the_gguf_keeps_mlx(self, repo: Path) -> None:
        _install(repo, ModelFormat.GGUF)
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        assert registry.remove(f"org/repo/{_GGUF}") is True
        assert [m.ref for m in registry.list_installed()] == ["org/repo#mlx"]
        assert _blob_names(repo) == {
            _digest(n) for n in ("model.safetensors", "config.json", "tokenizer.json")
        }
        assert (registry.resolve("org/repo#mlx") / "model.safetensors").is_file()

    def test_removing_mlx_keeps_the_blobs_safetensors_uses(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        _install(repo, ModelFormat.SAFETENSORS)
        registry = ModelRegistry(repo)
        assert registry.remove("org/repo#mlx") is True
        assert [m.ref for m in registry.list_installed()] == ["org/repo#safetensors"]
        loaded = registry.resolve("org/repo#safetensors")
        assert {p.name for p in loaded.iterdir() if p.exists()} >= {
            "model.safetensors",
            "config.json",
            "tokenizer.json",
        }
        assert (loaded / "model.safetensors").read_bytes() == _FILES["model.safetensors"]

    def test_removing_the_last_model_leaves_no_cache(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        registry = ModelRegistry(repo)
        assert registry.remove("org/repo#mlx") is True
        assert not (repo / "models--org--repo").exists()
        assert registry.list_installed() == []

    def test_removing_a_model_whose_snapshot_folder_is_gone_still_frees_it(
        self, repo: Path
    ) -> None:
        _install(repo, ModelFormat.GGUF)
        _install(repo, ModelFormat.MLX)
        snapshot = repo / "models--org--repo" / "snapshots" / _REV
        for link in snapshot.iterdir():
            link.unlink()
        snapshot.rmdir()
        snapshot.parent.rmdir()
        registry = ModelRegistry(repo)
        assert registry.remove("org/repo#mlx") is True
        assert _blob_names(repo) == {_digest(_GGUF)}

    def test_removing_a_format_that_is_not_installed_is_false(self, repo: Path) -> None:
        _install(repo, ModelFormat.GGUF)
        registry = ModelRegistry(repo)
        assert registry.remove("org/repo#mlx") is False
        assert [m.ref for m in registry.list_installed()] == [f"org/repo/{_GGUF}"]


class TestManifestPath:
    def test_each_format_has_its_own_manifest_file(self, repo: Path) -> None:
        _install(repo, ModelFormat.MLX)
        _install(repo, ModelFormat.SAFETENSORS)
        names = sorted(p.name for p in (repo / "manifests" / "org--repo").iterdir())
        assert names == ["mlx.json", "safetensors.json"]

    def test_an_entry_file_that_escapes_the_repo_is_refused(self, repo: Path) -> None:
        registry = ModelRegistry(repo)
        bad = _entry(ModelFormat.MLX, "../model.safetensors")
        with pytest.raises(ValueError, match="Invalid gguf_filename"):
            registry.install(
                bad.hf_repo,
                bad.gguf_filename,
                repo / "models--org--repo" / "snapshots" / _REV / "model.safetensors",
                _manifest_for(bad),
            )


def _manifest_for(entry: CatalogModel) -> ModelManifest:
    return ModelManifest(
        hf_repo=entry.hf_repo,
        gguf_filename=entry.gguf_filename,
        size_bytes=1,
        task=entry.task,
        downloaded_at="2026-01-01T00:00:00+00:00",
        format=entry.format,
    )


class TestThroughTheModelManager:
    def test_install_list_find_and_remove_by_ref(self, repo: Path) -> None:
        _install(repo, ModelFormat.GGUF)
        _install(repo, ModelFormat.MLX)
        manager = ModelManager(repo, engine=LLAMA)
        assert manager.list_installed(ModelSource.NATIVE) == [
            "org/repo#mlx",
            f"org/repo/{_GGUF}",
        ]
        assert manager.is_installed("org/repo#mlx", ModelSource.NATIVE)
        assert manager.get_source("org/repo#mlx") is ModelSource.NATIVE
        assert manager.remove("org/repo#mlx") is True
        assert manager.list_installed(ModelSource.NATIVE) == [f"org/repo/{_GGUF}"]
        assert not manager.is_installed("org/repo#mlx", ModelSource.NATIVE)
