"""Tests for registry.py -- content-addressable model storage."""

import hashlib
import json
from pathlib import Path

import pytest

from lilbee.registry import (
    ModelManifest,
    ModelRef,
    ModelRegistry,
    _coerce_ref,
    _match_catalog_entry,
    _sha256_file,
    _validate_ref_segment,
)


def _create_hf_cache_structure(models_dir: Path, source_repo: str, content: bytes) -> Path:
    """Create a file in HF cache structure and return the path."""
    safe_repo = source_repo.replace("/", "--")
    cache_path = models_dir / f"models--{safe_repo}"
    blobs_dir = cache_path / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    blob_path = blobs_dir / digest
    blob_path.write_bytes(content)
    return blob_path


class TestModelRef:
    def test_parse_name_only(self) -> None:
        ref = ModelRef.parse("nomic-embed-text")
        assert ref.name == "nomic-embed-text"
        assert ref.tag == "latest"

    def test_parse_name_and_tag(self) -> None:
        ref = ModelRef.parse("qwen3:8b")
        assert ref.name == "qwen3"
        assert ref.tag == "8b"

    def test_parse_name_with_multiple_colons(self) -> None:
        ref = ModelRef.parse("org/repo:v1.0")
        assert ref.name == "org/repo"
        assert ref.tag == "v1.0"

    def test_str_representation(self) -> None:
        ref = ModelRef(name="qwen3", tag="8b")
        assert str(ref) == "qwen3:8b"

    def test_str_latest(self) -> None:
        ref = ModelRef(name="nomic-embed-text")
        assert str(ref) == "nomic-embed-text:latest"

    def test_frozen(self) -> None:
        ref = ModelRef(name="test")
        with pytest.raises(AttributeError):
            ref.name = "other"  # type: ignore[misc]

    def test_default_tag(self) -> None:
        ref = ModelRef(name="test")
        assert ref.tag == "latest"

    def test_parse_rejects_path_traversal_in_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            ModelRef.parse("../etc/passwd")

    def test_parse_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            ModelRef.parse("")

    def test_parse_rejects_path_traversal_in_tag(self) -> None:
        with pytest.raises(ValueError, match="Invalid model tag"):
            ModelRef.parse("model:../../evil")

    def test_parse_allows_namespaced_models(self) -> None:
        ref = ModelRef.parse("org/repo:v1.0")
        assert ref.name == "org/repo"
        assert ref.tag == "v1.0"


class TestValidateRefSegment:
    def test_valid_segment(self) -> None:
        assert _validate_ref_segment("qwen3-8B", "name") == "qwen3-8B"

    def test_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            _validate_ref_segment("..", "name")

    def test_allows_spaces(self) -> None:
        assert _validate_ref_segment("Nomic Embed Text v1.5", "name") == "Nomic Embed Text v1.5"

    def test_rejects_special_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid model tag"):
            _validate_ref_segment("bad;tag", "tag")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            _validate_ref_segment("", "name")


class TestCoerceRef:
    def test_string_input(self) -> None:
        ref = _coerce_ref("qwen3:8b")
        assert ref == ModelRef(name="qwen3", tag="8b")

    def test_model_ref_passthrough(self) -> None:
        original = ModelRef(name="test", tag="v1")
        assert _coerce_ref(original) is original


class TestSha256File:
    def test_computes_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _sha256_file(p) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(p) == expected


def _make_manifest(
    name: str = "test-model",
    tag: str = "latest",
    size_bytes: int = 1000,
    task: str = "chat",
    source_repo: str = "org/repo",
    source_filename: str = "model.gguf",
    downloaded_at: str = "2026-01-01T00:00:00+00:00",
    blob: str = "abc123",
) -> ModelManifest:
    return ModelManifest(
        name=name,
        tag=tag,
        size_bytes=size_bytes,
        task=task,
        source_repo=source_repo,
        source_filename=source_filename,
        downloaded_at=downloaded_at,
        blob=blob,
    )


class TestModelRegistry:
    def test_resolve_not_installed(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve("missing-model")

    def test_resolve_missing_blob(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)
        manifest = _make_manifest(source_repo="org/repo", blob="deadbeef")
        ref = ModelRef(name="test-model")
        registry._write_manifest(ref, manifest)
        # Create cache folder but not the blob file
        cache_path = models_dir / "models--org--repo"
        (cache_path / "blobs").mkdir(parents=True)
        with pytest.raises(KeyError, match="Blob file missing"):
            registry.resolve(ref)

    def test_resolve_missing_cache_folder(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)
        manifest = _make_manifest(source_repo="org/repo", blob="deadbeef")
        ref = ModelRef(name="test-model")
        registry._write_manifest(ref, manifest)
        with pytest.raises(KeyError, match="Cache folder missing"):
            registry.resolve(ref)

    def test_resolve_missing_blobs_dir(self, tmp_path: Path) -> None:
        """Test when cache folder exists but blobs/ subdir is missing."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        cache_path = models_dir / "models--org--repo"
        cache_path.mkdir(parents=True)

        manifest = _make_manifest(source_repo="org/repo", blob="deadbeef")
        ref = ModelRef(name="test-model")
        registry._write_manifest(ref, manifest)

        with pytest.raises(KeyError, match="Blobs directory missing"):
            registry.resolve(ref)

    def test_install_and_resolve_roundtrip(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"fake model data"
        blob_path = _create_hf_cache_structure(models_dir, "org/repo", content)

        ref = ModelRef(name="test-model", tag="latest")
        manifest = _make_manifest(
            source_repo="org/repo",
            size_bytes=len(content),
            source_filename="model.gguf",
        )

        result = registry.install(ref, blob_path, manifest)

        assert result == blob_path
        assert blob_path.exists()

        resolved = registry.resolve(ref)
        assert resolved == blob_path

    def test_install_copies_source_to_cache(self, tmp_path: Path) -> None:
        """install() copies source file into HF cache when not already present."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        source = tmp_path / "external.gguf"
        source.write_bytes(b"data")

        ref = ModelRef(name="test-model")
        manifest = _make_manifest(source_repo="org/repo")

        blob_path = registry.install(ref, source, manifest)
        assert blob_path.exists()
        assert registry.is_installed(ref)

    def test_install_same_content_creates_same_blob(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"same content"
        blob_path = _create_hf_cache_structure(models_dir, "org/repo-a", content)

        ref1 = ModelRef(name="model-a", tag="latest")
        ref2 = ModelRef(name="model-b", tag="latest")
        manifest1 = _make_manifest(name="model-a", source_repo="org/repo-a")
        manifest2 = _make_manifest(name="model-b", source_repo="org/repo-a")

        result1 = registry.install(ref1, blob_path, manifest1)
        result2 = registry.install(ref2, blob_path, manifest2)

        assert result1 == result2  # Same blob for same content

    def test_is_installed_true(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"data"
        blob_path = _create_hf_cache_structure(models_dir, "org/repo", content)
        ref = ModelRef(name="test")
        registry.install(ref, blob_path, _make_manifest(source_repo="org/repo"))

        assert registry.is_installed("test") is True
        assert registry.is_installed("test:latest") is True

    def test_is_installed_false(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.is_installed("nonexistent") is False

    def test_is_installed_invalid_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.is_installed("") is False

    def test_remove_invalid_ref_returns_false(self, tmp_path: Path) -> None:
        """remove() returns False when given an invalid ref that raises ValueError."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        # Empty string ref triggers ValueError in _coerce_ref
        assert registry.remove("") is False

    def test_blob_referenced(self, tmp_path: Path) -> None:
        """_blob_referenced checks if any manifest references a given digest."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"shared-data"
        blob_path = _create_hf_cache_structure(models_dir, "org/shared", content)
        digest = blob_path.name

        ref = ModelRef(name="model-a")
        manifest = _make_manifest(name="model-a", source_repo="org/shared", blob=digest)
        registry.install(ref, blob_path, manifest)

        assert registry._blob_referenced(digest) is True
        assert registry._blob_referenced("nonexistent-hash") is False

    def test_remove_deletes_manifest_keeps_cache(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"unique data"
        blob_path = _create_hf_cache_structure(models_dir, "org/repo", content)
        ref = ModelRef(name="removeme")
        manifest = _make_manifest(name="removeme", source_repo="org/repo")
        registry.install(ref, blob_path, manifest)

        assert registry.remove("removeme") is True
        assert blob_path.exists()
        assert registry.is_installed("removeme") is False

    def test_remove_multiple_manifests_keeps_cache(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"shared blob data"
        blob_path = _create_hf_cache_structure(models_dir, "org/repo-shared", content)

        ref1 = ModelRef(name="model-a")
        ref2 = ModelRef(name="model-b")
        m1 = _make_manifest(name="model-a", source_repo="org/repo-shared")
        m2 = _make_manifest(name="model-b", source_repo="org/repo-shared")
        registry.install(ref1, blob_path, m1)
        registry.install(ref2, blob_path, m2)

        assert registry.remove("model-a") is True
        assert blob_path.exists()  # still exists, still referenced by model-b
        assert registry.is_installed("model-b") is True

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.remove("ghost") is False

    def test_list_installed_empty(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.list_installed() == []

    def test_list_installed_multiple(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        for name, repo in (("alpha", "org/alpha"), ("beta", "org/beta")):
            content = f"data-{name}".encode()
            blob_path = _create_hf_cache_structure(models_dir, repo, content)
            ref = ModelRef(name=name)
            registry.install(ref, blob_path, _make_manifest(name=name, source_repo=repo))

        installed = registry.list_installed()
        names = [m.name for m in installed]
        assert "alpha" in names
        assert "beta" in names
        assert len(installed) == 2

    def test_list_installed_skips_non_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"data"
        blob_path = _create_hf_cache_structure(models_dir, "org/real", content)
        m = _make_manifest(name="real", source_repo="org/real")
        registry.install(ModelRef(name="real"), blob_path, m)

        (models_dir / "manifests" / "stray.txt").write_text("junk")

        installed = registry.list_installed()
        assert len(installed) == 1
        assert installed[0].name == "real"

    def test_tag_support(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content_latest = b"latest-data"
        content_v2 = b"v2-data"

        blob_latest = _create_hf_cache_structure(models_dir, "org/qwen3", content_latest)
        blob_v2 = _create_hf_cache_structure(models_dir, "org/qwen3-v2", content_v2)

        ref_latest = ModelRef(name="qwen3", tag="latest")
        ref_v2 = ModelRef(name="qwen3", tag="0.6b")

        m1 = _make_manifest(name="qwen3", tag="latest", source_repo="org/qwen3")
        m2 = _make_manifest(name="qwen3", tag="0.6b", source_repo="org/qwen3-v2")
        registry.install(ref_latest, blob_latest, m1)
        registry.install(ref_v2, blob_v2, m2)

        assert registry.is_installed("qwen3:latest")
        assert registry.is_installed("qwen3:0.6b")

        path_latest = registry.resolve("qwen3:latest")
        path_v2 = registry.resolve("qwen3:0.6b")
        assert path_latest != path_v2

    def test_resolve_with_string(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"content"
        blob_path = _create_hf_cache_structure(models_dir, "org/mymodel", content)
        m = _make_manifest(name="mymodel", source_repo="org/mymodel")
        registry.install(ModelRef(name="mymodel"), blob_path, m)

        path = registry.resolve("mymodel:latest")
        assert path.exists()

    def test_corrupt_manifest_skipped(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        manifest_dir = models_dir / "manifests" / "broken"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "latest.json").write_text("not valid json {{{")

        assert registry.list_installed() == []

    def test_manifest_path_traversal_blocked(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        # Directly constructed ref with traversal (bypassing parse validation)
        evil_ref = ModelRef.__new__(ModelRef)
        object.__setattr__(evil_ref, "name", "../../etc")
        object.__setattr__(evil_ref, "tag", "passwd")
        with pytest.raises(ValueError, match="Path escapes"):
            registry._manifest_path(evil_ref)

    def test_corrupt_manifest_missing_fields(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        manifest_dir = models_dir / "manifests" / "broken"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "latest.json").write_text(json.dumps({"name": "broken"}))

        assert registry.list_installed() == []

    def test_remove_cleans_empty_name_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)

        content = b"data"
        blob_path = _create_hf_cache_structure(models_dir, "org/cleanup", content)
        ref = ModelRef(name="cleanup-test")
        manifest = _make_manifest(name="cleanup-test", source_repo="org/cleanup")
        registry.install(ref, blob_path, manifest)

        name_dir = models_dir / "manifests" / "cleanup-test"
        assert name_dir.exists()

        registry.remove("cleanup-test")
        assert not name_dir.exists()


class TestMigrateLegacy:
    def test_migrates_gguf_files(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "some-model.gguf").write_bytes(b"model-bytes")

        registry = ModelRegistry(models_dir)
        count = registry.migrate_legacy()

        assert count == 1
        assert not (models_dir / "some-model.gguf").exists()  # moved to cache
        installed = registry.list_installed()
        assert len(installed) == 1
        assert installed[0].source_filename == "some-model.gguf"

    def test_migrates_to_hf_cache_structure(self, tmp_path: Path) -> None:
        """Verifies migrated files go to models--NAME/blobs/ structure (no repo)."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "some-model.gguf").write_bytes(b"model-bytes")

        registry = ModelRegistry(models_dir)
        count = registry.migrate_legacy()

        assert count == 1
        # When repo is empty, uses models--NAME format (keeps hyphens)
        cache_dir = models_dir / "models--some-model"
        assert cache_dir.exists()
        blobs_dir = cache_dir / "blobs"
        assert blobs_dir.exists()
        blob_files = list(blobs_dir.glob("*"))  # HF cache uses raw hash, no prefix
        assert len(blob_files) == 1

    def test_migrates_known_catalog_model(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "nomic-embed-text-v1.5-GGUF-Q4_K_M.gguf").write_bytes(b"embed-bytes")

        registry = ModelRegistry(models_dir)
        count = registry.migrate_legacy()

        assert count == 1
        installed = registry.list_installed()
        # 2 manifests: v1.5 + latest alias (recommended model)
        assert len(installed) == 2
        tags = {m.tag for m in installed}
        assert tags == {"v1.5", "latest"}
        assert all(m.task == "embedding" for m in installed)
        assert all(m.source_repo == "nomic-ai/nomic-embed-text-v1.5-GGUF" for m in installed)

    def test_no_gguf_files(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "readme.txt").write_text("nothing here")

        registry = ModelRegistry(models_dir)
        assert registry.migrate_legacy() == 0

    def test_missing_root_dir(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "nonexistent")
        assert registry.migrate_legacy() == 0

    def test_migrates_multiple(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "a.gguf").write_bytes(b"aaa")
        (models_dir / "b.gguf").write_bytes(b"bbb")

        registry = ModelRegistry(models_dir)
        assert registry.migrate_legacy() == 2
        assert len(registry.list_installed()) == 2


class TestMatchCatalogEntry:
    def test_matches_featured_embedding(self) -> None:
        name, tag, task, repo = _match_catalog_entry("nomic-embed-text-v1.5-GGUF-Q4_K_M.gguf")
        assert name == "nomic-embed-text"
        assert tag == "v1.5"
        assert task == "embedding"
        assert "nomic" in repo.lower()

    def test_matches_featured_chat(self) -> None:
        name, tag, task, repo = _match_catalog_entry("Qwen3-8B-GGUF-Q4_K_M.gguf")
        assert name == "qwen3"
        assert tag == "8b"
        assert task == "chat"
        assert "Qwen" in repo

    def test_fallback_unknown_model(self) -> None:
        name, tag, task, repo = _match_catalog_entry("totally-custom-model.gguf")
        assert name == "totally-custom-model"
        assert tag == "latest"
        assert task == "chat"
        assert repo == ""


class TestWriteLatestAlias:
    def test_noop_for_missing_manifest(self, tmp_path: Path) -> None:
        """write_latest_alias does nothing when the ref has no manifest."""
        registry = ModelRegistry(tmp_path / "models")
        ref = ModelRef(name="nonexistent", tag="v1")
        registry.write_latest_alias(ref)
        assert not (tmp_path / "models" / "manifests").exists()


class TestWriteManifestErrorPath:
    def test_temp_file_cleaned_up_on_replace_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When os.replace raises, the temp file is cleaned up."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)
        ref = ModelRef(name="fail-model")
        manifest = _make_manifest(name="fail-model")

        def failing_replace(src: str, dst: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", failing_replace)

        with pytest.raises(OSError, match="disk full"):
            registry._write_manifest(ref, manifest)

        # Verify no temp files left behind
        manifest_dir = models_dir / "manifests" / "fail-model"
        if manifest_dir.exists():
            tmp_files = list(manifest_dir.glob("*.tmp"))
            assert tmp_files == []
