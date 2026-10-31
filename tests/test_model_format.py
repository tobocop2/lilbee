"""Model format on references and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lilbee.catalog.refs import (
    ModelRef,
    is_gguf_filename,
    is_native_gguf_shape,
    model_format,
    parse_model_ref,
)
from lilbee.catalog.types import ModelFormat
from lilbee.data.store.lance_helpers import refs_compatible
from lilbee.modelhub.registry import ModelManifest, ModelRegistry, parse_hf_ref
from lilbee.providers.model_ref import (
    parse_model_ref as parse_provider_ref,
)
from lilbee.providers.model_ref import routes_to_native_gguf, routes_to_native_model
from lilbee.providers.server_registry import BUILTIN_PREFIXES, LOCAL_PROVIDER


class TestParseModelRef:
    def test_native_gguf_ref_keeps_its_string(self) -> None:
        parsed = parse_model_ref("org/repo/Q4_K_M/m-00001-of-00002.gguf")
        assert parsed == ModelRef("org/repo", "Q4_K_M/m-00001-of-00002.gguf", ModelFormat.GGUF)
        assert parsed.canonical == "org/repo/Q4_K_M/m-00001-of-00002.gguf"

    def test_bare_repo_reads_gguf(self) -> None:
        parsed = parse_model_ref("org/repo")
        assert parsed == ModelRef("org/repo", "", ModelFormat.GGUF)
        assert parsed.canonical == "org/repo"

    @pytest.mark.parametrize(
        "fmt", [ModelFormat.MLX, ModelFormat.OPENVINO, ModelFormat.SAFETENSORS]
    )
    def test_qualified_ref_reads_its_format(self, fmt: ModelFormat) -> None:
        parsed = parse_model_ref(f"org/repo.gguf#{fmt}")
        assert parsed == ModelRef("org/repo.gguf", "", fmt)
        assert parsed.canonical == f"org/repo.gguf#{fmt}"

    def test_gguf_qualifier_canonicalizes_to_the_plain_string(self) -> None:
        assert parse_model_ref("o/r/f.gguf#gguf").canonical == "o/r/f.gguf"

    @pytest.mark.parametrize(
        ("raw", "named"),
        [("o/r/f.gguf#mlx", "o/r/f.gguf#mlx"), ("o/r#nosuch", "nosuch"), ("o/r#", "''")],
    )
    def test_refuses_a_bad_qualifier_naming_the_value(self, raw: str, named: str) -> None:
        with pytest.raises(ValueError, match=r"Model ref .*#") as info:
            parse_model_ref(raw)
        assert named in str(info.value)

    def test_provider_ref_reads_gguf_with_the_whole_string_as_repo(self) -> None:
        assert parse_model_ref("ollama/llama3:8b") == ModelRef(
            "ollama/llama3:8b", "", ModelFormat.GGUF
        )


class TestModelFormat:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("o/r/f.gguf", ModelFormat.GGUF),
            ("o/r", ModelFormat.GGUF),
            ("o/r#mlx", ModelFormat.MLX),
            ("o/r#nosuch", ModelFormat.GGUF),
            ("", ModelFormat.GGUF),
        ],
    )
    def test_total_over_any_string(self, raw: str, expected: ModelFormat) -> None:
        assert model_format(raw) is expected


class TestSuffixHelpers:
    def test_filename_suffix_is_case_sensitive(self) -> None:
        assert is_gguf_filename("m.gguf") is True
        assert is_gguf_filename("m.GGUF") is False

    def test_native_shape_needs_two_slashes(self) -> None:
        assert is_native_gguf_shape("o/r/m.gguf") is True
        assert is_native_gguf_shape("o/m.gguf") is False


class TestRouting:
    @pytest.mark.parametrize("fmt", ["mlx", "openvino", "safetensors"])
    def test_every_format_routes_native(self, fmt: str) -> None:
        assert routes_to_native_model(f"org/repo#{fmt}") is True
        assert routes_to_native_gguf(f"org/repo#{fmt}") is False

    def test_gguf_ref_routes_to_both(self) -> None:
        assert routes_to_native_model("org/repo/f.gguf") is True
        assert routes_to_native_gguf("org/repo/f.gguf") is True

    @pytest.mark.parametrize("prefix", sorted(BUILTIN_PREFIXES))
    def test_every_provider_prefix_wins_over_the_format(self, prefix: str) -> None:
        ref = parse_provider_ref(f"{prefix}/x#mlx")
        assert ref.provider != LOCAL_PROVIDER
        assert routes_to_native_model(f"{prefix}/x#mlx") is False

    def test_a_format_ref_under_an_org_that_is_no_provider_classifies_local(self) -> None:
        assert parse_provider_ref("someorg/gpt-oss-20b#mlx").provider == LOCAL_PROVIDER

    def test_a_server_preset_prefix_wins_over_the_format(self) -> None:
        route = SimpleNamespace(name="mybox")
        with patch(
            "lilbee.providers.model_ref.route_for",
            side_effect=lambda p: route if p == "mybox" else None,
        ):
            assert parse_provider_ref("mybox/x#mlx").provider == "mybox"
            assert routes_to_native_model("mybox/x#mlx") is True
            assert routes_to_native_model("mybox/x#mlx", read_presets=True) is False

    def test_server_prefix_wins_over_the_format(self) -> None:
        assert routes_to_native_model("ollama/llama3#mlx") is False

    def test_malformed_qualifier_is_not_native(self) -> None:
        assert routes_to_native_model("org/repo#nosuch") is False


_OLD = {
    "hf_repo": "org/repo",
    "gguf_filename": "m.gguf",
    "size_bytes": 5,
    "task": "chat",
    "downloaded_at": "2026-01-01T00:00:00+00:00",
    "blob": "ab" * 32,
    "total_size_bytes": None,
    "shard_blobs": [],
}


class TestManifestFormat:
    def test_an_old_manifest_file_loads_as_gguf_with_its_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        path = registry._manifest_path("org/repo", "m.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_OLD), encoding="utf-8")

        loaded = registry._load_manifest_file(path)

        assert loaded is not None
        assert loaded.format is ModelFormat.GGUF
        assert loaded.ref == "org/repo/m.gguf"

    def test_a_stored_format_round_trips_and_sets_the_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        path = registry._manifest_path("org/repo", "m.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**_OLD, "format": "mlx"}),
            encoding="utf-8",
        )

        loaded = registry._load_manifest_file(path)

        assert loaded is not None
        assert loaded.format is ModelFormat.MLX
        assert loaded.ref == "org/repo#mlx"

    def test_an_unknown_format_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nosuch"):
            ModelManifest(**{**_OLD, "format": "nosuch"})

    def test_a_gguf_manifest_saves_without_a_format_key(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        manifest = ModelManifest(**_OLD)
        registry._write_manifest(manifest)

        saved = json.loads(
            registry._manifest_path("org/repo", "m.gguf").read_text(encoding="utf-8")
        )

        assert saved == _OLD

    def test_a_non_gguf_manifest_saves_its_format_key(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry._write_manifest(ModelManifest(**{**_OLD, "format": "mlx"}))

        saved = json.loads(
            registry._manifest_path("org/repo", "m.gguf").read_text(encoding="utf-8")
        )

        assert saved["format"] == "mlx"

    def test_a_manifest_with_an_unknown_format_reads_as_corrupt(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        path = registry._manifest_path("org/repo", "m.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**_OLD, "format": "nosuch"}), encoding="utf-8")

        assert registry._load_manifest_file(path) is None

    def test_parse_hf_ref_refuses_a_format_qualified_ref(self) -> None:
        with pytest.raises(ValueError, match="not a HuggingFace ref"):
            parse_hf_ref("org/repo#mlx")


class TestStoredRefCompatibility:
    @pytest.mark.parametrize(
        ("persisted", "current"),
        [
            ("/Users/me/c#/m.gguf", "org/repo/f.gguf"),
            ("org/repo", "/Users/me/c#/m.gguf"),
            ("/Users/me/c#/m", "org/repo/f.gguf"),
            ("org/repo", "/Users/me/c#nosuch"),
            ("a#b#c", "x#y"),
        ],
    )
    def test_a_ref_with_a_hash_answers_and_does_not_raise(
        self, persisted: str, current: str
    ) -> None:
        assert refs_compatible(persisted, current, 8, 8) is False

    def test_a_repo_named_like_a_gguf_file_is_not_a_bare_repo(self) -> None:
        assert refs_compatible("a/b.gguf", "a/b.gguf/c.gguf", 8, 8) is False

    def test_bare_repo_matches_its_gguf_ref(self) -> None:
        assert refs_compatible("org/repo", "org/repo/f.gguf", 8, 8) is True

    def test_two_gguf_files_are_not_lumped_together(self) -> None:
        assert refs_compatible("org/repo/a.gguf", "org/repo/b.gguf", 8, 8) is False

    def test_a_format_ref_does_not_match_a_gguf_ref_of_the_repo(self) -> None:
        assert refs_compatible("org/repo#mlx", "org/repo/f.gguf", 8, 8) is False
        assert refs_compatible("org/repo", "org/repo#mlx", 8, 8) is False

    def test_the_same_format_ref_matches_itself(self) -> None:
        assert refs_compatible("org/repo#mlx", "org/repo#mlx", 8, 8) is True


class TestInstallKeepsTheFormat:
    def test_install_writes_the_format_of_the_manifest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        source = tmp_path / "source.gguf"
        source.write_bytes(b"GGUF\x00\x00")
        manifest = ModelManifest(**{**_OLD, "format": "mlx", "size_bytes": 6})

        registry.install("org/repo", "m.gguf", source, manifest)

        written = json.loads(
            registry._manifest_path("org/repo", "m.gguf").read_text(encoding="utf-8")
        )
        assert written["format"] == "mlx"
