"""Regression tests for the per-backend engine-wheel index generator.

The GH release build-tags EVERY engine wheel (``-1.<backend>``); the index must
link to those exact filenames. An earlier version exempted vulkan/metal, whose
indexes then pointed at untagged filenames that were never uploaded -- a 404 for
every Mac (metal) and Vulkan user installing the engine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_pep503_indexes.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_pep503_indexes", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bpi = _load()

_RAW = "lilbee_engine-0.6.90b420.dev721-py3-none-macosx_11_0_arm64.whl"


def test_every_backend_gets_a_build_tag() -> None:
    # No backend is exempt: the release tags them all (the index links those
    # tagged release assets), so build_tag_for_backend never returns None.
    for backend in ("metal", "vulkan", "cu124", "cu125", "cu121", "cpu", "rocm"):
        assert bpi.build_tag_for_backend(backend) == f"1.{backend}"


def test_metal_and_vulkan_are_not_treated_as_untagged_defaults() -> None:
    # The exact regression: these two must NOT keep the plain filename.
    for backend in ("metal", "vulkan"):
        renamed = bpi.rename_for_release(_RAW, backend)
        assert f"-1.{backend}-" in renamed, renamed
        assert renamed != _RAW


def test_rename_matches_release_asset_layout() -> None:
    assert (
        bpi.rename_for_release(_RAW, "metal")
        == "lilbee_engine-0.6.90b420.dev721-1.metal-py3-none-macosx_11_0_arm64.whl"
    )


def test_index_href_targets_the_tagged_asset(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    (art / "wheel-multigpu-macos-metal").mkdir(parents=True)
    (art / "wheel-multigpu-macos-metal" / _RAW).write_bytes(b"x")
    site = tmp_path / "site"
    bpi.write_backend_indexes(
        site,
        bpi.collect_wheels(art),
        "https://github.com/tobocop2/lilbee/releases/download",
    )
    index = (site / "metal" / "lilbee-engine" / "index.html").read_text()
    assert "dev721-1.metal-py3-none-macosx_11_0_arm64.whl" in index
    # the plain (never-uploaded) name must not appear
    assert "dev721-py3-none-macosx_11_0_arm64.whl</a>" not in index
    assert 'dev721-py3-none-macosx_11_0_arm64.whl"' not in index
