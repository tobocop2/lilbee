"""A ``#gguf`` suffix on a ref reads as the plain ref at every reader of a stored ref."""

from __future__ import annotations

import pytest

from lilbee import catalog
from lilbee.catalog import compat
from lilbee.catalog.formatting import download_task_name
from lilbee.catalog.refs import (
    gguf_filename_from_ref,
    hf_repo_from_ref,
    is_bare_hf_repo,
    is_native_gguf_shape,
)
from lilbee.data.store.lance_helpers import refs_compatible

_PLAIN = "org/Repo-GGUF/Q4_K_M/model-Q4_K_M.gguf"
_SUFFIXED = f"{_PLAIN}#gguf"
_BARE = "org/Repo-GGUF"
_BARE_SUFFIXED = f"{_BARE}#gguf"


def test_the_one_parse_reads_the_suffixed_ref_as_the_plain_ref() -> None:
    assert hf_repo_from_ref(_SUFFIXED) == hf_repo_from_ref(_PLAIN) == "org/Repo-GGUF"
    assert gguf_filename_from_ref(_SUFFIXED) == "Q4_K_M/model-Q4_K_M.gguf"
    assert is_native_gguf_shape(_SUFFIXED) is True
    assert is_bare_hf_repo(_BARE_SUFFIXED) is True
    assert is_native_gguf_shape(_BARE_SUFFIXED) is False


@pytest.mark.parametrize("ref", [_PLAIN, _SUFFIXED])
def test_the_display_label_of_a_native_ref_is_the_clean_repo_name(ref: str) -> None:
    assert catalog.display_label_for_ref(ref) == catalog.display_label_for_ref(_PLAIN)
    assert catalog.display_label_for_ref(ref) == catalog.clean_display_name("org/Repo-GGUF")


@pytest.mark.parametrize("ref", [_PLAIN, _SUFFIXED])
def test_the_download_task_name_of_a_native_ref_is_the_clean_repo_name(ref: str) -> None:
    assert download_task_name(ref) == catalog.clean_display_name("org/Repo-GGUF")


def test_a_one_slash_gguf_file_with_the_suffix_has_no_repo_label() -> None:
    assert download_task_name("org/model.gguf#gguf") == ""


@pytest.mark.parametrize("ref", [_PLAIN, _SUFFIXED])
def test_the_pull_target_of_a_native_ref_is_the_same_file(ref: str) -> None:
    entry = catalog.resolve_pull_target(ref)
    assert entry is not None
    assert (entry.hf_repo, entry.gguf_filename) == ("org/Repo-GGUF", "Q4_K_M/model-Q4_K_M.gguf")


@pytest.mark.parametrize("ref", [_BARE, _BARE_SUFFIXED])
def test_the_pull_target_of_a_bare_repo_is_the_repo(ref: str) -> None:
    entry = catalog.resolve_pull_target(ref)
    assert entry is not None
    assert entry.hf_repo == _BARE


@pytest.mark.parametrize("ref", [_PLAIN, _SUFFIXED])
def test_the_blob_url_of_a_native_ref_names_the_file(
    ref: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compat, "hf_hub_url", lambda repo, filename: f"{repo}|{filename}")
    assert compat._resolve_blob_url(ref) == "org/Repo-GGUF|Q4_K_M/model-Q4_K_M.gguf"


@pytest.mark.parametrize(
    ("stored", "configured"),
    [
        (_SUFFIXED, _PLAIN),
        (_PLAIN, _SUFFIXED),
        (_SUFFIXED, _SUFFIXED),
        (_BARE, _SUFFIXED),
        (_BARE_SUFFIXED, _PLAIN),
    ],
)
def test_a_store_written_under_either_spelling_reads_as_the_same_embedder(
    stored: str, configured: str
) -> None:
    assert refs_compatible(stored, configured, 8, 8) is True


def test_a_suffixed_ref_of_another_file_still_reads_as_another_embedder() -> None:
    assert refs_compatible(_SUFFIXED, "org/Repo-GGUF/other.gguf#gguf", 8, 8) is False
    assert refs_compatible(_SUFFIXED, _PLAIN, 8, 16) is False
