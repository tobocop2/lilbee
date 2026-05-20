"""Catalog detail drawer renders the right compat sentence per verdict."""

from __future__ import annotations

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.catalog_detail import _compat_sentence


def _row(compat: ModelCompat, *, arch: str = "") -> LocalCatalogRow:
    catalog_model = (
        CatalogModel(
            hf_repo="acme/foo-GGUF",
            gguf_filename="*.gguf",
            size_gb=1.0,
            min_ram_gb=2.0,
            description="",
            featured=False,
            downloads=0,
            task=ModelTask.CHAT,
            architecture=arch,
            compat=compat,
        )
        if arch or compat is not ModelCompat.UNKNOWN
        else None
    )
    return LocalCatalogRow(
        name="x",
        task="chat",
        params="",
        size="",
        quant="",
        downloads="",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0,
        ref="acme/foo-GGUF",
        backend="native",
        compat=compat,
        catalog_model=catalog_model,
    )


def test_supported_sentence() -> None:
    assert _compat_sentence(_row(ModelCompat.SUPPORTED, arch="llama")) == (
        msg.COMPAT_DETAIL_SENTENCE_SUPPORTED
    )


def test_unsupported_sentence_includes_arch() -> None:
    out = _compat_sentence(_row(ModelCompat.UNSUPPORTED, arch="kimi_k2"))
    assert "kimi_k2" in out


def test_unknown_sentence_uses_unknown_label_when_no_arch() -> None:
    out = _compat_sentence(_row(ModelCompat.UNKNOWN))
    assert out == msg.COMPAT_DETAIL_SENTENCE_UNKNOWN
