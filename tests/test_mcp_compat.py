"""MCP surface: catalog_browse carries compat fields; model_pull refuses unsupported."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lilbee.catalog.compat import UnsupportedArchError
from lilbee.catalog.models import CatalogModel, CatalogResult
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.mcp_server import catalog_browse, model_pull


@pytest.fixture
def unsupported_row() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/foo-GGUF",
        gguf_filename="*.gguf",
        size_gb=2.0,
        min_ram_gb=3.0,
        description="",
        featured=False,
        downloads=10,
        task=ModelTask.CHAT,
        architecture="kimi_k2_unsupported",
        compat=ModelCompat.UNSUPPORTED,
    )


def test_catalog_browse_carries_compat_fields(unsupported_row: CatalogModel) -> None:
    result = CatalogResult(total=1, limit=50, offset=0, models=[unsupported_row], has_more=False)
    with patch("lilbee.catalog.query.get_catalog", return_value=result):
        payload = catalog_browse()
    entry = payload["models"][0]
    assert entry["architecture"] == "kimi_k2_unsupported"
    assert entry["compat"] == ModelCompat.UNSUPPORTED.value


@pytest.mark.asyncio
async def test_model_pull_returns_unsupported_arch_error() -> None:
    def _raise(*args: object, **kwargs: object) -> object:
        raise UnsupportedArchError("acme/foo-GGUF", "kimi_k2")

    with patch("lilbee.app.models.pull_model_data", side_effect=_raise):
        result = await model_pull(model="acme/foo-GGUF")

    assert result["ok"] is False
    assert result["error"]["code"] == "unsupported_arch"
    assert result["error"]["arch"] == "kimi_k2"
    assert result["error"]["ref"] == "acme/foo-GGUF"
    assert isinstance(result["error"]["supported_examples"], list)
    assert isinstance(result["error"]["total_supported"], int)


@pytest.mark.asyncio
async def test_model_pull_accepts_allow_unsupported() -> None:
    seen: dict[str, object] = {}

    def _capture(ref, src, *, on_update, allow_unsupported):
        seen["ref"] = ref
        seen["allow_unsupported"] = allow_unsupported
        from lilbee.app.models import PullResult, PullStatus

        return PullResult(model=ref, source="native", status=PullStatus.OK)

    with patch("lilbee.app.models.pull_model_data", side_effect=_capture):
        await model_pull(model="acme/foo-GGUF", allow_unsupported=True)

    assert seen["ref"] == "acme/foo-GGUF"
    assert seen["allow_unsupported"] is True
