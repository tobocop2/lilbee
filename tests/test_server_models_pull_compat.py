"""HTTP layer: /api/models/pull returns 409 on unsupported, accepts override."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from litestar.exceptions import HTTPException

from lilbee.catalog.compat import UnsupportedArchError
from lilbee.server import handlers


@pytest.mark.asyncio
async def test_pull_native_unsupported_raises_409() -> None:
    """When _enforce_arch_compat raises, the handler converts to HTTPException 409."""
    mock_manager = MagicMock()
    mock_manager._enforce_arch_compat.side_effect = UnsupportedArchError("acme/foo-GGUF", "kimi_k2")

    with (
        patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        gen = handlers.models_pull("acme/foo-GGUF", source="native")
        await gen.__anext__()

    assert excinfo.value.status_code == 409
    extra = excinfo.value.extra
    assert extra["code"] == "unsupported_arch"
    assert extra["arch"] == "kimi_k2"
    assert extra["ref"] == "acme/foo-GGUF"
    assert "supported_examples" in extra
    assert isinstance(extra["total_supported"], int)


@pytest.mark.asyncio
async def test_pull_native_with_allow_unsupported_skips_precheck() -> None:
    """allow_unsupported=True must bypass the pre-stream gate entirely."""
    mock_manager = MagicMock()

    def fake_pull(model, source, *, on_progress=None, on_bytes=None, allow_unsupported=False):
        return None

    mock_manager.pull.side_effect = fake_pull

    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=mock_manager),
    ):
        events = [
            e
            async for e in handlers.models_pull(
                "acme/foo-GGUF", source="native", allow_unsupported=True
            )
        ]

    mock_manager._enforce_arch_compat.assert_not_called()
    mock_manager.pull.assert_called_once()
    call_kwargs = mock_manager.pull.call_args.kwargs
    assert call_kwargs["allow_unsupported"] is True
    assert events is not None


@pytest.mark.asyncio
async def test_pull_remote_skips_arch_precheck() -> None:
    """REMOTE source never invokes the pre-stream gate."""
    mock_manager = MagicMock()

    def fake_pull(model, source, *, on_progress=None, on_bytes=None, allow_unsupported=False):
        return None

    mock_manager.pull.side_effect = fake_pull

    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=mock_manager),
    ):
        async for _ in handlers.models_pull("ollama:llama3", source="remote"):
            pass

    mock_manager._enforce_arch_compat.assert_not_called()
