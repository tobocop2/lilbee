"""HTTP layer: /api/models/pull returns 409 on unsupported, accepts override."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from litestar.exceptions import HTTPException

from lilbee.catalog.compat import UnsupportedArchError
from lilbee.server import handlers


@pytest.mark.asyncio
async def test_enforce_arch_compat_raises_409() -> None:
    """The route-level precheck converts an unsupported arch to HTTPException 409.

    It must live outside the models_pull async generator: a raise there fires only
    on first iteration, after the 200 SSE headers flush, and can't set the status.
    """
    mock_manager = MagicMock()
    mock_manager._enforce_arch_compat.side_effect = UnsupportedArchError("acme/foo-GGUF", "kimi_k2")

    with (
        patch(
            "lilbee.server.handlers.models.get_services",
            return_value=MagicMock(model_manager=mock_manager),
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await handlers.enforce_pull_arch_compat("acme/foo-GGUF", source="native")

    assert excinfo.value.status_code == 409
    extra = excinfo.value.extra
    assert extra["code"] == "unsupported_arch"
    assert extra["arch"] == "kimi_k2"
    assert extra["ref"] == "acme/foo-GGUF"
    assert "supported_examples" in extra
    assert isinstance(extra["total_supported"], int)


@pytest.mark.asyncio
async def test_enforce_arch_compat_skips_remote_and_override() -> None:
    """Remote source and allow_unsupported both bypass the native arch precheck."""
    mock_manager = MagicMock()
    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=mock_manager),
    ):
        await handlers.enforce_pull_arch_compat("ollama:llama3", source="remote")
        await handlers.enforce_pull_arch_compat("acme/foo", source="native", allow_unsupported=True)
    mock_manager._enforce_arch_compat.assert_not_called()


@pytest.mark.asyncio
async def test_models_pull_generator_does_not_precheck() -> None:
    """The generator no longer prechecks (the route does); a raise here would be too
    late. manager.pull still enforces compatibility during the pull itself."""
    mock_manager = MagicMock()
    mock_manager._enforce_arch_compat.side_effect = UnsupportedArchError("acme/foo-GGUF", "kimi_k2")
    mock_manager.pull.return_value = None

    with patch(
        "lilbee.server.handlers.models.get_services",
        return_value=MagicMock(model_manager=mock_manager),
    ):
        events = [e async for e in handlers.models_pull("acme/foo-GGUF", source="native")]

    mock_manager._enforce_arch_compat.assert_not_called()
    mock_manager.pull.assert_called_once()
    assert events is not None


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
