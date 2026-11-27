"""Every entry point checks the per-call OCR options through the one shared check."""

from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.core.config import validate_ocr_options
from lilbee.core.config.enums import OcrMode
from lilbee.mcp_server import add, sync
from lilbee.server import auth as _auth_mod
from lilbee.server.handlers import ingest as _ingest_h

_REFUSAL = "ocr options refused by the shared check"


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise ValueError(_REFUSAL)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


class TestSharedCheckMessage:
    @pytest.mark.parametrize("bad", ["bogus", "ALL", " all", 5])
    def test_a_bad_mode_names_the_option_and_the_value(self, bad):
        with pytest.raises(ValueError, match=rf"ocr must be one of auto, all, off; got {bad!r}"):
            validate_ocr_options(bad, None)

    def test_a_plain_string_is_read_as_the_mode(self):
        assert validate_ocr_options("all", None) is OcrMode.ALL

    def test_the_http_decoder_gives_the_same_message(self):
        with pytest.raises(ValueError, match=r"ocr must be one of .*; got 'bogus'"):
            _ingest_h.parse_ocr_mode("bogus")


class TestHttpUsesTheSharedCheck:
    async def test_sync_route_refuses_through_the_shared_check(self):
        from lilbee.server.app import create_app

        with (
            mock.patch("lilbee.server.routes.documents.validate_ocr_options", _refuse),
            mock.patch("lilbee.server.handlers.sync_stream") as stream,
        ):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post("/api/sync", json={}, headers=_auth_headers())
        assert resp.status_code == 400
        assert _REFUSAL in resp.text
        stream.assert_not_called()

    async def test_upload_route_refuses_through_the_shared_check(self):
        from lilbee.server.app import create_app

        with mock.patch("lilbee.server.routes.documents.validate_ocr_options", _refuse):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/add/upload",
                    files=[("data", ("notes.txt", b"x", "text/plain"))],
                    headers=_auth_headers(),
                )
        assert resp.status_code == 400
        assert _REFUSAL in resp.text

    async def test_upload_route_reports_a_bad_mode_by_option(self):
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add/upload?ocr=bogus",
                files=[("data", ("notes.txt", b"x", "text/plain"))],
                headers=_auth_headers(),
            )
        assert resp.status_code == 400
        assert "ocr must be one of auto, all, off; got 'bogus'" in resp.text

    def test_add_request_params_refuse_through_the_shared_check(self):
        with (
            mock.patch("lilbee.server.handlers.ingest.validate_ocr_options", _refuse),
            pytest.raises(ValueError, match=_REFUSAL),
        ):
            _ingest_h._parse_ocr_params({"ocr": "all"})


class TestMcpUsesTheSharedCheck:
    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
    async def test_sync_refuses_through_the_shared_check(self, run_sync):
        with mock.patch("lilbee.mcp_server.validate_ocr_options", _refuse):
            result = await sync()
        assert result == {"error": _REFUSAL}
        run_sync.assert_not_called()

    @mock.patch("lilbee.data.ingest.sync", new_callable=AsyncMock)
    async def test_add_refuses_through_the_shared_check(self, run_sync, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content", encoding="utf-8")
        with mock.patch("lilbee.mcp_server.validate_ocr_options", _refuse):
            result = await add([str(src)])
        assert result == {"error": _REFUSAL}
        run_sync.assert_not_called()


class TestBadModeThroughEachEntryPoint:
    def test_add_handler_refuses_a_bad_mode_by_option(self):
        """The add route's body model rejects a bad mode first; the handler still checks."""
        from lilbee.server import handlers

        with pytest.raises(ValueError, match=r"ocr must be one of auto, all, off; got 'bogus'"):
            handlers.validate_add_paths({"paths": ["notes.txt"], "ocr": "bogus"})

    async def test_mcp_sync_refuses_a_bad_mode_by_option(self):
        result = await sync(ocr="bogus")
        assert result == {"error": "ocr must be one of auto, all, off; got 'bogus'"}

    async def test_mcp_add_refuses_a_bad_mode_by_option(self, tmp_path):
        src = tmp_path / "file.txt"
        src.write_text("content", encoding="utf-8")
        result = await add([str(src)], ocr="bogus")
        assert result == {"error": "ocr must be one of auto, all, off; got 'bogus'"}
