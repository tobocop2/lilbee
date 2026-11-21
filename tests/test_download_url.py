"""The URL downloader against a loopback server that stands in for a release download."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from huggingface_hub import constants as hub_constants

from lilbee.catalog import download as dl
from lilbee.catalog.download import download_url, url_filename
from tests._release_server import REDIRECT_PATH, release_server

BODY = bytes(range(256)) * 400


@pytest.fixture(autouse=True)
def hub_online(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite switches the hub's network off; these tests talk to a loopback server."""
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)


def test_url_filename_is_the_last_path_segment() -> None:
    assert url_filename("https://example.com/a/b/engine-1.whl?x=1#sha256=ab") == "engine-1.whl"


def test_download_writes_the_named_file_into_the_destination(tmp_path: Path) -> None:
    with release_server() as server:
        server.files["/f"] = BODY
        got = download_url(f"{server.url}/f", tmp_path / "dl", len(BODY))
    assert got == tmp_path / "dl" / "f"
    assert got.read_bytes() == BODY


def test_download_follows_a_redirect_and_sends_no_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_must_not_leave")
    with release_server() as server:
        server.files["/f"] = BODY
        got = download_url(f"{server.url}{REDIRECT_PATH}", tmp_path, len(BODY))
    assert got.read_bytes() == BODY
    assert len(server.requests) == 2
    assert all("authorization" not in request for request in server.requests)


def test_download_resumes_a_partial_file_with_a_range_request(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(BODY[:1000])
    with release_server() as server:
        server.files["/f"] = BODY
        got = download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert got.read_bytes() == BODY
    assert server.requests[-1]["range"].startswith("bytes=1000-")


def test_download_of_a_missing_file_fails_at_once_without_a_retry(tmp_path: Path) -> None:
    with (
        release_server() as server,
        mock.patch.object(dl.time, "sleep") as slept,
        pytest.raises(
            RuntimeError, match=r"The server cannot give the file at .*/gone \(HTTP 404\)\."
        ),
    ):
        download_url(f"{server.url}/gone", tmp_path, 1)
    assert len(server.requests) == 1
    slept.assert_not_called()


def test_download_retries_a_server_fault_then_succeeds(tmp_path: Path) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep") as slept:
        server.files["/f"] = BODY
        server.fail_next = 1
        got = download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert got.read_bytes() == BODY
    assert len(server.requests) == 2
    slept.assert_called_once()


def test_download_gives_up_after_three_server_faults(tmp_path: Path) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep"):
        server.files["/f"] = BODY
        server.fail_next = 10
        with pytest.raises(RuntimeError, match="The transfer failed 3 times"):
            download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert len(server.requests) == 3


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_a_refusal_the_server_will_repeat_is_permanent_after_one_request(
    tmp_path: Path, status: int
) -> None:
    with (
        release_server() as server,
        mock.patch.object(dl.time, "sleep") as slept,
        pytest.raises(RuntimeError, match=rf"\(HTTP {status}\)"),
    ):
        server.files["/f"] = BODY
        server.fail_next = 5
        server.fail_status = status
        download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert len(server.requests) == 1
    slept.assert_not_called()


def test_a_reset_mid_body_is_retried_and_the_file_completes(tmp_path: Path) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep") as slept:
        server.files["/f"] = BODY
        server.reset_next = 1
        got = download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert got.read_bytes() == BODY
    assert len(server.requests) == 2
    slept.assert_called_once()


def test_a_connection_closed_before_a_status_line_is_retried(tmp_path: Path) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep") as slept:
        server.files["/f"] = BODY
        server.close_next = 1
        got = download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert got.read_bytes() == BODY
    assert len(server.requests) == 2
    slept.assert_called_once()


def test_a_server_that_keeps_resetting_ends_in_the_transfer_failed_text(tmp_path: Path) -> None:
    with release_server() as server, mock.patch.object(dl.time, "sleep"):
        server.files["/f"] = BODY
        server.reset_next = 10
        with pytest.raises(RuntimeError, match="The transfer failed 3 times"):
            download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert len(server.requests) == 3


def test_the_hub_offline_switch_ends_the_transfer_with_no_request_and_no_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    with (
        release_server() as server,
        mock.patch.object(dl.time, "sleep") as slept,
        pytest.raises(RuntimeError, match="Downloads are switched off"),
    ):
        server.files["/f"] = BODY
        download_url(f"{server.url}/f", tmp_path, len(BODY))
    assert server.requests == []
    slept.assert_not_called()
