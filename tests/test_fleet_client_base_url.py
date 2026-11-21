"""The base URL of a server client: a trailing slash never reaches the request URLs."""

from __future__ import annotations

import httpx
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole


def _client(base_url: str) -> LlamaServerClient:
    return LlamaServerClient(base_url, "m", profile=LLAMA.request_profile(WorkerRole.CHAT))


@pytest.mark.parametrize(
    "base_url", ["http://gpu0/", "http://gpu0//", "http://gpu0"], ids=["one", "two", "none"]
)
def test_a_base_url_gives_single_slash_request_urls(base_url: str) -> None:
    client = _client(base_url)

    assert client.sent(httpx.Request("GET", "http://gpu0/v1/models"))
    assert str(client._http.build_request("GET", "/v1/models").url) == "http://gpu0/v1/models"


def test_a_base_url_that_ends_in_an_upper_case_x_keeps_the_letter() -> None:
    """Only the slashes go; a letter at the end of the host stays (the host is lower-cased)."""
    client = _client("http://gpuX/")

    assert str(client._http.build_request("GET", "/v1/models").url) == "http://gpux/v1/models"
