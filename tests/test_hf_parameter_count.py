"""The Hub reads that feed the safetensors memory estimate: tensor counts and config.json."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from lilbee.catalog import hf_client
from lilbee.catalog.hf_client import read_model_config, read_parameter_count


def _api_returning(safetensors: object) -> type:
    class _Api:
        def model_info(self, repo: str, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(safetensors=safetensors)

    return _Api


def _api_raising(error: Exception) -> type:
    class _Api:
        def model_info(self, repo: str, **_kwargs: object) -> None:
            raise error

    return _Api


class TestReadParameterCount:
    def test_returns_the_per_dtype_count_of_one_model_info_request(self, monkeypatch) -> None:
        seen: list[dict[str, object]] = []

        class _Api:
            def model_info(self, repo: str, **kwargs: object) -> SimpleNamespace:
                seen.append({"repo": repo, **kwargs})
                return SimpleNamespace(safetensors=SimpleNamespace(parameters={"BF16": 4}))

        monkeypatch.setattr(hf_client, "HfApi", _Api)
        assert read_parameter_count("org/model", token="t") == {"BF16": 4}
        assert seen == [{"repo": "org/model", "expand": ["safetensors"], "token": "t"}]

    def test_a_repo_with_no_safetensors_figures_gives_an_empty_count(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_returning(None))
        assert read_parameter_count("org/model") == {}

    def test_an_offline_hub_gives_an_empty_count(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_raising(OSError("offline")))
        assert read_parameter_count("org/model") == {}

    def test_a_transport_error_gives_an_empty_count(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_raising(httpx.ConnectError("down")))
        assert read_parameter_count("org/model") == {}

    def test_an_http_error_gives_an_empty_count(self, monkeypatch) -> None:
        response = httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
        monkeypatch.setattr(
            hf_client, "HfApi", _api_raising(HfHubHTTPError("404", response=response))
        )
        assert read_parameter_count("org/missing") == {}

    @pytest.mark.parametrize("parameters", [["BF16", 4], 7, None, {"BF16": 1}.keys()], ids=repr)
    def test_figures_of_the_wrong_shape_give_an_empty_count(
        self, monkeypatch, parameters: object
    ) -> None:
        monkeypatch.setattr(
            hf_client, "HfApi", _api_returning(SimpleNamespace(parameters=parameters))
        )
        assert read_parameter_count("org/model") == {}

    @pytest.mark.parametrize(
        "error", [TypeError("bad shape"), ValueError("bad int"), KeyError("dtype")]
    )
    def test_a_parse_failure_inside_the_hub_read_gives_an_empty_count(
        self, monkeypatch, error
    ) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_raising(error))
        assert read_parameter_count("org/model") == {}


class _Session:
    """A stand-in for the library's HTTP session that records the one request it gets."""

    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _response(status: int, body: bytes) -> httpx.Response:
    return httpx.Response(
        status, content=body, request=httpx.Request("GET", "https://hub.test/org/model/config")
    )


class TestReadModelConfig:
    def test_returns_the_config_of_one_request_with_the_token(self, monkeypatch) -> None:
        session = _Session(_response(200, b'{"num_hidden_layers": 36}'))
        monkeypatch.setattr(hf_client, "get_session", lambda: session)
        assert read_model_config("org/model", token="secret") == {"num_hidden_layers": 36}
        [(url, kwargs)] = session.calls
        assert url.endswith("/org/model/resolve/main/config.json")
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        assert headers["authorization"] == "Bearer secret"
        assert kwargs["follow_redirects"] is True

    def test_a_missing_file_gives_none(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "get_session", lambda: _Session(_response(404, b"")))
        assert read_model_config("org/model") is None

    def test_a_config_that_is_not_a_mapping_gives_none(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "get_session", lambda: _Session(_response(200, b"[1, 2]")))
        assert read_model_config("org/model") is None

    def test_a_body_that_is_not_json_gives_none(self, monkeypatch) -> None:
        monkeypatch.setattr(hf_client, "get_session", lambda: _Session(_response(200, b"<html>")))
        assert read_model_config("org/model") is None

    def test_a_transport_error_gives_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            hf_client, "get_session", lambda: _Session(httpx.ConnectError("offline"))
        )
        assert read_model_config("org/model") is None
