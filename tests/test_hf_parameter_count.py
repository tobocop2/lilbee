"""The Hub header read that feeds the safetensors memory estimate."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError,
    NotASafetensorsRepoError,
    SafetensorsParsingError,
)

from lilbee.catalog import hf_client
from lilbee.catalog.hf_client import read_parameter_count


def _api_returning(parameter_count: object) -> type:
    class _Api:
        def get_safetensors_metadata(self, repo: str, *, token: object) -> SimpleNamespace:
            return SimpleNamespace(parameter_count=parameter_count)

    return _Api


def _api_raising(error: Exception) -> type:
    class _Api:
        def get_safetensors_metadata(self, repo: str, *, token: object) -> None:
            raise error

    return _Api


class TestReadParameterCount:
    def test_returns_the_per_dtype_count_from_the_header_read(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        class _Api:
            def get_safetensors_metadata(self, repo: str, *, token: object) -> SimpleNamespace:
                seen.update(repo=repo, token=token)
                return SimpleNamespace(parameter_count={"BF16": 4})

        monkeypatch.setattr(hf_client, "HfApi", _Api)
        assert read_parameter_count("org/model", token="t") == {"BF16": 4}
        assert seen == {"repo": "org/model", "token": "t"}

    @pytest.mark.parametrize(
        "error",
        [
            NotASafetensorsRepoError("no header"),
            SafetensorsParsingError("bad header"),
            OSError("offline"),
        ],
    )
    def test_an_unreadable_header_gives_an_empty_count(self, monkeypatch, error) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_raising(error))
        assert read_parameter_count("org/model") == {}

    def test_an_http_error_gives_an_empty_count(self, monkeypatch) -> None:
        response = httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
        monkeypatch.setattr(
            hf_client, "HfApi", _api_raising(HfHubHTTPError("404", response=response))
        )
        assert read_parameter_count("org/missing") == {}

    @pytest.mark.parametrize(
        "parameter_count", [["BF16", 4], 7, None, {"BF16": 1}.keys()], ids=repr
    )
    def test_a_header_of_the_wrong_shape_gives_an_empty_count(
        self, monkeypatch, parameter_count: object
    ) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_returning(parameter_count))
        assert read_parameter_count("org/model") == {}

    @pytest.mark.parametrize(
        "error", [TypeError("bad shape"), ValueError("bad int"), KeyError("dtype")]
    )
    def test_a_parse_failure_inside_the_hub_read_gives_an_empty_count(
        self, monkeypatch, error
    ) -> None:
        monkeypatch.setattr(hf_client, "HfApi", _api_raising(error))
        assert read_parameter_count("org/model") == {}
