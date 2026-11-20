"""The engine service lists every engine a model can run on, with state, backend and size."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from tests.conformance.fake_adapter import BundledFakeAdapter

GGUF = "org/repo/file.gguf"


class _Gguf(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


@pytest.fixture(autouse=True)
def _engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Gguf())  # type: ignore[arg-type]
    monkeypatch.setattr(engine_service, "pack_backend", lambda _host: "metal")


def _linux_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
    monkeypatch.setattr(runnable, "session_host", lambda: host)


def _rows(model: str) -> dict[str, engine_service.EngineOption]:
    return {option.name: option for option in engine_service.engine_options(model)}


def test_the_first_row_is_auto_and_names_the_engine_auto_picks() -> None:
    first = engine_service.engine_options(GGUF)[0]
    assert (first.name, first.auto_now) == ("auto", "llama.cpp")


def test_auto_names_no_engine_when_none_loads_the_model() -> None:
    assert engine_service.engine_options("org/repo#safetensors")[0].auto_now is None


def test_an_installed_engine_row_has_no_download_and_a_bundled_one_says_so() -> None:
    row = _rows(GGUF)["llama.cpp"]
    assert (row.installed, row.bundled, row.backend, row.download_bytes) == (
        True,
        True,
        "metal",
        None,
    )


def test_an_engine_with_no_build_shows_its_backend_and_the_size_of_its_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux_cpu_host(monkeypatch)
    row = _rows(GGUF)["ik_llama.cpp"]
    assert row.installed is False
    assert row.backend == "cpu"
    assert isinstance(row.download_bytes, int) and row.download_bytes > 0


def test_an_engine_the_host_refuses_is_not_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    host = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["sse4_2"]}, nvidia=None)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    assert "ik_llama.cpp" not in _rows(GGUF)


def test_an_engine_that_does_not_load_the_format_is_not_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux_cpu_host(monkeypatch)
    assert "ik_llama.cpp" not in _rows("org/repo#mlx")


def test_an_engine_with_no_published_pack_has_no_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_cpu_host(monkeypatch)

    def unpublished(_pin: object) -> None:
        raise choice.NoPublishedPackError("no pack")

    monkeypatch.setattr(choice, "pin_lock", unpublished)
    row = _rows(GGUF)["ik_llama.cpp"]
    assert (row.installed, row.download_bytes, row.backend) == (False, None, "")
