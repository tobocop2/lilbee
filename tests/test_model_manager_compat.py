"""ModelManager.pull's architecture compatibility gate."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.catalog import compat
from lilbee.catalog.compat import UnsupportedArchError, UnsupportedQuantError
from lilbee.catalog.header_probe import GgufHeader
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelSource
from lilbee.modelhub.model_manager import ModelManager


def _stub_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch get_services() so enforce_arch_compat doesn't need a real container."""
    import lilbee.app.services as services_mod

    stub = mock.MagicMock()
    stub.hf_client = mock.MagicMock()
    monkeypatch.setattr(services_mod, "get_services", lambda: stub)


def test_pull_refuses_unsupported_arch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_services(monkeypatch)
    monkeypatch.setattr(
        compat, "resolve_arch_for_pull", lambda _ref, _client: "kimi_k2_unsupported"
    )

    mgr = ModelManager(tmp_path / "models")
    with pytest.raises(UnsupportedArchError) as excinfo:
        mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
    assert excinfo.value.architecture == "kimi_k2_unsupported"
    assert excinfo.value.ref == "acme/foo-GGUF"


def test_pull_bypassed_with_allow_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_services(monkeypatch)
    monkeypatch.setattr(
        compat, "resolve_arch_for_pull", lambda _ref, _client: "kimi_k2_unsupported"
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mgr = ModelManager(models_dir)
    captured: list[str] = []
    monkeypatch.setattr(
        mgr,
        "_pull_native",
        lambda model, on_bytes, cancel, allow_unsupported: (
            captured.append(model) or models_dir / "x.gguf"
        ),
    )

    mgr.pull("acme/foo-GGUF", ModelSource.NATIVE, allow_unsupported=True)
    assert captured == ["acme/foo-GGUF"]


def test_pull_proceeds_for_supported_arch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_services(monkeypatch)
    monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "llama")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mgr = ModelManager(models_dir)
    captured: list[str] = []
    monkeypatch.setattr(
        mgr,
        "_pull_native",
        lambda model, on_bytes, cancel, allow_unsupported: (
            captured.append(model) or models_dir / "x.gguf"
        ),
    )
    mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
    assert captured == ["acme/foo-GGUF"]


def test_pull_proceeds_for_unknown_arch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UNKNOWN means we couldn't determine; post-download check is the guard."""
    _stub_services(monkeypatch)
    monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mgr = ModelManager(models_dir)
    monkeypatch.setattr(
        mgr,
        "_pull_native",
        lambda model, on_bytes, cancel, allow_unsupported: models_dir / "x.gguf",
    )
    result = mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
    assert result is not None


def test_pull_remote_source_refused_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REMOTE source is read-only: refused before the arch gate can run."""
    mgr = ModelManager(tmp_path / "models")

    def _fail(_ref: str, _client: object) -> str:
        raise AssertionError("gate must not fire for a refused REMOTE pull")

    monkeypatch.setattr(compat, "resolve_arch_for_pull", _fail)
    # Generic REMOTE source maps to no specific server, so the refusal names
    # "the configured server" rather than Ollama/LM Studio.
    with pytest.raises(ValueError, match="the configured server"):
        mgr.pull("ollama:llama3", ModelSource.REMOTE)


class TestLoadabilityGate:
    """The pull refuses a file the bundled engine cannot decode.

    The gate lives inside ``_pull_native``, so these stub the download rather
    than that method: stubbing ``_pull_native`` would stub away the gate itself
    and leave the assertions unable to fail.
    """

    @staticmethod
    def _refusing(monkeypatch: pytest.MonkeyPatch) -> None:
        def _refuse(repo: str, filename: str, _token: str | None = None) -> None:
            raise UnsupportedQuantError(f"{repo}/{filename}", "GGMLType(42)")

        monkeypatch.setattr("lilbee.providers.fleet.loadability.assert_engine_can_load", _refuse)

    @staticmethod
    def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelManager:
        _stub_services(monkeypatch)
        monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "llama")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(
            "lilbee.catalog.download.resolve_filename", lambda _entry: "weights-Q2_0.gguf"
        )
        return ModelManager(models_dir)

    def test_refuses_a_file_the_engine_cannot_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._refusing(monkeypatch)
        mgr = self._manager(tmp_path, monkeypatch)
        with pytest.raises(UnsupportedQuantError) as excinfo:
            mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert excinfo.value.quant == "GGMLType(42)"

    def test_refuses_before_any_transfer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the gate is that no bytes move."""
        self._refusing(monkeypatch)
        mgr = self._manager(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "lilbee.catalog.download_model",
            lambda *_a, **_kw: pytest.fail("the download started"),
        )
        with pytest.raises(UnsupportedQuantError):
            mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)

    def test_allow_unsupported_downloads_it_anyway(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._refusing(monkeypatch)
        mgr = self._manager(tmp_path, monkeypatch)
        downloaded: list[str] = []

        def _fake_download(entry: CatalogModel, **_kw: object) -> Path:
            downloaded.append(entry.hf_repo)
            return tmp_path / "models" / "x.gguf"

        monkeypatch.setattr("lilbee.catalog.download_model", _fake_download)
        mgr.pull("acme/foo-GGUF", ModelSource.NATIVE, allow_unsupported=True)
        assert downloaded == ["acme/foo-GGUF"]

    def test_a_repo_that_will_not_resolve_is_left_to_the_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 404 is the download's error to report, not the gate's."""

        def _unresolvable(_entry: object) -> str:
            raise RuntimeError("Cannot query files for acme/foo-GGUF")

        _stub_services(monkeypatch)
        monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "llama")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr("lilbee.catalog.download.resolve_filename", _unresolvable)
        monkeypatch.setattr(
            "lilbee.providers.fleet.loadability.assert_engine_can_load",
            lambda *_a, **_kw: pytest.fail("checked a file it could not name"),
        )
        reached: list[str] = []
        monkeypatch.setattr(
            "lilbee.catalog.download_model",
            lambda entry, **_kw: reached.append(entry.hf_repo) or models_dir / "x.gguf",
        )
        ModelManager(models_dir).pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert reached == ["acme/foo-GGUF"]

    def test_refuses_an_unsupported_arch_on_the_resolved_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare repo ref names no file, so the ref-level guard decides nothing."""
        _stub_services(monkeypatch)
        monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(
            "lilbee.catalog.download.resolve_filename", lambda _entry: "weights-Q4_K_M.gguf"
        )
        monkeypatch.setattr(
            "lilbee.catalog.compat.file_header",
            lambda _repo, _name: GgufHeader(architecture="inkling", file_type="model"),
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.loadability.assert_engine_can_load",
            lambda *_a, **_kw: pytest.fail("ran the parser on an unsupported architecture"),
        )
        with pytest.raises(UnsupportedArchError) as excinfo:
            ModelManager(models_dir).pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert excinfo.value.architecture == "inkling"
        assert excinfo.value.ref == "acme/foo-GGUF/weights-Q4_K_M.gguf"
