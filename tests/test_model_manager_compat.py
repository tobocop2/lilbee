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
    """The engine's verdict picks the file, and refuses when no file answers.

    The gate lives inside ``_pull_native``, so these stub the download rather
    than that method: stubbing ``_pull_native`` would stub away the gate itself
    and leave the assertions unable to fail.
    """

    @staticmethod
    def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelManager:
        _stub_services(monkeypatch)
        monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "llama")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        return ModelManager(models_dir)

    @staticmethod
    def _repo_holds(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        monkeypatch.setattr(
            "lilbee.catalog.download._repo_sibling_files", lambda _repo: list(names)
        )
        monkeypatch.setattr(
            "lilbee.catalog.download.file_header",
            lambda _repo, _name: GgufHeader(architecture="qwen35", file_type="model"),
        )

    @staticmethod
    def _engine_reads(monkeypatch: pytest.MonkeyPatch, *loadable: str) -> None:
        def _check(repo: str, filename: str, _token: str | None = None) -> None:
            if filename not in loadable:
                raise UnsupportedQuantError(f"{repo}/{filename}", "GGMLType(42)")

        monkeypatch.setattr("lilbee.providers.fleet.loadability.assert_engine_can_load", _check)

    @staticmethod
    def _capture_download(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> list[str]:
        chosen: list[str] = []

        def _fake(entry: CatalogModel, **_kw: object) -> Path:
            chosen.append(entry.gguf_filename)
            return models_dir / "x.gguf"

        monkeypatch.setattr("lilbee.catalog.download_model", _fake)
        return chosen

    def test_skips_a_file_the_engine_cannot_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo publishing one packing this build reads and one it does not."""
        mgr = self._manager(tmp_path, monkeypatch)
        self._repo_holds(monkeypatch, "w-PQ2_0.gguf", "w-Q2_0.gguf", "w-Q2_0_g64.gguf")
        self._engine_reads(monkeypatch, "w-Q2_0_g64.gguf")
        chosen = self._capture_download(monkeypatch, tmp_path / "models")
        mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert chosen == ["w-Q2_0_g64.gguf"]

    def test_refuses_when_no_file_in_the_repo_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        self._repo_holds(monkeypatch, "w-PQ2_0.gguf", "w-Q2_0.gguf")
        self._engine_reads(monkeypatch)
        monkeypatch.setattr(
            "lilbee.catalog.download_model",
            lambda *_a, **_kw: pytest.fail("the download started"),
        )
        with pytest.raises(UnsupportedQuantError) as excinfo:
            mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert excinfo.value.quant == "GGMLType(42)"

    def test_allow_unsupported_takes_the_best_ranked_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the check off, ranking alone decides and nothing is refused."""
        mgr = self._manager(tmp_path, monkeypatch)
        self._repo_holds(monkeypatch, "w-PQ2_0.gguf", "w-Q2_0_g64.gguf")
        self._engine_reads(monkeypatch)
        chosen = self._capture_download(monkeypatch, tmp_path / "models")
        mgr.pull("acme/foo-GGUF", ModelSource.NATIVE, allow_unsupported=True)
        assert chosen == ["w-PQ2_0.gguf"]

    def test_refuses_an_unsupported_arch_on_the_resolved_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare repo ref names no file, so the ref-level guard decides nothing."""
        _stub_services(monkeypatch)
        monkeypatch.setattr(compat, "resolve_arch_for_pull", lambda _ref, _client: "")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(
            "lilbee.catalog.download._repo_sibling_files", lambda _repo: ["w-Q4_K_M.gguf"]
        )
        monkeypatch.setattr(
            "lilbee.catalog.download.file_header",
            lambda _repo, _name: GgufHeader(architecture="inkling", file_type="model"),
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.loadability.assert_engine_can_load",
            lambda *_a, **_kw: pytest.fail("asked the parser about an unsupported arch"),
        )
        chosen = self._capture_download(monkeypatch, models_dir)
        ModelManager(models_dir).pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert chosen == ["w-Q4_K_M.gguf"]

    def test_a_repo_that_will_not_resolve_is_left_to_the_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 404 is the download's error to report, not the gate's."""

        def _unresolvable(_repo: str) -> list[str]:
            raise RuntimeError("Cannot query files for acme/foo-GGUF")

        mgr = self._manager(tmp_path, monkeypatch)
        monkeypatch.setattr("lilbee.catalog.download._repo_sibling_files", _unresolvable)
        chosen = self._capture_download(monkeypatch, tmp_path / "models")
        mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
        assert chosen == ["*.gguf"]
