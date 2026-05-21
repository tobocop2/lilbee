"""ModelManager.pull's architecture compatibility gate."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.catalog import compat
from lilbee.catalog.compat import UnsupportedArchError
from lilbee.catalog.types import ModelSource
from lilbee.modelhub.model_manager import ModelManager


def _stub_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch get_services() so _enforce_arch_compat doesn't need a real container."""
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
        lambda model, on_bytes: captured.append(model) or models_dir / "x.gguf",
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
        lambda model, on_bytes: captured.append(model) or models_dir / "x.gguf",
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
    monkeypatch.setattr(mgr, "_pull_native", lambda model, on_bytes: models_dir / "x.gguf")
    result = mgr.pull("acme/foo-GGUF", ModelSource.NATIVE)
    assert result is not None


def test_pull_remote_source_skips_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REMOTE source delegates to an SDK backend; lilbee's gate doesn't apply."""
    mgr = ModelManager(tmp_path / "models")
    monkeypatch.setattr(mgr, "_pull_remote", lambda model, on_progress: None)

    def _fail(_ref: str, _client: object) -> str:
        raise AssertionError("gate must not fire for REMOTE source")

    monkeypatch.setattr(compat, "resolve_arch_for_pull", _fail)
    mgr.pull("ollama:llama3", ModelSource.REMOTE)
