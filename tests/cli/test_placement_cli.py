"""Tests for the `lilbee placement` CLI sub-app."""

import json

from typer.testing import CliRunner

import lilbee.cli.placement as cli_placement
from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.roles import WorkerRole

runner = CliRunner()
_GIB = 1024**3


def _view(manual: bool = False) -> PlacementView:
    return PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100-SXM4-80GB", 80 * _GIB, 72 * _GIB),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=manual,
        spec_json='{"chat": {"devices": [0]}}' if manual else None,
    )


def test_show_renders_cards(monkeypatch: object) -> None:
    """show calls get_placement and renders GPU label, name, and roles."""
    monkeypatch.setattr(cli_placement, "get_placement", lambda: _view())
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    assert "CUDA0" in result.stdout
    assert "NVIDIA A100-SXM4-80GB" in result.stdout
    assert "chat" in result.stdout


def _view_with_skipped() -> PlacementView:
    from lilbee.app.placement import SkippedRole

    return PlacementView(
        gpus=(GpuInfo(0, "MTL", "MTL0", "Apple M3", 24 * _GIB, 20 * _GIB),),
        roles=(RolePlacementView(WorkerRole.EMBED, "org/embed.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=False,
        spec_json=None,
        skipped_not_installed=(SkippedRole(WorkerRole.CHAT, "org/Qwen3-4B.gguf"),),
    )


def test_show_renders_not_downloaded_note(monkeypatch: object) -> None:
    """A role skipped for a missing model prints a 'not downloaded' line."""
    monkeypatch.setattr(cli_placement, "get_placement", _view_with_skipped)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    assert "chat" in result.stdout
    assert "not downloaded" in result.stdout


def test_show_json_includes_skipped_not_installed(monkeypatch: object) -> None:
    """The canonical JSON surfaces skipped-not-installed roles for HTTP/MCP/CLI parity."""
    monkeypatch.setattr(cli_placement, "get_placement", _view_with_skipped)
    monkeypatch.setattr(cfg, "json_mode", True)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["skipped_not_installed"] == [{"role": "chat", "model": "org/Qwen3-4B.gguf"}]


def test_show_json_mode_emits_valid_json(monkeypatch: object) -> None:
    """--json (global flag -> cfg.json_mode) makes placement emit machine-readable
    JSON instead of a Rich table, so scripts can consume it."""
    monkeypatch.setattr(cli_placement, "get_placement", lambda: _view(manual=True))
    monkeypatch.setattr(cfg, "json_mode", True)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)  # must parse, unlike the Rich table
    assert data["manual"] is True
    assert data["gpus"][0]["label"] == "CUDA0"
    assert data["roles"][0]["role"] == "chat"
    assert data["spec_json"] == '{"chat": {"devices": [0]}}'


def test_show_json_mode_error_emits_error_object(monkeypatch: object) -> None:
    """A known failure in --json mode emits {"error": ...}, not a Rich error line."""

    def _boom() -> PlacementView:
        raise ProviderError("engine binary missing", provider="llama-server")

    monkeypatch.setattr(cli_placement, "get_placement", _boom)
    monkeypatch.setattr(cfg, "json_mode", True)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "engine binary missing"


def test_preview_auto(monkeypatch: object) -> None:
    """preview with no --spec calls preview_placement(None)."""
    monkeypatch.setattr(cli_placement, "preview_placement", lambda spec: _view())
    result = runner.invoke(placement_app, ["preview"])
    assert result.exit_code == 0
    assert "CUDA0" in result.stdout


def test_set_reads_spec_file(tmp_path: object, monkeypatch: object) -> None:
    """set --spec FILE parses the JSON and passes a PlacementSpec to set_placement."""
    seen: dict[str, object] = {}

    def _fake_set(spec: object) -> PlacementView:
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(cli_placement, "set_placement", _fake_set)
    f = tmp_path / "spec.json"
    f.write_text('{"chat": {"devices": [0]}}')
    result = runner.invoke(placement_app, ["set", "--spec", str(f)])
    assert result.exit_code == 0
    assert isinstance(seen["spec"], PlacementSpec)
    assert seen["spec"].roles[WorkerRole.CHAT].devices == (0,)


def test_set_reads_spec_from_stdin(monkeypatch: object) -> None:
    """set --spec - reads JSON from stdin and passes a PlacementSpec to set_placement."""
    seen: dict[str, object] = {}

    def _fake_set(spec: object) -> PlacementView:
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(cli_placement, "set_placement", _fake_set)
    result = runner.invoke(
        placement_app, ["set", "--spec", "-"], input='{"chat": {"devices": [0]}}'
    )
    assert result.exit_code == 0
    assert isinstance(seen["spec"], PlacementSpec)
    assert seen["spec"].roles[WorkerRole.CHAT].devices == (0,)


def test_clear(monkeypatch: object) -> None:
    """clear calls set_placement(None)."""
    seen: dict[str, object] = {}

    def _fake_set(spec: object) -> PlacementView:
        seen["spec"] = spec
        return _view()

    monkeypatch.setattr(cli_placement, "set_placement", _fake_set)
    result = runner.invoke(placement_app, ["clear"])
    assert result.exit_code == 0
    assert seen["spec"] is None


def test_show_unplaceable_role(monkeypatch: object) -> None:
    """show renders a red line for each unplaceable role."""
    view = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "A100", 80 * _GIB, 72 * _GIB),),
        roles=(),
        unplaceable=(WorkerRole.EMBED,),
        manual=False,
        spec_json=None,
    )
    monkeypatch.setattr(cli_placement, "get_placement", lambda: view)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    assert "embed" in result.stdout


def test_show_reports_co_tenant_roles(monkeypatch: object) -> None:
    """Co-tenants are placed but never co-resident, so the card they name is not
    over-committed; show says so instead of listing two roles on one GPU."""
    view = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "A100", 80 * _GIB, 72 * _GIB),),
        roles=(),
        unplaceable=(),
        manual=False,
        spec_json=None,
        co_tenants=(WorkerRole.CHAT, WorkerRole.VISION),
    )
    monkeypatch.setattr(cli_placement, "get_placement", lambda: view)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 0
    assert "chat, vision" in result.stdout
    assert "one loaded at a time" in result.stdout


def test_preview_with_spec_file(tmp_path: object, monkeypatch: object) -> None:
    """preview --spec FILE parses the JSON and passes PlacementSpec to preview_placement."""
    seen: dict[str, object] = {}

    def _fake_preview(spec: object) -> PlacementView:
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(cli_placement, "preview_placement", _fake_preview)
    f = tmp_path / "spec.json"
    f.write_text('{"chat": {"devices": [0]}}')
    result = runner.invoke(placement_app, ["preview", "--spec", str(f)])
    assert result.exit_code == 0
    assert isinstance(seen["spec"], PlacementSpec)


def test_preview_placement_error(monkeypatch: object) -> None:
    """preview prints the error and exits 1 when preview_placement raises PlacementError."""

    def _raise(spec: object) -> PlacementView:
        raise PlacementError("bad spec")

    monkeypatch.setattr(cli_placement, "preview_placement", _raise)
    result = runner.invoke(placement_app, ["preview"])
    assert result.exit_code == 1
    assert "bad spec" in result.stdout


def test_set_placement_error(tmp_path: object, monkeypatch: object) -> None:
    """set prints the error and exits 1 when set_placement raises PlacementError."""

    def _raise(spec: object) -> PlacementView:
        raise PlacementError("no fit")

    monkeypatch.setattr(cli_placement, "set_placement", _raise)
    f = tmp_path / "spec.json"
    f.write_text('{"chat": {"devices": [0]}}')
    result = runner.invoke(placement_app, ["set", "--spec", str(f)])
    assert result.exit_code == 1
    assert "no fit" in result.stdout


def test_set_missing_spec_file_exits_clean(tmp_path: object, monkeypatch: object) -> None:
    """set --spec with a nonexistent file exits 1 without a traceback."""
    monkeypatch.setattr(cli_placement, "set_placement", lambda spec: _view(True))
    missing = tmp_path / "nope.json"
    result = runner.invoke(placement_app, ["set", "--spec", str(missing)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_preview_missing_spec_file_exits_clean(tmp_path: object, monkeypatch: object) -> None:
    """preview --spec with a nonexistent file exits 1 without a traceback."""
    monkeypatch.setattr(cli_placement, "preview_placement", lambda spec: _view(True))
    missing = tmp_path / "nope.json"
    result = runner.invoke(placement_app, ["preview", "--spec", str(missing)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def _raises_provider_error() -> PlacementView:
    raise ProviderError("llama-server binary not found")


def test_show_provider_error_exits_clean(monkeypatch: object) -> None:
    """show exits 1 with a clean message when get_placement raises ProviderError."""
    monkeypatch.setattr(cli_placement, "get_placement", _raises_provider_error)
    result = runner.invoke(placement_app, ["show"])
    assert result.exit_code == 1
    assert "llama-server binary not found" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_clear_provider_error_exits_clean(monkeypatch: object) -> None:
    """clear exits 1 with a clean message when set_placement raises ProviderError."""

    def _raise(spec: object) -> PlacementView:
        raise ProviderError("no engine")

    monkeypatch.setattr(cli_placement, "set_placement", _raise)
    result = runner.invoke(placement_app, ["clear"])
    assert result.exit_code == 1
    assert "no engine" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_preview_provider_error_exits_clean(monkeypatch: object) -> None:
    """preview exits 1 cleanly when preview_placement raises ProviderError."""

    def _raise(spec: object) -> PlacementView:
        raise ProviderError("no engine")

    monkeypatch.setattr(cli_placement, "preview_placement", _raise)
    result = runner.invoke(placement_app, ["preview"])
    assert result.exit_code == 1
    assert "no engine" in result.stdout


def test_set_inline_json_spec(monkeypatch: object) -> None:
    """set --spec accepts inline JSON (not only a file path)."""
    seen: dict[str, object] = {}

    def _fake_set(spec: object) -> PlacementView:
        seen["spec"] = spec
        return _view(True)

    monkeypatch.setattr(cli_placement, "set_placement", _fake_set)
    result = runner.invoke(placement_app, ["set", "--spec", '{"chat": {"devices": [0, 1]}}'])
    assert result.exit_code == 0
    assert isinstance(seen["spec"], PlacementSpec)
    assert seen["spec"].roles[WorkerRole.CHAT].devices == (0, 1)
