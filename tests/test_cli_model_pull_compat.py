"""CLI: --allow-unsupported overrides the gate; refusal exits non-zero."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from lilbee.catalog.compat import UnsupportedArchError
from lilbee.cli.model import model_app


def test_pull_exits_nonzero_on_unsupported() -> None:
    runner = CliRunner()
    with patch(
        "lilbee.cli.model.pull_model_data",
        side_effect=UnsupportedArchError("acme/foo-GGUF", "kimi_k2"),
    ):
        result = runner.invoke(model_app, ["pull", "acme/foo-GGUF"])
    assert result.exit_code == 1
    assert "kimi_k2" in result.stdout
    assert "--allow-unsupported" in result.stdout


def test_pull_with_override_proceeds() -> None:
    runner = CliRunner()
    seen: dict[str, object] = {}

    def _capture(ref, src, *, on_update, allow_unsupported):
        seen["ref"] = ref
        seen["allow_unsupported"] = allow_unsupported
        from lilbee.app.models import PullResult, PullStatus

        return PullResult(model=ref, source="native", status=PullStatus.OK)

    with patch("lilbee.cli.model.pull_model_data", side_effect=_capture):
        result = runner.invoke(model_app, ["pull", "acme/foo-GGUF", "--allow-unsupported"])
    assert result.exit_code == 0
    assert seen["allow_unsupported"] is True


def test_pull_json_mode_emits_structured_error_on_unsupported() -> None:
    """`--json` is a top-level lilbee flag; invoke the root CLI to enable it."""
    from lilbee.cli.app import app as root_app
    from lilbee.core.config import cfg

    runner = CliRunner()
    original_json_mode = cfg.json_mode
    try:
        with patch(
            "lilbee.cli.model.pull_model_data",
            side_effect=UnsupportedArchError("acme/foo-GGUF", "kimi_k2"),
        ):
            result = runner.invoke(root_app, ["--json", "model", "pull", "acme/foo-GGUF"])
        assert result.exit_code == 1
        out = result.stdout.replace(" ", "")
        assert '"error":"unsupported_arch"' in out
        assert '"arch":"kimi_k2"' in out
    finally:
        cfg.json_mode = original_json_mode
