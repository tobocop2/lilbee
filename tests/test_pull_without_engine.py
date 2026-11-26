"""A pull of a model no engine serves yet downloads it; a GGUF pull still needs its engine."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from litestar.exceptions import HTTPException
from textual.app import ComposeResult
from textual.widgets import Static

from conftest import make_hf_client
from lilbee.app import engine_choice
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelSource, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import ModelManager
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.server import handlers
from tests._async_wait import wait_for_pull_check
from tests._lilbee_app_test_host import LilbeeAppHost

SAFETENSORS_REF = "acme/tiny-model#safetensors"
MLX_REF = "acme/tiny-model#mlx"
GGUF_REF = "acme/tiny-model-GGUF"


@pytest.fixture(autouse=True)
def _no_engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_choice, "installed_engines", lambda: {})


def _entry(ref: str, fmt: ModelFormat, task: ModelTask = ModelTask.CHAT) -> CatalogModel:
    return CatalogModel(
        hf_repo=ref.split("#")[0],
        gguf_filename="model.safetensors",
        size_gb=0.0,
        min_ram_gb=0.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
        format=fmt,
    )


@pytest.mark.parametrize(
    ("ref", "fmt"),
    [(SAFETENSORS_REF, ModelFormat.SAFETENSORS), (MLX_REF, ModelFormat.MLX)],
)
def test_the_manager_downloads_a_model_with_no_engine_installed(
    ref: str, fmt: ModelFormat, tmp_path: Path
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    dest = models_dir / "model.safetensors"
    with (
        mock.patch("lilbee.catalog.resolve_pull_target", return_value=_entry(ref, fmt)),
        mock.patch("lilbee.catalog.pull_fit.chat_pull_fit") as fit,
        mock.patch("lilbee.catalog.download_model", return_value=dest) as download,
    ):
        path = ModelManager(models_dir, cfg, hf_client=make_hf_client(cfg)).pull(
            ref, ModelSource.NATIVE
        )
    assert path == dest
    assert download.call_args.args[1] is None
    fit.assert_not_called()


def test_the_no_engine_notice_arrives_after_the_download(tmp_path: Path) -> None:
    events: list[str] = []
    entry = _entry(SAFETENSORS_REF, ModelFormat.SAFETENSORS)
    with (
        mock.patch("lilbee.catalog.resolve_pull_target", return_value=entry),
        mock.patch(
            "lilbee.catalog.download_model",
            side_effect=lambda *_a, **_k: events.append("download") or tmp_path,
        ),
    ):
        ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg)).pull(
            SAFETENSORS_REF, ModelSource.NATIVE, on_notice=events.append
        )
    assert events == [
        "download",
        "No installed engine loads the safetensors format of acme/tiny-model#safetensors. "
        "Run lilbee engine list to see the installed engines.",
    ]


def test_a_failed_download_gives_no_engine_notice(tmp_path: Path) -> None:
    notices: list[str] = []
    entry = _entry(SAFETENSORS_REF, ModelFormat.SAFETENSORS)
    with (
        mock.patch("lilbee.catalog.resolve_pull_target", return_value=entry),
        mock.patch("lilbee.catalog.download_model", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg)).pull(
            SAFETENSORS_REF, ModelSource.NATIVE, on_notice=notices.append
        )
    assert notices == []


def test_a_gguf_pull_with_no_engine_installed_names_the_missing_engine(tmp_path: Path) -> None:
    with (
        mock.patch("lilbee.catalog.download_model") as download,
        pytest.raises(NoEngineError, match="No installed engine loads the gguf format"),
    ):
        ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg)).pull(
            GGUF_REF, ModelSource.NATIVE
        )
    download.assert_not_called()


def test_a_gguf_pull_that_allows_unsupported_still_needs_an_engine(tmp_path: Path) -> None:
    entry = _entry(GGUF_REF, ModelFormat.GGUF)
    with (
        mock.patch("lilbee.catalog.resolve_pull_target", return_value=entry),
        mock.patch("lilbee.catalog.download_model", return_value=tmp_path) as download,
        pytest.raises(NoEngineError, match="No installed engine loads the gguf format"),
    ):
        ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg)).pull(
            GGUF_REF, ModelSource.NATIVE, allow_unsupported=True
        )
    download.assert_not_called()


@pytest.mark.asyncio
async def test_the_pull_preflight_lets_a_model_with_no_engine_through(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg))
    with mock.patch(
        "lilbee.server.handlers.models.get_services",
        return_value=mock.MagicMock(model_manager=manager),
    ):
        await handlers.enforce_pull_arch_compat(SAFETENSORS_REF, source="native")
        await handlers.enforce_pull_arch_compat(MLX_REF, source="native")


@pytest.mark.asyncio
async def test_the_pull_preflight_refuses_a_gguf_with_no_engine(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg))
    with (
        mock.patch(
            "lilbee.server.handlers.models.get_services",
            return_value=mock.MagicMock(model_manager=manager),
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await handlers.enforce_pull_arch_compat(GGUF_REF, source="native")
    assert excinfo.value.status_code == 409
    assert "No installed engine loads the gguf format" in str(excinfo.value.detail)


def test_a_gguf_download_without_an_engine_is_refused(tmp_path: Path) -> None:
    from lilbee.catalog.download import fetch_model_files

    entry = _entry(GGUF_REF, ModelFormat.GGUF)
    with pytest.raises(RuntimeError, match="A GGUF download needs an engine"):
        fetch_model_files(entry, tmp_path, None, None)


def test_no_file_is_named_for_a_gguf_entry_without_an_engine(tmp_path: Path) -> None:
    entry = _entry(GGUF_REF, ModelFormat.GGUF)
    with mock.patch("lilbee.catalog.download.resolve_filename") as resolve:
        assert (
            ModelManager(tmp_path, cfg, hf_client=make_hf_client(cfg))._resolved_entry(
                entry, None, verify=True
            )
            is entry
        )
    resolve.assert_not_called()


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


async def _install(model: CatalogModel) -> tuple[mock.MagicMock, mock.MagicMock]:
    """Run the catalog's install step; return the enqueue mock and the notify mock."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        with (
            mock.patch.object(screen, "_enqueue_download") as enqueue,
            mock.patch.object(screen, "notify") as notify,
            mock.patch("lilbee.cli.tui.screens.catalog.disk_shortfall", return_value=None),
        ):
            screen._install_format(model)
            await wait_for_pull_check(screen.app)
        return enqueue, notify


async def test_the_catalog_enqueues_a_model_with_no_engine_installed() -> None:
    model = _entry(SAFETENSORS_REF, ModelFormat.SAFETENSORS)
    enqueue, notify = await _install(model)
    enqueue.assert_called_once_with(model)
    notify.assert_not_called()


async def test_the_catalog_refuses_a_gguf_with_no_engine_installed() -> None:
    model = _entry(GGUF_REF, ModelFormat.GGUF)
    enqueue, notify = await _install(model)
    enqueue.assert_not_called()
    assert "No installed engine loads the gguf format" in notify.call_args.args[0]
