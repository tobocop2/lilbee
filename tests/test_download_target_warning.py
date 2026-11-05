"""A pull warning from the task bar's download reaches the user as a warning toast."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.app.models import PullResult, PullStatus
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

_WARNING = "Could not check acme/vl-GGUF for a vision projector. Run the pull again."


def _model() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/vl-GGUF",
        gguf_filename="model.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


@pytest.mark.parametrize(
    ("warning", "expected"),
    [(_WARNING, [(_WARNING, "warning")]), (None, [])],
)
async def test_download_shows_the_pull_warning_as_a_warning_toast(
    monkeypatch: pytest.MonkeyPatch, warning: str | None, expected: list[tuple[str, str]]
) -> None:
    result = PullResult(
        model="acme/vl-GGUF", source="native", status=PullStatus.OK, warning=warning
    )
    monkeypatch.setattr("lilbee.app.models.pull_model_data", lambda *a, **kw: result)
    async with _App().run_test(size=(120, 40)) as pilot:
        toasts: list[tuple[str, str]] = []
        monkeypatch.setattr(
            pilot.app,
            "notify",
            lambda message, *, severity="information", **_kw: toasts.append(
                (str(message), severity)
            ),
        )
        pilot.app.task_bar.start_download(_model())
        await wait_until(pilot, lambda: not pilot.app.task_bar.queue.active_tasks, max_pauses=100)
        await wait_until(pilot, lambda: bool(toasts), max_pauses=20)
        assert toasts == expected
