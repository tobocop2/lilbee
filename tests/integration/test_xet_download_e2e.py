"""End-to-end acceptance for the xet download path, against real HuggingFace.

Covers:

  * every role downloads and lands on disk
  * a vision model brings its mmproj with it
  * every transfer really went over xet, not the HTTP fallback
  * asking twice for one model downloads it once
  * a cancelled download can be started again and finishes
  * four downloads run concurrently, and cancelling one stops only its bytes

Moves roughly 1.1GB, and `make test-integration` has no slow filter, so it is
off unless asked for:

    LILBEE_E2E_DOWNLOADS=1 uv run pytest tests/integration/test_xet_download_e2e.py -v -m slow
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.widgets import Static

from lilbee.catalog import CatalogModel, download_model
from lilbee.catalog.types import ModelTask
from lilbee.cli.tui.task_queue import TERMINAL_STATUSES, TaskStatus
from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
from tests._lilbee_app_test_host import LilbeeAppHost

pytestmark = [
    pytest.mark.slow,
    # Every test here pulls hundreds of MB, well past the project's 60s default.
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        not os.environ.get("LILBEE_E2E_DOWNLOADS"),
        reason="moves ~1.1GB; set LILBEE_E2E_DOWNLOADS=1 to run",
    ),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="lilbee routes Windows downloads to plain HTTP (xet stalls there)",
    ),
]


def _entry(hf_repo: str, gguf: str, task: str, size_gb: float) -> CatalogModel:
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename=gguf,
        size_gb=size_gb,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
    )


# Pinned rather than taken from the picks: the picks are whatever is trending
# and can be enormous. Each is xet-backed, and SmolVLM ships an mmproj.
CHAT = _entry("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", ModelTask.CHAT, 0.4)
EMBED = _entry(
    "nomic-ai/nomic-embed-text-v1.5-GGUF",
    "nomic-embed-text-v1.5.Q4_K_M.gguf",
    ModelTask.EMBEDDING,
    0.1,
)
RERANK = _entry(
    "gpustack/bge-reranker-v2-m3-GGUF", "bge-reranker-v2-m3-Q2_K.gguf", ModelTask.RERANK, 0.3
)
VISION = _entry(
    "ggml-org/SmolVLM-256M-Instruct-GGUF",
    "SmolVLM-256M-Instruct-Q8_0.gguf",
    ModelTask.VISION,
    0.3,
)

ROLES = [
    pytest.param(CHAT, id="chat"),
    pytest.param(EMBED, id="embed"),
    pytest.param(RERANK, id="rerank"),
    pytest.param(VISION, id="vision"),
]


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A models dir of its own, so every pull in this module is a real one."""
    from lilbee.core.config.model import cfg

    target = tmp_path / "models"
    target.mkdir()
    monkeypatch.setattr(cfg, "models_dir", target)
    return target


@pytest.fixture
def xet_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every file huggingface_hub actually fetched over xet.

    Availability is not use: HF_HUB_DISABLE_XET, a non-xet repo or a metadata
    gap all silently route to plain HTTP, and the download still succeeds. Only
    counting xet_get proves which transport ran.
    """
    import huggingface_hub.file_download as fd

    seen: list[str] = []
    real = fd.xet_get

    def _spy(**kwargs: object) -> object:
        seen.append(str(kwargs.get("displayed_filename") or kwargs.get("incomplete_path")))
        return real(**kwargs)  # ty: ignore[missing-argument]

    monkeypatch.setattr(fd, "xet_get", _spy)
    return seen


def _repo_root(models_dir: Path, hf_repo: str) -> Path:
    """Cache folder holding *hf_repo*, whether or not it exists yet."""
    from huggingface_hub.file_download import repo_folder_name

    return models_dir / repo_folder_name(repo_id=hf_repo, repo_type="model")


def _bytes_in_flight(models_dir: Path, hf_repo: str | None = None) -> int:
    """Bytes sitting in partial blobs, which only grow while a transfer runs.

    Scoped to one repo when *hf_repo* is given, so a cancelled model can be
    measured while the next queued download is writing.
    """
    root = models_dir
    if hf_repo is not None:
        root = _repo_root(models_dir, hf_repo)
        if not root.is_dir():
            return 0
    return sum(p.stat().st_size for p in root.rglob("*.incomplete"))


def _gguf_names(models_dir: Path) -> set[str]:
    return {p.name for p in models_dir.rglob("*.gguf")}


def _repo_gguf_count(models_dir: Path, hf_repo: str) -> int:
    """GGUFs landed for *hf_repo*, identified by repo rather than filename.

    The requested quant is not the one that necessarily arrives: resolution
    resolves to the best available, so asserting on the filename asserts a
    choice the catalog is free to make.
    """
    root = _repo_root(models_dir, hf_repo)
    return len(list(root.rglob("*.gguf"))) if root.is_dir() else 0


@pytest.mark.parametrize("entry", ROLES)
def test_every_role_downloads_over_xet(
    entry: CatalogModel, models_dir: Path, xet_calls: list[str]
) -> None:
    path = download_model(entry)

    assert path.exists(), f"{entry.hf_repo} reported success but nothing is on disk"
    assert path.stat().st_size > 0
    assert xet_calls, f"{entry.hf_repo} downloaded over HTTP, not xet"


def test_vision_model_brings_its_projector(models_dir: Path, xet_calls: list[str]) -> None:
    """The mmproj is a second file on a separate path, and a vision model is
    unusable without it: the role dies at plan time with a warning no re-pull
    cures, so 'the GGUF arrived' is not enough."""
    download_model(VISION)

    names = _gguf_names(models_dir)
    assert any("mmproj" in n.lower() for n in names), f"no projector among {sorted(names)}"
    assert any("mmproj" not in n.lower() for n in names), "projector but no model"
    assert len(xet_calls) >= 2, f"expected both files over xet, saw {xet_calls}"


class _Host(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


async def _await_terminal(
    pilot: Pilot[None], controller: TaskBarController, task_id: str, timeout: float = 1500.0
) -> TaskStatus:
    """Wait by yielding to Textual, never by sleeping.

    The worker marshals completion back with app.call_from_thread, so a blocking
    sleep here starves the loop that has to drain it and the task never leaves
    ACTIVE however long the download took.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = controller.queue.get_task(task_id)
        if task is not None and task.status in TERMINAL_STATUSES:
            return task.status
        await pilot.pause()
        await asyncio.sleep(0.2)
    raise AssertionError(f"task {task_id} never finished")


async def _await_active(pilot: Pilot[None], controller: TaskBarController, task_id: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        task = controller.queue.get_task(task_id)
        if task is not None and task.status is not TaskStatus.QUEUED:
            return
        await pilot.pause()
        await asyncio.sleep(0.1)
    raise AssertionError("download never started")


async def test_asking_twice_downloads_once(models_dir: Path) -> None:
    """Two install presses on one model must not queue it twice.

    The transfer runs in a child process, so the in-process xet spy cannot
    count it; one task means one child means one transfer.
    """
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        first = controller.start_download(EMBED)
        second = controller.start_download(EMBED)

        assert first == second
        assert len(controller.queue.active_tasks) + len(controller.queue.queued_tasks) == 1

        assert await _await_terminal(pilot, controller, first) is TaskStatus.DONE
        assert _gguf_names(models_dir)


async def test_a_cancelled_download_can_be_started_again(models_dir: Path) -> None:
    """Cancelling must not poison the model: dedupe spans queued and active work
    only, so a terminal task has to leave the way clear for a retry."""
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)

        first = controller.start_download(CHAT)
        await _await_active(pilot, controller, first)
        # Through the controller, which is what every cancel key binding calls.
        controller.cancel_task(first)
        assert await _await_terminal(pilot, controller, first) is TaskStatus.CANCELLED

        # The row saying cancelled is not the claim; the bytes stopping is.
        # The worker terminates the download's child process, and the transfer
        # dies with it.
        settled = _bytes_in_flight(models_dir)
        await asyncio.sleep(5)
        assert _bytes_in_flight(models_dir) <= settled, "cancelled download kept transferring"

        second = controller.start_download(CHAT)
        assert second != first, "the cancelled task was handed back instead of a new one"
        assert await _await_terminal(pilot, controller, second) is TaskStatus.DONE
        assert any("Qwen3-0.6B" in n for n in _gguf_names(models_dir))


async def test_cancelling_one_of_four_concurrent_downloads_leaves_the_rest_running(
    models_dir: Path,
) -> None:
    """Four models run at once; cancelling one stops only its bytes.

    Each download owns a child process, so the cancelled one's terminate must
    not touch its siblings, and the freed slot must take a new request.
    """
    entries = (CHAT, EMBED, RERANK, VISION)
    app = _Host()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        first, second, third, fourth = (controller.start_download(m) for m in entries)

        for task_id in (first, second, third, fourth):
            await _await_active(pilot, controller, task_id)
        assert len(controller.queue.active_tasks) == 4, "downloads must run concurrently"

        controller.cancel_task(first)
        assert await _await_terminal(pilot, controller, first) is TaskStatus.CANCELLED

        # Scoped to the cancelled repo: the three live downloads are writing
        # too. Must not grow. A cancel that discards its partial drops to
        # zero, which is also a stop.
        settled = _bytes_in_flight(models_dir, CHAT.hf_repo)
        await asyncio.sleep(5)
        assert _bytes_in_flight(models_dir, CHAT.hf_repo) <= settled, (
            "cancelled model kept transferring"
        )

        # Terminating the cancelled child must not have taken its siblings.
        assert await _await_terminal(pilot, controller, second) is TaskStatus.DONE
        assert await _await_terminal(pilot, controller, third) is TaskStatus.DONE
        assert await _await_terminal(pilot, controller, fourth) is TaskStatus.DONE

        again = controller.start_download(CHAT)
        assert again not in (first, second, third, fourth)
        assert await _await_terminal(pilot, controller, again) is TaskStatus.DONE

        for entry in entries:
            assert _repo_gguf_count(models_dir, entry.hf_repo), f"{entry.hf_repo} produced no GGUF"
