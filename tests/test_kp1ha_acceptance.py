"""Acceptance for bb-kp1ha: the TUI pre-pull check and the real download agree.

Each case builds the same model cache twice. The TUI install decision runs on
one copy and the real download (``download_model``, which the suite runs in
process) runs on the other. Both talk to a fake HuggingFace Hub over HTTP, so
no lilbee function is patched: the cases hold for any internal shape of the fix.
The volume is simulated by patching ``shutil.disk_usage``: free space is a fixed
capacity minus the bytes on disk under the cache, so a deleted partial frees
space exactly as on a real disk.

Copy this file into the lilbee tests folder and run it with pytest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.tui.task_queue import TaskQueue
from tests._async_wait import wait_for_pull_check
from tests._lilbee_app_test_host import LilbeeAppHost

COMMIT = "a" * 40
INDEX = "model.safetensors.index.json"


def _blob(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class FakeRepo:
    files: dict[str, bytes]
    gated: bool = False


@dataclass
class FakeHub:
    repos: dict[str, FakeRepo] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)


class _HubHandler(BaseHTTPRequestHandler):
    hub: FakeHub

    def log_message(self, *_args: object) -> None:
        return

    def _send(self, code: int, headers: dict[str, str], body: bytes, head: bool) -> None:
        self.send_response(code)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head: bool) -> None:
        path = unquote(self.path.split("?", 1)[0])
        self.hub.requests.append(f"{self.command} {path}")
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "models"] and len(parts) >= 4:
            self._listing("/".join(parts[2:4]), head)
        elif len(parts) >= 5 and parts[2] == "resolve":
            self._file("/".join(parts[:2]), "/".join(parts[4:]), head)
        else:
            self._send(404, {}, b"", head)

    def _listing(self, repo_id: str, head: bool) -> None:
        repo = self.hub.repos.get(repo_id)
        if repo is None:
            self._send(404, {"X-Error-Code": "RepoNotFound"}, b"{}", head)
            return
        info = {
            "id": repo_id,
            "modelId": repo_id,
            "sha": COMMIT,
            "private": False,
            "gated": "manual" if repo.gated else False,
            "tags": [],
            "siblings": [{"rfilename": name} for name in sorted(repo.files)],
        }
        body = json.dumps(info).encode()
        self._send(200, {"Content-Type": "application/json"}, body, head)

    def _file(self, repo_id: str, name: str, head: bool) -> None:
        repo = self.hub.repos.get(repo_id)
        if repo is None or name not in repo.files:
            self._send(404, {"X-Error-Code": "EntryNotFound"}, b"", head)
            return
        if repo.gated:
            headers = {
                "X-Error-Code": "GatedRepo",
                "X-Error-Message": "Access to this repo is restricted.",
            }
            self._send(401, headers, b"", head)
            return
        content = repo.files[name]
        headers = {
            "ETag": f'"{_blob(content)}"',
            "X-Repo-Commit": COMMIT,
            "Content-Length": str(len(content)),
            "Content-Type": "application/octet-stream",
        }
        self._send(200, headers, content, head)

    def do_HEAD(self) -> None:
        self._route(head=True)

    def do_GET(self) -> None:
        self._route(head=False)


def _handler(hub: FakeHub) -> type[BaseHTTPRequestHandler]:
    return type("Handler", (_HubHandler,), {"hub": hub})


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    from huggingface_hub import constants

    state = FakeHub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(constants, "ENDPOINT", endpoint)
    monkeypatch.setattr(
        constants,
        "HUGGINGFACE_CO_URL_TEMPLATE",
        endpoint + "/{repo_id}/resolve/{revision}/{filename}",
    )
    # The suite runs with hub offline mode on; the fake hub is the network here.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
    yield state
    server.shutdown()
    server.server_close()


def _bytes_under(root: Path) -> int:
    total = 0
    for folder, _dirs, files in os.walk(root):
        for name in files:
            info = os.lstat(os.path.join(folder, name))
            if not stat.S_ISLNK(info.st_mode):
                total += info.st_size
    return total


class Volume:
    """Two caches, each on its own simulated volume of a fixed capacity."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.capacity: dict[Path, int] = {}
        real = shutil.disk_usage

        def _usage(path: object) -> object:
            probe = Path(os.fspath(path)).resolve()
            for root, capacity in self.capacity.items():
                if probe == root or root in probe.parents:
                    used = _bytes_under(root)
                    return shutil._ntuple_diskusage(capacity, used, max(capacity - used, 0))
            return real(path)

        monkeypatch.setattr(shutil, "disk_usage", _usage)

    def mount(self, root: Path, free_bytes: int) -> None:
        """Make *root* a volume with *free_bytes* free in its current state."""
        root.mkdir(parents=True, exist_ok=True)
        self.capacity[root.resolve()] = _bytes_under(root) + free_bytes


@pytest.fixture
def volume(monkeypatch: pytest.MonkeyPatch) -> Volume:
    return Volume(monkeypatch)


def _repo_dir(root: Path, hf_repo: str) -> Path:
    return root / ("models--" + hf_repo.replace("/", "--"))


def _plant_partial(root: Path, hf_repo: str, blob: str, size: int) -> Path:
    blobs = _repo_dir(root, hf_repo) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    partial = blobs / f"{blob}.4f2a91cc0d.incomplete"
    partial.write_bytes(b"p" * size)
    return partial


def _entry(hf_repo: str, fmt: ModelFormat, filename: str, size_gb: float) -> CatalogModel:
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename=filename,
        size_gb=size_gb,
        min_ram_gb=1.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        format=fmt,
        formats=frozenset({fmt}),
    )


# A row estimate far above any real file here, for a pull that must go ahead:
# a check that charges the row estimate instead of the real files refuses it.
HUGE_GB = 50.0
# A row estimate far below any real file here, for a pull that must be refused:
# a check that charges the row estimate instead of the real files lets it through.
TINY_GB = 1e-12


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


@dataclass
class TuiOutcome:
    enqueued: list[str]
    notices: list[str]


async def _tui_install(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path, model: CatalogModel
) -> TuiOutcome:
    """Press install on *model* in the catalog screen and record what happened."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "models_dir", models_dir)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()
        outcome = TuiOutcome([], [])
        monkeypatch.setattr(
            screen, "_enqueue_download", lambda m: outcome.enqueued.append(m.hf_repo)
        )
        monkeypatch.setattr(
            screen, "notify", lambda message, **_kw: outcome.notices.append(str(message))
        )
        screen._install_model(model)
        await wait_for_pull_check(screen.app)
        return outcome


def _download(monkeypatch: pytest.MonkeyPatch, models_dir: Path, model: CatalogModel) -> object:
    """Run the real pull of *model* into *models_dir*; return its path or the exception."""
    from lilbee.app.services import get_services
    from lilbee.catalog.download import download_model
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "models_dir", models_dir)
    engine = None
    if model.format is ModelFormat.GGUF:
        engine = get_services().engines.for_ref(model.pull_ref)
    try:
        return download_model(model, engine)
    except Exception as exc:  # the outcome under test
        return exc


def _assert_enqueued(outcome: TuiOutcome, model: CatalogModel) -> None:
    assert outcome.notices == [], f"the TUI refused a pull the download completes: {outcome}"
    assert outcome.enqueued == [model.hf_repo]


def _assert_refused(outcome: TuiOutcome, text: str) -> None:
    assert outcome.enqueued == [], f"the TUI enqueued a pull the download refuses: {outcome}"
    assert any(text in notice for notice in outcome.notices), outcome.notices


# Repo contents shared by several cases.
GGUF_REPO = "acme/tiny-GGUF"
GGUF_FILE = "tiny-Q4_K_M.gguf"
GGUF_BYTES = b"G" * 5000
PROJECTOR = "mmproj-tiny-f16.gguf"
PROJECTOR_BYTES = b"M" * 3000
MLX_REPO = "acme/tiny-mlx"
MLX_FILES = {
    "model.safetensors": b"W" * 4000,
    "config.json": b'{"model_type": "llama"}',
    "tokenizer.json": b"{}" * 100,
}
ST_REPO = "acme/tiny-sharded"
ST_SHARDS = {
    "model-00001-of-00002.safetensors": b"A" * 3000,
    "model-00002-of-00002.safetensors": b"B" * 3000,
}


def _index(shards: list[str]) -> bytes:
    return json.dumps(
        {"metadata": {}, "weight_map": {f"w{i}": s for i, s in enumerate(shards)}}
    ).encode()


def _sharded_files(declared: list[str]) -> dict[str, bytes]:
    return {**ST_SHARDS, INDEX: _index(declared), "config.json": b'{"model_type": "llama"}'}


def _total(files: dict[str, bytes]) -> int:
    return sum(len(content) for content in files.values())


# The hub writes refs/main (40 bytes) and CACHEDIR.TAG (191 bytes) when the first
# file lands. A pull that fetches one file before its space check (the safetensors
# index) needs that much room on top of its files.
REFS_SLACK = 512
SPARE = pytest.mark.parametrize("spare", [REFS_SLACK, -1], ids=["fits", "one-byte-short"])


# ---- 1. A large partial that is reclaimed, so the pull goes ahead ---------------


@pytest.mark.parametrize("fmt", [ModelFormat.GGUF, ModelFormat.MLX])
async def test_a_large_reclaimable_partial_lets_the_pull_go_ahead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume, fmt: ModelFormat
) -> None:
    """The leftover partial of the very file the pull fetches is reclaimed before measuring.

    Free space before the reclaim is 100 bytes; after it, the pull fits with 100 to spare.
    """
    if fmt is ModelFormat.GGUF:
        repo, files, filename = GGUF_REPO, {GGUF_FILE: GGUF_BYTES}, GGUF_FILE
        partial_of = GGUF_BYTES
    else:
        repo, files, filename = MLX_REPO, MLX_FILES, "model.safetensors"
        partial_of = MLX_FILES["model.safetensors"]
    hub.repos[repo] = FakeRepo(files)
    model = _entry(repo, fmt, filename, HUGE_GB)
    needed = _total(files)
    partial_size = needed + 9000
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        _plant_partial(root, repo, _blob(partial_of), partial_size)
        volume.mount(root, free_bytes=100)
    # The partial is gone after the reclaim, so free space is 100 + partial_size >= needed.

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, Path), f"the download itself refused: {result!r}"
    _assert_enqueued(outcome, model)


async def test_a_pull_too_big_even_after_reclaim_is_refused_by_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume
) -> None:
    """The reclaim must not turn into a pass: one byte short after the reclaim is refused."""
    hub.repos[GGUF_REPO] = FakeRepo({GGUF_FILE: GGUF_BYTES})
    model = _entry(GGUF_REPO, ModelFormat.GGUF, GGUF_FILE, TINY_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        _plant_partial(root, GGUF_REPO, _blob(GGUF_BYTES), 2000)
        volume.mount(root, free_bytes=len(GGUF_BYTES) - 2000 - 1)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, RuntimeError) and "Not enough disk space" in str(result), result
    _assert_refused(outcome, "Not enough disk space")


# ---- 2. An XDNA row: an engine tag, nothing from the hub ------------------------


async def test_an_xdna_row_installs_with_no_hub_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume
) -> None:
    """An XDNA pull writes a ticket; its name is an engine tag, never a hub repo id."""
    model = _entry("qwen3:0.6b", ModelFormat.XDNA, "", 0.5)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=10**12)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, Path), f"the XDNA download itself failed: {result!r}"
    _assert_enqueued(outcome, model)
    assert hub.requests == [], f"an XDNA pull asked the hub: {hub.requests}"


# ---- 3. An empty selection --------------------------------------------------------


async def test_a_repo_with_no_files_of_the_format_is_refused_by_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume
) -> None:
    """The format fetch refuses a repo that holds no files of the format; the TUI must too."""
    hub.repos["acme/no-weights"] = FakeRepo({"README.md": b"# nothing here"})
    model = _entry("acme/no-weights", ModelFormat.MLX, "model.safetensors", HUGE_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=10**12)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, RuntimeError) and "No mlx model files found" in str(result), result
    _assert_refused(outcome, "No mlx model files found")


# ---- 4. A safetensors index ----------------------------------------------------------


@SPARE
async def test_a_sharded_pull_charges_the_index_and_every_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume, spare: int
) -> None:
    """The TUI charges the same file set the download fetches: index, shards and config."""
    files = _sharded_files(sorted(ST_SHARDS))
    hub.repos[ST_REPO] = FakeRepo(files)
    model = _entry(ST_REPO, ModelFormat.SAFETENSORS, INDEX, HUGE_GB if spare >= 0 else TINY_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=_total(files) + spare)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    if spare >= 0:
        on_disk = sorted(
            (str(p.relative_to(roots[1])), p.lstat().st_size) for p in roots[1].rglob("*")
        )
        assert isinstance(result, Path), (
            f"the download itself refused: {result!r}; total={_total(files)} on disk={on_disk}"
        )
        _assert_enqueued(outcome, model)
    else:
        assert isinstance(result, RuntimeError) and "Not enough disk space" in str(result), result
        _assert_refused(outcome, "Not enough disk space")


async def test_an_index_that_declares_a_missing_shard_is_refused_by_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume
) -> None:
    """The download reads the index and refuses a shard set it declares but the repo lacks."""
    declared = [*sorted(ST_SHARDS), "extra-weights.safetensors"]
    hub.repos[ST_REPO] = FakeRepo(_sharded_files(declared))
    model = _entry(ST_REPO, ModelFormat.SAFETENSORS, INDEX, HUGE_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=10**12)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, ValueError) and "extra-weights.safetensors" in str(result), result
    _assert_refused(outcome, "extra-weights.safetensors")


# ---- 5. A gated repo -------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [ModelFormat.GGUF, ModelFormat.MLX])
async def test_a_gated_repo_shows_the_login_message_and_enqueues_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume, fmt: ModelFormat
) -> None:
    """The hub lists a gated repo but refuses its files; the download names the login step.

    A task that fails at once is terminal, so the TUI refuses before the enqueue, as it
    does for a disk shortfall. No partial of the repo is touched.
    """
    if fmt is ModelFormat.GGUF:
        repo, files, filename = "acme/gated-GGUF", {GGUF_FILE: GGUF_BYTES}, GGUF_FILE
    else:
        repo, files, filename = "acme/gated-mlx", MLX_FILES, "model.safetensors"
    hub.repos[repo] = FakeRepo(files, gated=True)
    model = _entry(repo, fmt, filename, TINY_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    partials = []
    for root in roots:
        partials.append(_plant_partial(root, repo, "0" * 64, 500))
        volume.mount(root, free_bytes=10**12)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, PermissionError), result
    assert "requires HuggingFace authentication" in str(result)
    _assert_refused(outcome, "requires HuggingFace authentication")
    assert all(partial.exists() for partial in partials)


# ---- 6. No partial ------------------------------------------------------------------


@SPARE
@pytest.mark.parametrize("fmt", [ModelFormat.GGUF, ModelFormat.MLX])
async def test_with_no_partial_the_tui_charges_exactly_the_download_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hub: FakeHub,
    volume: Volume,
    fmt: ModelFormat,
    spare: int,
) -> None:
    """No partial of this pull: both sides agree at the byte boundary of the real files.

    A partial of another quant of the same repo sits in the cache. It is live
    for all the TUI knows, so the TUI decision never deletes it.
    """
    if fmt is ModelFormat.GGUF:
        repo, files, filename = GGUF_REPO, {GGUF_FILE: GGUF_BYTES}, GGUF_FILE
    else:
        repo, files, filename = MLX_REPO, MLX_FILES, "model.safetensors"
    hub.repos[repo] = FakeRepo(files)
    model = _entry(repo, fmt, filename, HUGE_GB if spare >= 0 else TINY_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    other = []
    for root in roots:
        other.append(_plant_partial(root, repo, "f" * 64, 700))
        volume.mount(root, free_bytes=_total(files) + spare)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert other[0].exists(), "the TUI deleted the partial of a file this pull does not fetch"
    if spare >= 0:
        assert isinstance(result, Path), f"the download itself refused: {result!r}"
        _assert_enqueued(outcome, model)
    else:
        assert isinstance(result, RuntimeError) and "Not enough disk space" in str(result), result
        _assert_refused(outcome, "Not enough disk space")


# ---- 7. Hub unreachable: the row estimate is the only figure -----------------------


async def test_an_unreachable_hub_still_enqueues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, volume: Volume
) -> None:
    """Offline, the TUI does not refuse on a figure it cannot get and deletes nothing."""
    from huggingface_hub import constants

    dead = "http://127.0.0.1:9"
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "ENDPOINT", dead)
    monkeypatch.setattr(
        constants, "HUGGINGFACE_CO_URL_TEMPLATE", dead + "/{repo_id}/resolve/{revision}/{filename}"
    )
    model = _entry(MLX_REPO, ModelFormat.MLX, "model.safetensors", 0.000001)
    root = tmp_path / "tui"
    partial = _plant_partial(root, MLX_REPO, "e" * 64, 300)
    volume.mount(root, free_bytes=10**9)

    outcome = await _tui_install(monkeypatch, root, model)

    _assert_enqueued(outcome, model)
    assert partial.exists()


# ---- 8. The projector a vision pull fetches -----------------------------------------


async def test_a_vision_pull_charges_its_projector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: FakeHub, volume: Volume
) -> None:
    """A GGUF pull fetches the projector the repo lists, so the TUI charges it too.

    Room for the model but not the projector: the download refuses at the
    projector, so the TUI must refuse before the enqueue.
    """
    files = {GGUF_FILE: GGUF_BYTES, PROJECTOR: PROJECTOR_BYTES}
    hub.repos["acme/tiny-VL-GGUF"] = FakeRepo(files)
    model = _entry("acme/tiny-VL-GGUF", ModelFormat.GGUF, GGUF_FILE, TINY_GB)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=_total(files) - 1)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    assert isinstance(result, RuntimeError) and "Not enough disk space" in str(result), result
    _assert_refused(outcome, "Not enough disk space")


# ---- 9. A wildcard row whose top file the engine cannot load -----------------------


TOP_QUANT = "tiny-Q4_K_M.gguf"
NEXT_QUANT = "tiny-Q8_0.gguf"


@pytest.mark.parametrize("loadable", [{NEXT_QUANT}, set()], ids=["next-file", "no-file"])
async def test_a_wildcard_row_skips_the_files_the_engine_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hub: FakeHub,
    volume: Volume,
    loadable: set[str],
) -> None:
    """Both sides ask the engine which file it can load, so both pick the same file.

    The engine's verdict is the one input set here, on the engine itself: the
    fake hub cannot serve a real GGUF header. The top-ranked file is too big for
    the disk, so a check that charges it refuses a pull the download completes.
    When the engine can load no file, the download fails at once, and so must
    the TUI, with the download's message.
    """
    from lilbee.app.services import get_services
    from lilbee.catalog.compat import UnsupportedQuantError
    from lilbee.catalog.refs import GGUF_GLOB

    repo = "acme/two-quants-GGUF"
    files = {TOP_QUANT: b"T" * 9000, NEXT_QUANT: b"N" * 2000}
    hub.repos[repo] = FakeRepo(files)
    model = _entry(repo, ModelFormat.GGUF, GGUF_GLOB, HUGE_GB)
    engine = get_services().engines.for_ref(model.pull_ref)

    def _assert_loadable(_self: object, hf_repo: str, filename: str, _token: object) -> None:
        if filename not in loadable:
            raise UnsupportedQuantError(f"{hf_repo}/{filename}", "IQ1_S")

    monkeypatch.setattr(type(engine), "assert_loadable", _assert_loadable)
    roots = (tmp_path / "tui", tmp_path / "dl")
    for root in roots:
        volume.mount(root, free_bytes=len(files[NEXT_QUANT]) + REFS_SLACK)

    outcome = await _tui_install(monkeypatch, roots[0], model)
    result = _download(monkeypatch, roots[1], model)

    if loadable:
        assert isinstance(result, Path) and result.name == NEXT_QUANT, result
        _assert_enqueued(outcome, model)
    else:
        assert isinstance(result, UnsupportedQuantError), result
        _assert_refused(outcome, str(result))


def test_the_cases_are_independent_of_the_row_estimate() -> None:
    """Guard on the fixtures: the estimates sit far from every real file size."""
    assert HUGE_GB * 1024**3 > 10**6
    assert TINY_GB * 1024**3 < 1
    assert replace(_entry(GGUF_REPO, ModelFormat.GGUF, GGUF_FILE, 1.0), size_gb=2.0).size_gb == 2.0
