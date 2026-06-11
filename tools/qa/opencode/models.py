"""Model matrix loading, HF manifest resolution, pulls, and per-cell cleanup."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from harness_config import _EMBED_REF, _MODEL_PULL_TIMEOUT_S, _SUSPENDED_SUFFIX, REPO_ROOT


def _models_manifests_dir() -> Path:
    """Locate lilbee's chat-manifests directory via cfg."""
    from lilbee.core.config import cfg

    return Path(cfg.models_dir) / "manifests"


def _list_chat_manifests() -> list[Path]:
    manifests_dir = _models_manifests_dir()
    if not manifests_dir.exists():
        return []
    chats: list[Path] = []
    for path in manifests_dir.rglob("*.gguf.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("task") == "chat":
            chats.append(path)
    return chats


def _ref_for_manifest(path: Path) -> str:
    """Convert a manifest path back to its canonical HF ref (subdir-aware)."""
    rel = path.relative_to(_models_manifests_dir())
    repo = rel.parts[0].replace("--", "/")
    rest = "/".join(rel.parts[1:]).removesuffix(".json")
    return f"{repo}/{rest}"


_PREWARM_CHUNK_BYTES = 64 * 1024 * 1024


def prewarm_model_blobs(ref: str) -> None:
    """Read the cell's GGUF shards once so the fleet's cold load hits page cache.

    On a network-volume models dir the first sequential read is the slow,
    fault-prone path; a giant read cold from the volume can outlast the
    fleet's health budget. Reading the shards here (outside any product
    timeout) leaves them in RAM, and the printed rate doubles as a volume
    health probe.
    """
    import time

    from lilbee.catalog.download import split_shard_filenames
    from lilbee.core.config import cfg
    from lilbee.modelhub.registry import ModelRegistry, parse_hf_ref

    first = ModelRegistry(Path(cfg.models_dir)).resolve(ref)
    _repo, filename = parse_hf_ref(ref)
    shard_names = [Path(s).name for s in split_shard_filenames(Path(filename).name)]
    total = 0
    start = time.monotonic()
    for name in shard_names:
        shard = first.parent / name
        if not shard.exists():
            continue
        with shard.open("rb") as f:
            while chunk := f.read(_PREWARM_CHUNK_BYTES):
                total += len(chunk)
    elapsed = max(time.monotonic() - start, 1e-6)
    print(f"prewarmed {total / 1e9:.1f} GB in {elapsed:.0f}s ({total / 1e6 / elapsed:.0f} MB/s)")


def _repo_of(model_ref: str) -> str:
    """The ``owner/repo`` prefix of a GGUF ref (used for HF-cache dir names)."""
    return "/".join(model_ref.split("/")[:2])


def is_ref_registered(ref: str) -> bool:
    """True when the registry lists exactly *ref* (manifest present + blob valid)."""
    return any(_ref_for_manifest(path) == ref for path in _list_chat_manifests())


def restore_suspended_manifests() -> int:
    """Heal ``*.qa-suspended`` manifests a crashed earlier run left behind.

    Older harness versions renamed competing chat manifests per cell (opencode
    used to boot on the first installed ref); a kill mid-cell left them
    suspended, so the registry under-reported installed models on the next
    run. The startup-model pin made suspension obsolete; this sweep remains so
    machines that ran the old harness recover. Returns the restore count.
    """
    restored = 0
    manifests_dir = _models_manifests_dir()
    if not manifests_dir.exists():
        return restored
    for path in manifests_dir.rglob(f"*{_SUSPENDED_SUFFIX}"):
        path.rename(path.with_name(path.name.removesuffix(_SUSPENDED_SUFFIX)))
        restored += 1
    return restored


@dataclass
class ModelCell:
    family: str
    ref: str
    size_gb: float
    skip: bool = False
    tier: str = "small"  # small | mid | giant -> picks the prompt from _TIER_PROMPTS


def load_models(path: Path) -> list[ModelCell]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return [
        ModelCell(
            family=raw["family"],
            ref=raw["ref"],
            size_gb=float(raw.get("size_gb", 0.0)),
            skip=bool(raw.get("skip", False)),
            tier=str(raw.get("tier", "small")),
        )
        for raw in data.get("model", [])
    ]


def ensure_embedding_model_pulled() -> None:
    """Idempotent: ensure the shared embedding model is in the registry.

    The matrix's per-cell `lilbee add` step requires the workspace's
    configured embedding model to be registered, otherwise indexing skips
    every fixture with "Model not found in registry" and `lilbee_search`
    comes up empty in opencode. Pulled by exact file ref: the per-cell
    config pins this quant, and a repo-level pull may install another.
    The registry is keyed off ``cfg.models_dir`` (global), so one pull at
    matrix start serves every cell -- no per-cell re-pull needed.
    """
    print(f"ensuring embedding model {_EMBED_REF} is registered")
    _run_pull_with_group_kill(_EMBED_REF)


_PULL_ATTEMPTS = 3
"""Pulls resume from partial downloads, so retries convert transient volume
I/O errors (network-FS Errno 5 on multi-GB shards) into incremental progress."""


def _run_pull_with_group_kill(pull_ref: str) -> None:
    """Run ``lilbee model pull`` in its own process group so a timeout reaps the
    full tree (otherwise ``uv``'s child python orphans and keeps the download
    running, contending for bandwidth with the next cell's pull).

    A non-zero exit is retried, then raised: a cell must never proceed past a
    failed pull (the launcher would serve zero models and the scenario would
    burn its full timeout against the client's fallback provider).
    Progress bars are suppressed so the matrix stdout stays grep-able.
    """
    import os
    import signal

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for attempt in range(1, _PULL_ATTEMPTS + 1):
        proc = subprocess.Popen(
            ["uv", "run", "lilbee", "model", "pull", pull_ref],
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=_MODEL_PULL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            raise
        if proc.returncode == 0:
            return
        print(f"pull of {pull_ref} exited {proc.returncode} (attempt {attempt}/{_PULL_ATTEMPTS})")
    raise RuntimeError(f"pull of {pull_ref} failed after {_PULL_ATTEMPTS} attempts")


def cleanup_cell_model(cell: ModelCell) -> None:
    """Delete the cell's chat GGUF from the HF cache + lilbee's manifest.

    Disk pressure on a dev laptop is the real limiter for an exhaustive
    sweep (qwen3-coder alone is 17 GB). Freeing the blob between cells
    keeps total disk usage flat at roughly the largest single model
    instead of the cumulative pull set.
    """
    from lilbee.core.config import cfg

    repo = _repo_of(cell.ref)
    # HF-cache directories ("models--Qwen--Qwen3-4B-GGUF") use the ``models--``
    # prefix; lilbee's manifests dir ("Qwen--Qwen3-4B-GGUF") does NOT. Cleaning
    # only the prefixed paths left stale manifests behind, which then convinced
    # the next pull that the model was cached and convinced lilbee's registry
    # the install was complete -- both inconsistent with the missing blob,
    # which surfaced to opencode as a 404 model_not_found at chat time.
    hf_cache_dir = "models--" + repo.replace("/", "--")
    manifest_dir = repo.replace("/", "--")
    models_root = Path(cfg.models_dir)
    for target in (
        models_root / hf_cache_dir,
        models_root / ".locks" / hf_cache_dir,
        models_root / "manifests" / manifest_dir,
    ):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
