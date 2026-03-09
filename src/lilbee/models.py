"""RAM detection, model selection, and auto-install for chat models."""

import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# RAM threshold for model selection (GB)
_RAM_THRESHOLD_GB = 16

# Model choices and approximate download sizes (GB)
_SMALL_MODEL = "qwen3:8b"
_SMALL_MODEL_SIZE_GB = 5
_LARGE_MODEL = "qwen3-coder:30b"
_LARGE_MODEL_SIZE_GB = 18

# Extra headroom required beyond model size (GB)
_DISK_HEADROOM_GB = 2


def get_system_ram_gb() -> float:
    """Return total system RAM in GB. Falls back to 8.0 if detection fails."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys / (1024**3)
        else:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
    except Exception:
        log.debug("RAM detection failed, falling back to 8.0 GB")
        return 8.0


def get_free_disk_gb(path: Path) -> float:
    """Return free disk space in GB for the filesystem containing *path*."""
    check_path = path if path.exists() else path.parent
    while not check_path.exists():
        check_path = check_path.parent
    usage = shutil.disk_usage(check_path)
    return usage.free / (1024**3)


def pick_default_model(ram_gb: float) -> str:
    """Choose a default chat model based on available RAM."""
    if ram_gb < _RAM_THRESHOLD_GB:
        return _SMALL_MODEL
    return _LARGE_MODEL


def _model_download_size_gb(model: str) -> float:
    """Estimated download size for a model."""
    sizes = {_SMALL_MODEL: _SMALL_MODEL_SIZE_GB, _LARGE_MODEL: _LARGE_MODEL_SIZE_GB}
    return sizes.get(model, _SMALL_MODEL_SIZE_GB)


def pull_with_progress(model: str) -> None:
    """Pull an Ollama model, showing a Rich progress bar on stderr."""
    import ollama
    from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        desc = f"Downloading model '{model}'..."
        ptask = progress.add_task(desc, total=None)
        for event in ollama.pull(model, stream=True):
            total = event.total or 0
            completed = event.completed or 0
            if total > 0:
                progress.update(ptask, total=total, completed=completed)
    sys.stderr.write(f"Model '{model}' ready.\n")


def ensure_chat_model() -> None:
    """If Ollama has no chat models installed, auto-pull one based on system RAM.

    Persists the chosen model in config.toml so it becomes the default.
    """
    import ollama

    import lilbee.config as cfg

    try:
        models = ollama.list()
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(f"Cannot connect to Ollama: {exc}. Is Ollama running?") from exc

    # Filter out embedding model — only check for chat models
    embed_base = cfg.EMBEDDING_MODEL.split(":")[0]
    chat_models = [
        m.model for m in models.models if m.model and m.model.split(":")[0] != embed_base
    ]
    if chat_models:
        return

    ram_gb = get_system_ram_gb()
    model = pick_default_model(ram_gb)
    required_gb = _model_download_size_gb(model) + _DISK_HEADROOM_GB
    free_gb = get_free_disk_gb(cfg.DATA_DIR)

    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough disk space to download '{model}': "
            f"need {required_gb:.1f} GB, have {free_gb:.1f} GB free. "
            f"Free up space or manually pull a smaller model with 'ollama pull <model>'."
        )

    sys.stderr.write(
        f"No chat model found. Auto-installing '{model}' (detected {ram_gb:.0f} GB RAM)...\n"
    )
    pull_with_progress(model)
    cfg.CHAT_MODEL = model

    from lilbee import settings

    settings.set_value("chat_model", model)
