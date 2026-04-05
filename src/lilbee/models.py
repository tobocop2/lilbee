"""RAM detection, model selection, interactive picker, and auto-install for chat models."""

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lilbee import settings
from lilbee.config import cfg


class ModelTask(StrEnum):
    """Task classification for models."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"


log = logging.getLogger(__name__)

# Extra headroom required beyond model size (GB)
_DISK_HEADROOM_GB = 2

MODELS_BROWSE_URL = "https://huggingface.co/models?library=gguf&sort=trending"


def ensure_tag(name: str) -> str:
    """Ensure a model name has an explicit tag (e.g. ``llama3`` → ``llama3:latest``)."""
    if not name or ":" in name:
        return name
    return f"{name}:latest"


@dataclass(frozen=True)
class ModelInfo:
    """A curated chat model with metadata for the picker UI."""

    name: str
    size_gb: float
    min_ram_gb: float
    description: str


def _catalog_from_featured(featured: tuple) -> tuple[ModelInfo, ...]:
    """Build a ModelInfo tuple from catalog.py's CatalogModel entries."""
    return tuple(ModelInfo(m.name, m.size_gb, m.min_ram_gb, m.description) for m in featured)


# Lazy singletons — resolved on first access to break the circular import
# between models.py (imports ModelTask) and catalog.py (imports from models).
_model_catalog: tuple[ModelInfo, ...] | None = None
_vision_catalog: tuple[ModelInfo, ...] | None = None


def _get_model_catalog() -> tuple[ModelInfo, ...]:
    global _model_catalog
    if _model_catalog is None:
        from lilbee.catalog import FEATURED_CHAT

        _model_catalog = _catalog_from_featured(FEATURED_CHAT)
    return _model_catalog


def _get_vision_catalog() -> tuple[ModelInfo, ...]:
    global _vision_catalog
    if _vision_catalog is None:
        from lilbee.catalog import FEATURED_VISION

        _vision_catalog = _catalog_from_featured(FEATURED_VISION)
    return _vision_catalog


def __getattr__(name: str) -> tuple[ModelInfo, ...]:
    if name == "MODEL_CATALOG":
        return _get_model_catalog()
    if name == "VISION_CATALOG":
        return _get_vision_catalog()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
    except (OSError, AttributeError, ValueError):
        log.debug("RAM detection failed, falling back to 8.0 GB")
        return 8.0


def get_free_disk_gb(path: Path) -> float:
    """Return free disk space in GB for the filesystem containing *path*."""
    check_path = path if path.exists() else path.parent
    while not check_path.exists():
        check_path = check_path.parent
    usage = shutil.disk_usage(check_path)
    return usage.free / (1024**3)


def pick_default_model(ram_gb: float) -> ModelInfo:
    """Choose the largest catalog model that fits in *ram_gb*."""
    best = _get_model_catalog()[0]
    for model in _get_model_catalog():
        if model.min_ram_gb <= ram_gb:
            best = model
    return best


def _model_download_size_gb(model: str) -> float:
    """Estimated download size for a model."""
    catalog_sizes = {m.name: m.size_gb for m in _get_model_catalog()}
    fallback = 5.0  # reasonable default for unknown models
    return catalog_sizes.get(model, fallback)


def display_model_picker(
    ram_gb: float, free_disk_gb: float, *, console: Console | None = None
) -> ModelInfo:
    """Show a Rich table of catalog models and return the recommended model."""
    console = console or Console(stderr=True)
    recommended = pick_default_model(ram_gb)

    table = Table(title="Available Models", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Description")

    for idx, model in enumerate(_get_model_catalog(), 1):
        num_str = str(idx)
        name = model.name
        size_str = f"{model.size_gb:.1f} GB"
        desc = model.description

        is_recommended = model == recommended
        disk_too_small = free_disk_gb < model.size_gb + _DISK_HEADROOM_GB

        if is_recommended:
            name = f"[bold]{name} ★[/bold]"
            desc = f"[bold]{desc}[/bold]"
            num_str = f"[bold]{num_str}[/bold]"

        if disk_too_small:
            size_str = f"[red]{model.size_gb:.1f} GB[/red]"

        table.add_row(num_str, name, size_str, desc)

    console.print()
    console.print("[bold]No chat model found.[/bold] Pick one to download:\n")
    console.print(table)
    console.print(f"\n  System: {ram_gb:.0f} GB RAM, {free_disk_gb:.1f} GB free disk")
    console.print("  \u2605 = recommended for your system")
    console.print(f"  Browse more models at {MODELS_BROWSE_URL}\n")

    return recommended


def pick_default_vision_model() -> ModelInfo:
    """Return the recommended vision model (first catalog entry, best quality)."""
    return _get_vision_catalog()[0]


def display_vision_picker(
    ram_gb: float, free_disk_gb: float, *, console: Console | None = None
) -> ModelInfo:
    """Show a Rich table of vision models and return the recommended model."""
    console = console or Console(stderr=True)
    recommended = pick_default_vision_model()

    table = Table(title="Vision OCR Models", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Description")

    for idx, model in enumerate(_get_vision_catalog(), 1):
        num_str = str(idx)
        name = model.name
        size_str = f"{model.size_gb:.1f} GB"
        desc = model.description

        is_recommended = model == recommended
        disk_too_small = free_disk_gb < model.size_gb + _DISK_HEADROOM_GB

        if is_recommended:
            name = f"[bold]{name} \u2605[/bold]"
            desc = f"[bold]{desc}[/bold]"
            num_str = f"[bold]{num_str}[/bold]"

        if disk_too_small:
            size_str = f"[red]{model.size_gb:.1f} GB[/red]"

        table.add_row(num_str, name, size_str, desc)

    console.print()
    console.print("[bold]Select a vision OCR model for scanned PDF extraction:[/bold]\n")
    console.print(table)
    console.print(f"\n  System: {ram_gb:.0f} GB RAM, {free_disk_gb:.1f} GB free disk")
    console.print("  \u2605 = recommended for your system")
    console.print(f"  Browse more models at {MODELS_BROWSE_URL}\n")

    return recommended


def prompt_model_choice(ram_gb: float) -> ModelInfo:
    """Prompt the user to pick a model by number. Returns the chosen ModelInfo."""
    free_disk_gb = get_free_disk_gb(cfg.data_dir)
    recommended = display_model_picker(ram_gb, free_disk_gb)
    default_idx = list(_get_model_catalog()).index(recommended) + 1

    while True:
        try:
            raw = input(f"Choice [{default_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return recommended

        if not raw:
            return recommended

        try:
            choice = int(raw)
        except ValueError:
            sys.stderr.write(f"Enter a number 1-{len(_get_model_catalog())}.\n")
            continue

        if 1 <= choice <= len(_get_model_catalog()):
            return _get_model_catalog()[choice - 1]

        sys.stderr.write(f"Enter a number 1-{len(_get_model_catalog())}.\n")


def validate_disk_and_pull(
    model_info: ModelInfo, free_gb: float, *, console: Console | None = None
) -> None:
    """Check disk space, pull the model, and persist the choice."""
    required_gb = model_info.size_gb + _DISK_HEADROOM_GB
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough disk space to download '{model_info.name}': "
            f"need {required_gb:.1f} GB, have {free_gb:.1f} GB free. "
            f"Free up space or choose a smaller model."
        )

    pull_with_progress(model_info.name, console=console)
    cfg.chat_model = model_info.name
    settings.set_value(cfg.data_root, "chat_model", model_info.name)


def pull_with_progress(model: str, *, console: Console | None = None) -> None:
    """Pull a model via model_manager, showing a Rich progress bar."""
    from lilbee.model_manager import ModelSource, get_model_manager

    if console is None:
        console = Console(file=sys.__stderr__ or sys.stderr)
    manager = get_model_manager()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        transient=True,
        console=console,
    ) as progress:
        desc = f"Downloading model '{model}'..."
        ptask = progress.add_task(desc, total=None)

        def _on_progress(data: dict) -> None:
            total = data.get("total", 0) or 0
            completed = data.get("completed", 0) or 0
            if total > 0:
                progress.update(ptask, total=total, completed=completed)

        manager.pull(model, ModelSource.NATIVE, on_progress=_on_progress)
    console.print(f"Model '{model}' ready.")


def ensure_chat_model() -> None:
    """If no chat models are installed, pick and pull one.

    Interactive (TTY): show catalog picker with descriptions and sizes.
    Non-interactive (CI/pipes): auto-pick recommended model silently.
    Persists the chosen model in config.toml so it becomes the default.
    """
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()
    try:
        installed = manager.list_installed()
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot list models: {exc}") from exc

    # Filter out embedding model — only check for chat models
    embed_base = cfg.embedding_model.split(":")[0]
    chat_models = [m for m in installed if m.split(":")[0] != embed_base]
    if chat_models:
        return

    ram_gb = get_system_ram_gb()
    free_gb = get_free_disk_gb(cfg.data_dir)

    if sys.stdin.isatty():
        model_info = prompt_model_choice(ram_gb)
    else:
        model_info = pick_default_model(ram_gb)
        sys.stderr.write(
            f"No chat model found. Auto-installing '{model_info.name}' "
            f"(detected {ram_gb:.0f} GB RAM)...\n"
        )

    validate_disk_and_pull(model_info, free_gb)


def list_installed_models(*, exclude_vision: bool = False) -> list[str]:
    """Return installed model names, excluding embedding models.

    When *exclude_vision* is True, also filters out known vision catalog models.
    """
    from lilbee.services import get_services

    try:
        provider = get_services().provider
        embed_base = cfg.embedding_model.split(":")[0]
        models = [m for m in provider.list_models() if m.split(":")[0] != embed_base]
        if exclude_vision:
            vision_names = {m.name for m in _get_vision_catalog()}
            models = [m for m in models if m not in vision_names]
        return models
    except Exception:
        return []
