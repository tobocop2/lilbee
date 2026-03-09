"""RAM detection, model selection, interactive picker, and auto-install for chat models."""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Extra headroom required beyond model size (GB)
_DISK_HEADROOM_GB = 2

OLLAMA_MODELS_URL = "https://ollama.com/library"


@dataclass(frozen=True)
class ModelInfo:
    """A curated chat model with metadata for the picker UI."""

    name: str
    size_gb: float
    min_ram_gb: float
    description: str


MODEL_CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo("qwen3:1.7b", 1.1, 4, "Tiny — fast on any machine"),
    ModelInfo("qwen3:4b", 2.5, 8, "Small — good balance for 8 GB RAM"),
    ModelInfo("mistral:7b", 4.4, 8, "Small — Mistral's fast 7B, 32K context"),
    ModelInfo("qwen3:8b", 5.0, 8, "Medium — strong general-purpose"),
    ModelInfo("mistral-small:24b", 14.0, 16, "Large — Mistral Small, 128K context"),
    ModelInfo("qwen3-coder:30b", 18.0, 32, "Extra large — best quality, needs 32 GB RAM"),
)


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


def pick_default_model(ram_gb: float) -> ModelInfo:
    """Choose the largest catalog model that fits in *ram_gb*."""
    best = MODEL_CATALOG[0]
    for model in MODEL_CATALOG:
        if model.min_ram_gb <= ram_gb:
            best = model
    return best


def _model_download_size_gb(model: str) -> float:
    """Estimated download size for a model."""
    catalog_sizes = {m.name: m.size_gb for m in MODEL_CATALOG}
    fallback = next(m.size_gb for m in MODEL_CATALOG if m.name == "qwen3:8b")
    return catalog_sizes.get(model, fallback)


def display_model_picker(ram_gb: float, free_disk_gb: float) -> ModelInfo:
    """Show a Rich table of catalog models on stderr and return the recommended model."""
    from rich.console import Console
    from rich.table import Table

    console = Console(stderr=True)
    recommended = pick_default_model(ram_gb)

    table = Table(title="Available Models", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Description")

    for idx, model in enumerate(MODEL_CATALOG, 1):
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
    console.print(f"  Browse more models at {OLLAMA_MODELS_URL}\n")

    return recommended


def prompt_model_choice(ram_gb: float) -> ModelInfo:
    """Prompt the user to pick a model by number. Returns the chosen ModelInfo."""
    import lilbee.config as cfg

    free_disk_gb = get_free_disk_gb(cfg.DATA_DIR)
    recommended = display_model_picker(ram_gb, free_disk_gb)
    default_idx = list(MODEL_CATALOG).index(recommended) + 1

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
            sys.stderr.write(f"Enter a number 1-{len(MODEL_CATALOG)}.\n")
            continue

        if 1 <= choice <= len(MODEL_CATALOG):
            return MODEL_CATALOG[choice - 1]

        sys.stderr.write(f"Enter a number 1-{len(MODEL_CATALOG)}.\n")


def _validate_disk_and_pull(model_info: ModelInfo, free_gb: float) -> None:
    """Check disk space, pull the model, and persist the choice."""
    import lilbee.config as cfg

    required_gb = model_info.size_gb + _DISK_HEADROOM_GB
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough disk space to download '{model_info.name}': "
            f"need {required_gb:.1f} GB, have {free_gb:.1f} GB free. "
            f"Free up space or manually pull a smaller model with 'ollama pull <model>'."
        )

    pull_with_progress(model_info.name)
    cfg.CHAT_MODEL = model_info.name

    from lilbee import settings

    settings.set_value("chat_model", model_info.name)


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
    """If Ollama has no chat models installed, pick and pull one.

    Interactive (TTY): show catalog picker with descriptions and sizes.
    Non-interactive (CI/pipes): auto-pick recommended model silently.
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
    free_gb = get_free_disk_gb(cfg.DATA_DIR)

    if sys.stdin.isatty():
        model_info = prompt_model_choice(ram_gb)
    else:
        model_info = pick_default_model(ram_gb)
        sys.stderr.write(
            f"No chat model found. Auto-installing '{model_info.name}' "
            f"(detected {ram_gb:.0f} GB RAM)...\n"
        )

    _validate_disk_and_pull(model_info, free_gb)
