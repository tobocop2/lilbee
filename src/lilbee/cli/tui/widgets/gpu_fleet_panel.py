"""Live GPU fleet panel widget for the Placement screen.

Renders per-card utilization and VRAM bars in the rose-pine palette and
refreshes on a ~1 s interval.  The probe runs off the UI thread so the
event loop is never blocked by the nvidia-smi subprocess.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.timer import Timer
from textual.widgets import Static

from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.providers.fleet.gpu_stats import GpuStat, probe_gpu_stats

if TYPE_CHECKING:
    from lilbee.providers.fleet.gpu_stats import _DeviceLike

log = logging.getLogger(__name__)

_CSS_FILE = Path(__file__).parent / "gpu_fleet_panel.tcss"

# Rose-pine palette (hex literals for Rich markup)
_COLOR_IRIS = "#c4a7e7"
_COLOR_FOAM = "#9ccfd8"
_COLOR_GOLD = "#f6c177"
_COLOR_LOVE = "#eb6f92"
_COLOR_TEXT = "#e0def4"
_COLOR_SUBTLE = "#908caa"
_COLOR_MUTED = "#6e6a86"
_COLOR_TRACK = "#403d52"

# Bar render constants
_BAR_FILL = "█"  # full block █
_BAR_TRACK = "─"  # horizontal box-drawing line ─
_BAR_WIDTH = 14  # number of cells in each bar
_BULLET = "●"  # ● colored dot before card label

# Utilization thresholds (%)
_UTIL_WARM = 40
_UTIL_HOT = 80

# VRAM saturation threshold (fraction)
_VRAM_HIGH = 0.85

# Refresh interval
_TICK_INTERVAL_S = 1.0

_GIB = 1024**3

# Column labels
_LABEL_UTIL = "util"
_LABEL_VRAM = "vram"
_TITLE_TEXT = "G P U   F L E E T"

# Displayed when no GPU info is available
_EMPTY_TEXT = "(no GPUs detected)"

# Shown as the util reading when utilization_pct is None
_UTIL_DASH = " -- "


def _heat_color(utilization_pct: int) -> str:
    """Rose-pine heat color keyed on utilization percentage."""
    if utilization_pct < _UTIL_WARM:
        return _COLOR_FOAM
    if utilization_pct < _UTIL_HOT:
        return _COLOR_GOLD
    return _COLOR_LOVE


def _bar(fraction: float, width: int, fill_color: str) -> str:
    """Render a filled bar with rose-pine track for the remainder."""
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    return (
        f"[{fill_color}]{_BAR_FILL * filled}[/][{_COLOR_TRACK}]{_BAR_TRACK * (width - filled)}[/]"
    )


def _render_stats(stats: dict[int, GpuStat], labels: dict[int, str], names: dict[int, str]) -> str:
    """Build the Rich markup string for all GPUs from a stat snapshot."""
    if not stats:
        return f"[{_COLOR_MUTED}]  {_EMPTY_TEXT}[/]"
    lines: list[str] = [
        f"[bold {_COLOR_IRIS}]  {_TITLE_TEXT}[/]",
        f"[{_COLOR_TRACK}]{'─' * (_BAR_WIDTH + 24)}[/]",
        "",
    ]
    for idx in sorted(stats):
        s = stats[idx]
        label = labels.get(idx, f"GPU{idx}")
        name = names.get(idx, "")
        used_bytes = s.total_bytes - s.free_bytes
        vram_frac = used_bytes / s.total_bytes if s.total_bytes else 0.0
        used_gib = used_bytes / _GIB
        total_gib = s.total_bytes / _GIB
        vram_color = _COLOR_LOVE if vram_frac >= _VRAM_HIGH else _COLOR_IRIS

        # Card heading row
        if s.utilization_pct is not None:
            dot_color = _heat_color(s.utilization_pct)
        else:
            dot_color = _COLOR_MUTED
        heading = (
            f"  [{dot_color}]{_BULLET}[/] [bold {_COLOR_TEXT}]{label}[/]  [{_COLOR_MUTED}]{name}[/]"
        )
        lines.append(heading)

        # Utilization bar
        if s.utilization_pct is not None:
            heat = _heat_color(s.utilization_pct)
            util_bar = _bar(s.utilization_pct / 100, _BAR_WIDTH, heat)
            util_pct = f"[{heat}]{s.utilization_pct:>3}%[/]"
        else:
            util_bar = f"[{_COLOR_TRACK}]{_BAR_TRACK * _BAR_WIDTH}[/]"
            util_pct = f"[{_COLOR_MUTED}]{_UTIL_DASH}[/]"
        lines.append(f"     [{_COLOR_SUBTLE}]{_LABEL_UTIL}[/]  {util_bar} {util_pct}")

        # VRAM bar
        vram_bar = _bar(vram_frac, _BAR_WIDTH, vram_color)
        lines.append(
            f"     [{_COLOR_SUBTLE}]{_LABEL_VRAM}[/]  {vram_bar}"
            f" [{_COLOR_SUBTLE}]{used_gib:>4.1f}/{total_gib:>2.0f}G[/]"
        )
        lines.append("")
    return "\n".join(lines)


class GpuFleetPanel(Static):
    """Live GPU utilization and VRAM panel, refreshed every ~1 s.

    Mount on any Screen or Widget; devices must be set via `set_devices`
    before the first tick fires meaningful data.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self) -> None:
        super().__init__(_EMPTY_TEXT, id="gpu-fleet-panel")
        self._devices: Sequence[_DeviceLike] = []
        self._labels: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self._timer: Timer | None = None

    def set_devices(
        self,
        devices: Sequence[_DeviceLike],
        *,
        labels: dict[int, str],
        names: dict[int, str],
    ) -> None:
        """Register the GPU devices the panel probes on each tick.

        `labels` maps device index to the short display label (e.g. "CUDA0").
        `names` maps device index to the GPU name (e.g. "NVIDIA A40").
        """
        self._devices = list(devices)
        self._labels = dict(labels)
        self._names = dict(names)

    def on_mount(self) -> None:
        self._timer = self.set_interval(_TICK_INTERVAL_S, self._request_stats)
        # Populate immediately instead of waiting one full interval
        self._request_stats()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _request_stats(self) -> None:
        """Kick off an off-thread stats probe."""
        self._probe_worker(list(self._devices), self._labels.copy(), self._names.copy())

    @work(thread=True, exit_on_error=False)
    def _probe_worker(
        self,
        devices: list[_DeviceLike],
        labels: dict[int, str],
        names: dict[int, str],
    ) -> None:
        """Probe GPU stats off the UI thread and push the result back."""
        try:
            stats = probe_gpu_stats(devices)
        except Exception:
            log.debug("gpu_fleet_panel: probe failed", exc_info=True)
            return
        call_from_thread(self, self._apply_stats, stats, labels, names)

    def _apply_stats(
        self,
        stats: dict[int, GpuStat],
        labels: dict[int, str],
        names: dict[int, str],
    ) -> None:
        """Update the rendered content with fresh stat data (main thread)."""
        markup = _render_stats(stats, labels, names)
        self.update(markup)
