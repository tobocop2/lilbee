"""Live GPU fleet panel widget for the Fleet view.

Renders per-card utilization and VRAM bars styled via rose-pine theme tokens and
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
    from lilbee.providers.fleet.gpu_stats import DeviceLike

log = logging.getLogger(__name__)

_CSS_FILE = Path(__file__).parent / "gpu_fleet_panel.tcss"

# Bar render constants. The empty track uses the same light-shade block as the
# Task Center download bars (progress_cell.py) so the GPU bars read as
# full-bodied siblings rather than a thin rule.
_BAR_FILL = "█"  # full block █
_BAR_TRACK = "░"  # light shade block ░ (matches the download progress bars)
_BAR_WIDTH = 16  # number of cells in each bar
_BULLET = "●"  # ● colored dot before card label

# Utilization thresholds (%)
_UTIL_WARM = 40
_UTIL_HOT = 80

# VRAM saturation threshold (fraction)
_VRAM_HIGH = 0.85

# Temperature threshold (°C) above which temp is shown in error color
_TEMP_HOT = 75

# Refresh interval
_TICK_INTERVAL_S = 1.0

_GIB = 1024**3

# Displayed when no GPU info is available
_EMPTY_TEXT = "(no GPUs detected)"

# Shown as the util reading when utilization_pct is None
_UTIL_DASH = " -- "

# Rich color used when a theme token is absent (renders in terminal default)
_COLOR_FALLBACK = "default"

# Badge role/model separator
_BADGE_SEP = " - "


def _theme_color(theme: dict[str, str], token: str) -> str:
    """Resolve a theme token to a Rich color, falling back to terminal default."""
    return theme.get(token, _COLOR_FALLBACK)


def _heat_color(utilization_pct: int, theme: dict[str, str]) -> str:
    """Theme-token heat color keyed on utilization percentage."""
    if utilization_pct < _UTIL_WARM:
        return _theme_color(theme, "success")
    if utilization_pct < _UTIL_HOT:
        return _theme_color(theme, "warning")
    return _theme_color(theme, "error")


def _bar(fraction: float, width: int, fill_color: str, track_color: str) -> str:
    """Render a filled bar with a track for the remainder."""
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    return f"[{fill_color}]{_BAR_FILL * filled}[/][{track_color}]{_BAR_TRACK * (width - filled)}[/]"


def _badge_role_markup(badge: str, secondary: str, muted: str) -> str:
    """Return Rich markup for the role portion of a badge.

    Splits "role - model" so the role renders in secondary and the model in muted.
    """
    if _BADGE_SEP in badge:
        role_part, model_part = badge.split(_BADGE_SEP, 1)
        return f"[{secondary}]{role_part}[/][{muted}]{_BADGE_SEP}{model_part}[/]"
    return f"[{secondary}]{badge}[/]"


def _render_row(
    s: GpuStat,
    label: str,
    badge: str,
    theme: dict[str, str],
) -> str:
    """Render one GPU as a single table row: bullet, label, util bar, vram bar, badge."""
    muted = _theme_color(theme, "text-muted")
    secondary = _theme_color(theme, "secondary")
    foreground = _theme_color(theme, "foreground")
    panel_color = _theme_color(theme, "panel")
    primary = _theme_color(theme, "primary")
    error = _theme_color(theme, "error")

    used_bytes = s.total_bytes - s.free_bytes
    vram_frac = used_bytes / s.total_bytes if s.total_bytes else 0.0
    used_gib = used_bytes / _GIB
    total_gib = s.total_bytes / _GIB
    vram_color = error if vram_frac >= _VRAM_HIGH else primary

    if s.utilization_pct is not None:
        heat = _heat_color(s.utilization_pct, theme)
        dot_color = heat
        util_bar = _bar(s.utilization_pct / 100, _BAR_WIDTH, heat, panel_color)
        util_pct = f"[{heat}]{s.utilization_pct:>3}%[/]"
    else:
        dot_color = muted
        util_bar = f"[{panel_color}]{_BAR_TRACK * _BAR_WIDTH}[/]"
        util_pct = f"[{muted}]{_UTIL_DASH}[/]"

    vram_bar = _bar(vram_frac, _BAR_WIDTH, vram_color, panel_color)
    vram_txt = f"[{muted}]{used_gib:>5.1f}/{total_gib:>2.0f}G[/]"
    badge_txt = f"  {_badge_role_markup(badge, secondary, muted)}" if badge else ""
    return (
        f"  [{dot_color}]{_BULLET}[/] [bold {foreground}]{label:<6}[/]"
        f" {util_bar} {util_pct}  {vram_bar} {vram_txt}{badge_txt}"
    )


def _render_stats(
    stats: dict[int, GpuStat],
    labels: dict[int, str],
    roles: dict[int, str],
    theme: dict[str, str],
) -> str:
    """Build the unified GPU table (one row per GPU) from a stat snapshot."""
    if not stats:
        muted = _theme_color(theme, "text-muted")
        return f"[{muted}]  {_EMPTY_TEXT}[/]"
    return "\n".join(
        _render_row(stats[idx], labels.get(idx, f"GPU{idx}"), roles.get(idx, ""), theme)
        for idx in sorted(stats)
    )


class GpuFleetPanel(Static):
    """Live GPU utilization and VRAM panel, refreshed every ~1 s.

    Mount on any Screen or Widget; devices must be set via `set_devices`
    before the first tick fires meaningful data.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self) -> None:
        super().__init__(_EMPTY_TEXT, id="gpu-fleet-panel")
        self._devices: Sequence[DeviceLike] = []
        self._labels: dict[int, str] = {}
        self._roles: dict[int, str] = {}
        self._timer: Timer | None = None
        # Single-flight: True while a probe worker is running, so a probe
        # slower than the tick interval skips ticks instead of stacking
        # threads (and their nvidia-smi subprocesses) without bound.
        self._probing = False

    def set_devices(
        self,
        devices: Sequence[DeviceLike],
        *,
        labels: dict[int, str],
        roles: dict[int, str] | None = None,
    ) -> None:
        """Register the GPU devices the panel probes on each tick.

        `labels` maps device index to the short display label (e.g. "CUDA0").
        `roles` maps device index to a badge string (e.g. "chat - Qwen3-235B"), "" when idle.
        """
        self._devices = list(devices)
        self._labels = dict(labels)
        self._roles = dict(roles) if roles is not None else {}

    def on_mount(self) -> None:
        self._timer = self.set_interval(_TICK_INTERVAL_S, self._request_stats)
        # Populate immediately instead of waiting one full interval
        self._request_stats()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _request_stats(self) -> None:
        """Kick off an off-thread stats probe unless one is already in flight."""
        if self._probing:
            return
        self._probing = True
        self._probe_worker(
            list(self._devices),
            self._labels.copy(),
            self._roles.copy(),
        )

    def _resolve_theme(self) -> dict[str, str]:
        """Return the current app theme variables with `$` prefix stripped."""
        try:
            raw: dict[str, str] = self.app.theme_variables
            return {k.lstrip("$"): v for k, v in raw.items()}
        except AttributeError:
            return {}

    @work(thread=True, exit_on_error=False)
    def _probe_worker(
        self,
        devices: list[DeviceLike],
        labels: dict[int, str],
        roles: dict[int, str],
    ) -> None:
        """Probe GPU stats off the UI thread and push the result back."""
        try:
            stats = probe_gpu_stats(devices)
        except Exception:
            log.debug("gpu_fleet_panel: probe failed", exc_info=True)
            return
        finally:
            call_from_thread(self, setattr, self, "_probing", False)
        call_from_thread(self, self._apply_stats, stats, labels, roles)

    def _apply_stats(
        self,
        stats: dict[int, GpuStat],
        labels: dict[int, str],
        roles: dict[int, str],
    ) -> None:
        """Update the rendered content with fresh stat data (main thread)."""
        theme = self._resolve_theme()
        markup = _render_stats(stats, labels, roles, theme)
        self.update(markup)
