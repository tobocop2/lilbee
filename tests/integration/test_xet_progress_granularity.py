"""Bounded-prefix QA matrix: real-time download-progress granularity, xet vs HTTP.

WHY: lilbee sets ``HF_HUB_DISABLE_XET=1`` (src/lilbee/__init__.py) because the xet
transfer layer historically reported download progress in 3-4 coarse jumps (HF issue
#4058), making bars look stuck on large files. Forcing the HTTP path restores smooth
per-chunk updates. This harness empirically measures whether hf-xet's transfer layer
now drives our progress callback (catalog/download_progress.py) at chunk granularity,
i.e. whether the bypass can be dropped.

It NEVER downloads the whole file. Each transfer path runs in a fresh subprocess
(``HF_HUB_DISABLE_XET`` is read at import time, so it must be set before ``import
lilbee``). The child streams progress samples; the parent kills it once a bounded byte
prefix has been sampled. Total bandwidth per run is ~2x ``PREFIX_BUDGET_BYTES``.

Manual run (prints the matrix + verdict):
    uv run python tests/integration/test_xet_progress_granularity.py
Pytest (slow):
    uv run pytest tests/integration/test_xet_progress_granularity.py -v -m slow
"""

from __future__ import annotations

import contextlib
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

# A xet-backed GGUF (get_hf_file_metadata reports XetFileData for this file).
# ~1.0 GB total; we only sample the prefix, so the full size never downloads.
TEST_REPO = "unsloth/Qwen3-4B-GGUF"
TEST_FILE = "Qwen3-4B-UD-IQ1_S.gguf"

# Stop each download once this many bytes have been observed. Large enough to
# reveal cadence on the xet path, small enough to stay cheap.
PREFIX_BUDGET_BYTES = 400 * 1024 * 1024
# Hard wall-clock cap per path, so a stalled transfer can't hang the run.
WALL_TIMEOUT_S = 600.0

# Thresholds for "real-time" over the sampled prefix. The HTTP path (200KB chunks,
# see _shrink_hf_download_chunk_size) is the known-good baseline and easily clears
# these; the xet path is what we're judging.
MIN_UPDATES = 20
MAX_JUMP_PCT = 10.0

_SAMPLE_PREFIX = "S\t"


@dataclass
class Metrics:
    path: str
    samples: list[tuple[int, int, float]] = field(default_factory=list)  # (bytes, total, ts)
    error: str | None = None

    @property
    def n_updates(self) -> int:
        """Distinct forward progress updates (a sample whose byte count grew)."""
        count = 0
        prev = -1
        for done, _total, _ts in self.samples:
            if done > prev:
                count += 1
                prev = done
        return count

    @property
    def total_size(self) -> int:
        for _done, total, _ts in self.samples:
            if total > 0:
                return total
        return 0

    @property
    def max_jump_pct(self) -> float:
        """Largest single-update byte jump as a percent of the full file size."""
        total = self.total_size
        if total <= 0 or len(self.samples) < 2:
            return 100.0
        prev = self.samples[0][0]
        worst = 0
        for done, _total, _ts in self.samples[1:]:
            worst = max(worst, done - prev)
            prev = done
        return worst * 100.0 / total

    @property
    def gaps(self) -> list[float]:
        return [b[2] - a[2] for a, b in zip(self.samples, self.samples[1:], strict=False)]

    @property
    def max_gap_s(self) -> float:
        gaps = self.gaps
        return max(gaps) if gaps else 0.0

    @property
    def median_gap_s(self) -> float:
        gaps = self.gaps
        return statistics.median(gaps) if gaps else 0.0

    @property
    def monotonic(self) -> bool:
        prev = -1
        for done, _total, _ts in self.samples:
            if done < prev:
                return False
            prev = done
        return True

    @property
    def is_smooth(self) -> bool:
        return (
            self.error is None
            and self.n_updates >= MIN_UPDATES
            and self.max_jump_pct <= MAX_JUMP_PCT
            and self.monotonic
        )


def _child_download(models_dir: str) -> int:
    """Child entry point: download into ``models_dir``, stream progress, abort at budget.

    Runs in a fresh process so the parent's ``HF_HUB_DISABLE_XET`` env is honored.
    """
    import lilbee  # noqa: F401  # sets HF env from the inherited environment
    from lilbee.catalog import CatalogModel, download_model
    from lilbee.core.config.model import cfg
    from lilbee.runtime.cancellation import TaskCancelledError

    cfg.models_dir = Path(models_dir)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    entry = CatalogModel(
        hf_repo=TEST_REPO,
        gguf_filename=TEST_FILE,
        size_gb=1.0,
        min_ram_gb=2.0,
        description="xet granularity probe",
        featured=False,
        downloads=0,
        task="chat",
    )

    def on_progress(downloaded: int, total: int) -> None:
        sys.stdout.write(f"{_SAMPLE_PREFIX}{downloaded}\t{total}\t{time.monotonic()}\n")
        sys.stdout.flush()
        if downloaded >= PREFIX_BUDGET_BYTES:
            # Best-effort clean stop (download.py re-raises TaskCancelledError).
            # The parent also hard-kills at budget in case xet swallows this.
            raise TaskCancelledError

    try:
        download_model(entry, on_progress=on_progress)
    except TaskCancelledError:
        pass
    except Exception as exc:
        sys.stdout.write(f"ERR\t{type(exc).__name__}: {exc}\n")
        sys.stdout.flush()
        return 1
    return 0


def _measure(path: str, base_dir: Path) -> Metrics:
    """Run one transfer path in a subprocess, killing it once the prefix is sampled."""
    assert path in ("xet", "http")
    models_dir = base_dir / "models"
    hf_home = base_dir / "hf_home"
    models_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "0" if path == "xet" else "1"
    env["HF_HOME"] = str(hf_home)  # isolates the xet chunk-cache for a cold download
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    proc = subprocess.Popen(
        [sys.executable, __file__, "--child", str(models_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    metrics = Metrics(path=path)
    deadline = time.monotonic() + WALL_TIMEOUT_S
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.monotonic() > deadline:
                metrics.error = metrics.error or "wall-clock timeout"
                break
            if line.startswith("ERR\t"):
                metrics.error = line[4:].strip()
                break
            if not line.startswith(_SAMPLE_PREFIX):
                continue
            try:
                done_s, total_s, ts_s = line[len(_SAMPLE_PREFIX) :].split("\t")
                metrics.samples.append((int(done_s), int(total_s), float(ts_s)))
            except ValueError:
                continue
            if metrics.samples[-1][0] >= PREFIX_BUDGET_BYTES:
                break
    finally:
        _kill_group(proc)
    return metrics


def _kill_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def run_matrix(base_dir: Path) -> tuple[Metrics, Metrics]:
    """Measure both transfer paths under isolated, cold caches."""
    http = _measure("http", base_dir / "http")
    xet = _measure("xet", base_dir / "xet")
    return http, xet


def _row(m: Metrics) -> str:
    if m.error and not m.samples:
        return f"  {m.path:<5} | ERROR: {m.error}"
    verdict = "PASS" if m.is_smooth else "FAIL"
    return (
        f"  {m.path:<5} | {verdict} | updates={m.n_updates:>5} "
        f"max_jump={m.max_jump_pct:6.2f}% max_gap={m.max_gap_s:6.2f}s "
        f"median_gap={m.median_gap_s:6.3f}s monotonic={m.monotonic}"
    )


def format_matrix(http: Metrics, xet: Metrics) -> str:
    total_mb = (http.total_size or xet.total_size) / (1024 * 1024)
    lines = [
        "",
        "Real-time download-progress QA matrix",
        f"  model: {TEST_REPO}/{TEST_FILE}  (~{total_mb:.0f} MB total, "
        f"sampled prefix ~{PREFIX_BUDGET_BYTES // (1024 * 1024)} MB)",
        f"  bar:   >= {MIN_UPDATES} updates, max single jump <= {MAX_JUMP_PCT:.0f}% of file",
        _row(http),
        _row(xet),
    ]
    if xet.error is None and xet.samples:
        if xet.is_smooth:
            lines.append("  VERDICT: xet now delivers real-time progress.")
            lines.append(
                "  -> RECOMMEND dropping HF_HUB_DISABLE_XET=1 (re-check the chunk shrink)."
            )
        else:
            lines.append("  VERDICT: xet progress is still coarse; keep the HTTP bypass.")
    lines.append("")
    return "\n".join(lines)


def test_progress_granularity_matrix(tmp_path: Path) -> None:
    """Run the xet-vs-HTTP matrix; guard the HTTP baseline, report the xet verdict."""
    http, xet = run_matrix(tmp_path)
    print(format_matrix(http, xet))

    if http.error and not http.samples:
        pytest.skip(f"HTTP baseline unavailable (network?): {http.error}")

    # The HTTP path is lilbee's shipped default: it must stay smooth (regression guard).
    assert http.is_smooth, f"HTTP baseline regressed: {_row(http)}"

    # The xet path is under investigation and disabled by default. We only require
    # that it produces progress at all; smoothness is reported, not asserted.
    if xet.error and not xet.samples:
        pytest.skip(f"xet path unavailable (network?): {xet.error}")
    assert xet.n_updates >= 1, "xet path produced no progress samples"


def _main() -> int:
    with tempfile.TemporaryDirectory(prefix="lilbee-xet-qa-") as tmp:
        http, xet = run_matrix(Path(tmp))
        print(format_matrix(http, xet))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        sys.exit(_child_download(sys.argv[2]))
    sys.exit(_main())
