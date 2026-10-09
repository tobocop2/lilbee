"""Speed: pages per second and time to first page, each sample with the machine's load beside it."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode
from tools.qa.crawl_parity.sides import CrawlRequest, CrawlResult, SideConfig, run_crawl
from tools.qa.crawl_parity.thresholds import SpeedLimits

PAGES_FEATURE = "pages-per-second"
FIRST_PAGE_FEATURE = "first-page"


@dataclass(frozen=True)
class Sample:
    """One timed crawl. A sample has no meaning without its load."""

    pages: int
    crawl_seconds: float
    first_page_seconds: float
    load: float

    @property
    def pages_per_second(self) -> float:
        """Saved pages divided by the time inside the crawl."""
        return self.pages / self.crawl_seconds if self.crawl_seconds > 0 else 0.0

    def text(self) -> str:
        """The sample with its load, the only form a sample is printed in."""
        return (
            f"{self.pages_per_second:.1f} pages/s, first page {self.first_page_seconds:.2f} s "
            f"(load {self.load:.2f})"
        )


@dataclass(frozen=True)
class SpeedComparison:
    """The verdict on speed, or the reason there is none."""

    compared: bool
    reason: str
    differences: tuple[Difference, ...] = ()


def sample_of(result: CrawlResult) -> Sample | None:
    """The sample a crawl gives, or None when the driver did not finish or saved nothing."""
    if result.crawl_seconds is None or result.first_page_seconds is None:
        return None
    return Sample(
        len(result.saved()), result.crawl_seconds, result.first_page_seconds, result.load_before
    )


def measure(
    config: SideConfig,
    layer: Layer,
    request: CrawlRequest,
    repeats: int,
    work: Path,
    before_each: Callable[[], None],
) -> list[Sample]:
    """Time *repeats* crawls; a crawl that gives no sample is left out, so the list can be short."""
    samples: list[Sample] = []
    for repeat in range(repeats):
        before_each()
        sample = sample_of(run_crawl(config, layer, request, work / f"repeat-{repeat}"))
        if sample is not None:
            samples.append(sample)
    return samples


def compare(
    oracle: list[Sample], candidate: list[Sample], limits: SpeedLimits, layer: Layer, mode: Mode
) -> SpeedComparison:
    """Compare medians; refuse when a sample is missing or was taken above the load limit."""
    if len(oracle) < limits.repeats or len(candidate) < limits.repeats:
        return SpeedComparison(
            False,
            f"{len(oracle)} oracle and {len(candidate)} candidate samples of {limits.repeats}",
        )
    worst = max(sample.load for sample in [*oracle, *candidate])
    if worst > limits.load_max:
        return SpeedComparison(False, f"load {worst:.2f} is above the limit {limits.load_max:.2f}")
    differences: list[Difference] = []
    oracle_rate = statistics.median(sample.pages_per_second for sample in oracle)
    candidate_rate = statistics.median(sample.pages_per_second for sample in candidate)
    slowdown = 1.0 - candidate_rate / oracle_rate if oracle_rate > 0 else 0.0
    if slowdown > limits.slowdown_max:
        detail = (
            f"oracle {oracle_rate:.1f} pages/s, candidate {candidate_rate:.1f} "
            f"(load at most {worst:.2f})"
        )
        differences.append(
            Difference(Kind.SLOWER, mode, layer, PAGES_FEATURE, amount=slowdown, detail=detail)
        )
    delay = statistics.median(s.first_page_seconds for s in candidate) - statistics.median(
        s.first_page_seconds for s in oracle
    )
    if delay > limits.first_page_delay_max:
        detail = f"first page {delay:.2f} s later than the oracle's (load at most {worst:.2f})"
        differences.append(
            Difference(Kind.SLOWER, mode, layer, FIRST_PAGE_FEATURE, amount=delay, detail=detail)
        )
    return SpeedComparison(True, f"load at most {worst:.2f}", tuple(differences))
