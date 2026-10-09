"""The limits the verdict uses, read from thresholds.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

THRESHOLDS_FILE = Path(__file__).parent / "thresholds.toml"


@dataclass(frozen=True)
class ParityLimits:
    """Limits of the comparison of candidate and oracle."""

    pages_lost_max: int
    pages_extra_unreadable_max: int
    words_lost_per_page_max: int
    words_added_per_page_max: int
    invisible_text_lost_counts: bool
    symbols_changed_per_page_max: int
    structure_changed_per_page_max: int


@dataclass(frozen=True)
class TruthLimits:
    """Limits of the comparison of one side and the rendered page, used with no oracle."""

    visible_recall_min: float
    unexplained_words_per_page_max: int


@dataclass(frozen=True)
class LeftBehindLimits:
    """Limits on what a run leaves."""

    processes_max: int
    temp_entries_max: int
    threads_max: int


@dataclass(frozen=True)
class SpeedLimits:
    """How speed is measured and how much slower the candidate may be."""

    repeats: int
    load_max: float
    slowdown_max: float
    first_page_delay_max: float


@dataclass(frozen=True)
class RetrievalLimits:
    """How retrieval is measured and how much recall the candidate may lose."""

    top_k: int
    sample_size: int
    recall_drop_max: float


@dataclass(frozen=True)
class Thresholds:
    """Every limit, by yardstick."""

    parity: ParityLimits
    truth: TruthLimits
    left_behind: LeftBehindLimits
    speed: SpeedLimits
    retrieval: RetrievalLimits


def load_thresholds(path: Path = THRESHOLDS_FILE) -> Thresholds:
    """The thresholds in *path*; a missing or unknown key is an error, never a default."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    return Thresholds(
        parity=ParityLimits(**document["parity"]),
        truth=TruthLimits(**document["truth"]),
        left_behind=LeftBehindLimits(**document["left_behind"]),
        speed=SpeedLimits(**document["speed"]),
        retrieval=RetrievalLimits(**document["retrieval"]),
    )
