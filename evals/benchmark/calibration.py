"""Calibrate the judge against human ratings that already exist.

The answer tier is one model grading another model's output. The obvious
question a reader asks is how anyone knows those grades track a human's, and
the honest answer has to be a number rather than an assurance.

Getting that number does not require anybody to annotate anything here.
SummEval already has it: 100 CNN/DailyMail articles, 16 machine summaries each,
every summary rated 1-5 by three experts on consistency and relevance. Running
this harness' own rubric over those 1600 pairs and correlating with the expert
means says how closely this judge tracks people who were annotating years before
this project existed, on labels nobody here could have tuned.

Two properties make it the right choice over the alternatives. Its scale is 1-5,
which is the scale ragas' rubric grades on, so the comparison needs no threshold
picked by hand. And its expert-to-expert agreement is published: about 0.80 on
consistency, lower on relevance. That is the reference the correlation is read
against. It is a reference and not a hard ceiling: the judge is correlated
against the mean of three experts, which by Spearman-Brown is more reliable than
any single expert, so the judge-vs-mean correlation can legitimately exceed
expert-vs-expert agreement. What the published figure does is turn a correlation
into something interpretable instead of a number floating free.

What this is not: SummEval is summarization, and this harness answers questions
over retrieved passages. The mapping is honest but it is a proxy -- article as
ground material, summary as response -- and a good result here means the judge
grades faithfulness sensibly in general, not that it is calibrated on this
corpus specifically. That distinction belongs in the report, not in a footnote.

Correlation rather than kappa, because the expert label is a mean of three
ratings and so is continuous, and because Spearman and Kendall are what the
summarization meta-evaluation literature reports on this dataset. Using the
same statistic is what makes this harness' number comparable to published ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.deps import install_hint

DATASETS_INSTALL_HINT = install_hint("datasets", "to load the judge-calibration set")

SUMMEVAL_DATASET = "mteb/summeval"

# SummEval dimension -> this harness' dimension. Citation is deliberately absent:
# a summary cites nothing, so there is no human label here to calibrate it
# against, and inventing a mapping would be worse than reporting the gap.
DIMENSION_MAP = {"consistency": "faithfulness", "relevance": "relevance"}

# Published expert-to-expert agreement on SummEval, the ceiling any judge is
# measured against. Recorded so the report can state it beside the result rather
# than leaving a correlation to be read as good or bad on vibes.
EXPERT_AGREEMENT = {"faithfulness": 0.798, "relevance": 0.398}


@dataclass(frozen=True)
class CalibrationPair:
    """One human-rated (ground material, response) pair from SummEval."""

    pair_id: str
    ground: str
    response: str
    human: dict[str, float]


def load_summeval(limit: int | None = None) -> list[CalibrationPair]:
    """Flatten SummEval into one row per rated summary.

    The dataset nests sixteen summaries and sixteen score lists under each
    article; the judge grades one response at a time, so it is flattened here.
    ``limit`` caps the number of articles, not the number of pairs, so a capped
    run still covers every system rather than the first system sixteen times.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(DATASETS_INSTALL_HINT) from exc
    rows = load_dataset(SUMMEVAL_DATASET, split="test")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    pairs: list[CalibrationPair] = []
    for record in rows:
        summaries = record["machine_summaries"]
        for index, summary in enumerate(summaries):
            human = {
                ours: float(record[theirs][index])
                for theirs, ours in DIMENSION_MAP.items()
                if index < len(record[theirs])
            }
            if len(human) != len(DIMENSION_MAP):
                continue
            pairs.append(
                CalibrationPair(
                    pair_id=f"{record['id']}::{index}",
                    ground=record["text"],
                    response=summary,
                    human=human,
                )
            )
    return pairs


@dataclass(frozen=True)
class Calibration:
    """How closely the judge tracked the human raters on one dimension."""

    dimension: str
    n: int
    spearman: float
    kendall: float
    expert_ceiling: float

    @property
    def fraction_of_ceiling(self) -> float:
        """The judge-human correlation relative to expert-to-expert agreement.

        A raw correlation of 0.6 means something different against a 0.80
        reference than against a 0.40 one, so it is reported as a ratio to the
        published inter-expert agreement. The ratio can exceed 1.0 and that is
        not an error: the judge is correlated against the mean of three experts,
        which by Spearman-Brown is more reliable than any single expert, so
        expert agreement is a reference point rather than a hard ceiling.
        """
        return self.spearman / self.expert_ceiling if self.expert_ceiling else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n": self.n,
            "spearman": round(self.spearman, 4),
            "kendall": round(self.kendall, 4),
            "expert_ceiling": self.expert_ceiling,
            "fraction_of_ceiling": round(self.fraction_of_ceiling, 4),
        }


class InsufficientCalibrationError(RuntimeError):
    """Too few graded pairs for a correlation to mean anything."""


# Below this a correlation is too noisy to publish. SummEval has 1600 pairs, so
# this only trips on a run that was cut short or mostly failed to grade.
MIN_CALIBRATION_PAIRS = 100


def calibrate(
    judge_scores: dict[str, dict[str, int]],
    pairs: list[CalibrationPair],
    *,
    minimum: int = MIN_CALIBRATION_PAIRS,
) -> list[Calibration]:
    """Correlate the judge's grades with the human ratings, per dimension.

    Both statistics are scipy's. Spearman is the headline because it is what the
    summarization meta-evaluation literature reports for this dataset; Kendall
    travels with it because it is less swayed by the judge's habit of bunching
    scores at one end, which Spearman handles less gracefully on a five-point
    scale with many ties.
    """
    from scipy import stats as scipy_stats

    human_by_id = {pair.pair_id: pair.human for pair in pairs}
    results: list[Calibration] = []
    for dimension in DIMENSION_MAP.values():
        graded = [
            (judge_scores[pair_id][dimension], human_by_id[pair_id][dimension])
            for pair_id in sorted(judge_scores)
            if pair_id in human_by_id and dimension in judge_scores[pair_id]
        ]
        if len(graded) < minimum:
            raise InsufficientCalibrationError(
                f"only {len(graded)} pairs were graded on {dimension}; a correlation "
                f"is not reported below {minimum}, since one on fewer pairs is too "
                "noisy to publish against a ceiling"
            )
        judged = [pair[0] for pair in graded]
        human = [pair[1] for pair in graded]
        results.append(
            Calibration(
                dimension=dimension,
                n=len(graded),
                spearman=float(scipy_stats.spearmanr(judged, human).statistic),
                kendall=float(scipy_stats.kendalltau(judged, human).statistic),
                expert_ceiling=EXPERT_AGREEMENT[dimension],
            )
        )
    return results
