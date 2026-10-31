"""Canary vectors: fixed texts that show whether the embedder still agrees with an index."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lilbee.core.vectors import Vector

CANARY_TEXTS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "def binary_search(items, target): return bisect.bisect_left(items, target)",
    "Der Bundestag verabschiedete gestern ein neues Gesetz zur Energieversorgung.",
    "La bibliothèque municipale ouvre ses portes à neuf heures chaque matin.",
    "El tren de alta velocidad conecta Madrid con Barcelona en menos de tres horas.",
    "Москва является крупнейшим городом России по численности населения.",
    "東京の桜は毎年三月の終わりから四月の初めに咲きます。",
    "人工智能正在改变医疗诊断和药物研发的方式。",
    "تقع مدينة القاهرة على ضفاف نهر النيل في شمال مصر.",
    "Invoice 4471: 3 units at $19.99, total $59.97, due 2026-11-30.",
    "ok",
)

# Cosine distance. The same model measured 3.46e-2 apart across backends; 0.05 sits clear of that.
CANARY_MAX_DISTANCE = 0.05
MAX_COSINE_DISTANCE = 2.0


@dataclass(frozen=True)
class CanaryState:
    """The baseline vectors of an index, the engine that last passed, and the verified flag."""

    vectors: list[list[float]]
    trigger: str
    verified: bool


@dataclass(frozen=True)
class CanaryProbe:
    """How a store embeds canaries and names the embedder it would embed them with."""

    embed: Callable[[list[str]], list[Vector]]
    trigger: Callable[[], str | None]


def canary_distance(baseline: list[list[float]], fresh: list[Vector]) -> float:
    """The largest cosine distance between a baseline vector and the vector of the same canary.

    Sets of another shape do not compare: that is 2.0, the largest cosine distance.
    """
    if np.shape(baseline) != np.shape(fresh):
        return MAX_COSINE_DISTANCE
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(fresh, dtype=np.float64)
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(1.0 - cos.min())
