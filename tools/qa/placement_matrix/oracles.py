"""What must be true of a placement, judged from a recorded observation.

Pure functions over :class:`Observation`, deliberately separated from the code
that produces one: the half that needs GPUs cannot be tested in CI, and the half
that decides pass or fail is the half worth testing. Every verdict here is
checked against fabricated observations in tests/test_placement_matrix.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

_GB = 1024**3
# Bumped whenever a field changes meaning. Results are merged across pods and
# possibly across commits, and a stale file read with today's defaults would
# report a field nobody measured (an unmeasured idle box reads as idle).
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Observation:
    """What one cell did on real hardware."""

    cell_id: str
    model_key: str
    cards: int
    total_free_bytes: int
    weights_bytes: int
    planned: bool
    """The planner emitted a chat launch."""
    refusal: str | None = None
    ctx: int = 0
    slots: int = 0
    argv: tuple[str, ...] = ()
    tight: bool = False
    est_by_device: dict[str, int] = field(default_factory=dict)
    actual_by_device: dict[str, int] = field(default_factory=dict)
    loaded: bool = False
    sustained: bool = False
    """Served consecutive requests that filled the served window."""
    forced_loaded: bool | None = None
    """Refused cells only: did a forced num_ctx launch load anyway."""
    forced_sustained: bool | None = None
    min_usable_ctx: int = 0
    skipped: str | None = None
    error: str | None = None
    """The cell raised instead of producing a verdict; never silently dropped."""
    forced_planned: bool | None = None
    """Refused cells only: whether a num_ctx pin even produced a launch to try."""
    vram_was_idle: bool = True
    """False when another process still held VRAM at launch, which taints the load."""
    generated_tokens: int = 0
    prompt_tokens: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Observation:
        """Parse one recorded cell, refusing anything this harness cannot read.

        Strict in both directions: a result from another schema, or one carrying a
        field this version does not know, is an error rather than a set of
        defaults that would look like measurements.
        """
        schema = payload.get("schema")
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"result was written by schema {schema!r}, this harness reads {SCHEMA_VERSION}"
            )
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known - {"schema"}
        if unknown:
            raise ValueError(f"result carries unknown fields {sorted(unknown)}")
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True)
class Failure:
    """One violated invariant, named by the rule that caught it."""

    rule: str
    cell_id: str
    detail: str


# A per-device estimate may exceed reality (conservative) by this much before it
# is worth reporting; under-estimating by any margin is what OOMs a card.
_OVER_ESTIMATE_TOLERANCE = 2 * _GB
_UNDER_ESTIMATE_TOLERANCE = 512 * 1024 * 1024


def judge(observation: Observation) -> list[Failure]:
    """Every invariant violated by one cell.

    An errored cell is a failure, not an absence: a run whose cells all raised
    would otherwise merge into a report with nothing to complain about.
    """
    if observation.error:
        return [Failure("cell-errored", observation.cell_id, observation.error)]
    if observation.skipped:
        return []
    checks = (
        _served_plans_load,
        _served_plans_sustain,
        _refusals_are_real,
        _estimates_bracket_reality,
        _readback_was_readable,
        _oversize_models_spill,
        _tight_groups_let_the_engine_fit,
        _the_load_was_measured_alone,
    )
    return [failure for check in checks for failure in check(observation)]


def _readback_was_readable(o: Observation) -> list[Failure]:
    """A load with no per-device report leaves the estimate check with nothing.

    Silent, it is indistinguishable from an accurate estimate, which is how a
    parser that stopped matching the engine's log format becomes invisible.
    """
    if o.loaded and o.est_by_device and not o.actual_by_device:
        return [
            Failure(
                "readback-missing",
                o.cell_id,
                "the engine reported no per-device memory where lilbee reads it, so "
                "the estimate could not be checked at all",
            )
        ]
    return []


def _the_load_was_measured_alone(o: Observation) -> list[Failure]:
    """A load racing another tenant measures that tenant, not the plan."""
    if o.loaded and not o.vram_was_idle:
        return [
            Failure(
                "load-measured-alone",
                o.cell_id,
                "another process still held VRAM at launch, so this cell's memory "
                "numbers describe a contended box",
            )
        ]
    return []


def _served_plans_load(o: Observation) -> list[Failure]:
    """A plan is a promise the launch loads."""
    if o.planned and not o.loaded:
        return [Failure("plan-loads", o.cell_id, "planner emitted a launch that failed to load")]
    return []


def _served_plans_sustain(o: Observation) -> list[Failure]:
    """Loading is not serving: the window the planner published must be usable."""
    if o.planned and o.loaded and not o.sustained:
        return [
            Failure(
                "plan-sustains",
                o.cell_id,
                f"loaded at ctx={o.ctx} but could not sustain a full window",
            )
        ]
    return []


def _refusals_are_real(o: Observation) -> list[Failure]:
    """A refusal claims no usable window exists, so forcing one must not work.

    This is the general form of the 70B-on-2x4090 bug: the fit said 512 tokens
    while a pinned 4096 loaded and answered.
    """
    if o.planned or o.refusal is None:
        return []
    if o.forced_planned is False:
        return [
            Failure(
                "refusal-is-testable",
                o.cell_id,
                "a num_ctx pin produced no launch either, so the refusal could not be "
                "contradicted and this cell proves nothing",
            )
        ]
    if o.forced_loaded is None:
        return []
    if o.forced_loaded and o.forced_sustained:
        return [
            Failure(
                "refusal-is-real",
                o.cell_id,
                f"refused as unservable, but a forced {o.min_usable_ctx}-token window "
                "loaded and sustained",
            )
        ]
    return []


def _estimates_bracket_reality(o: Observation) -> list[Failure]:
    """Per device, what was charged has to resemble what was allocated."""
    if not (o.loaded and o.est_by_device and o.actual_by_device):
        return []
    failures = []
    for device, actual in o.actual_by_device.items():
        estimated = o.est_by_device.get(device)
        if estimated is None:
            failures.append(
                Failure("estimate-covers-devices", o.cell_id, f"{device} was never charged")
            )
            continue
        if actual - estimated > _UNDER_ESTIMATE_TOLERANCE:
            failures.append(
                Failure(
                    "estimate-not-under",
                    o.cell_id,
                    f"{device} allocated {actual / _GB:.2f} GiB against an estimate of "
                    f"{estimated / _GB:.2f} GiB",
                )
            )
        elif estimated - actual > _OVER_ESTIMATE_TOLERANCE:
            failures.append(
                Failure(
                    "estimate-not-wildly-over",
                    o.cell_id,
                    f"{device} was charged {estimated / _GB:.2f} GiB for "
                    f"{actual / _GB:.2f} GiB, which costs context elsewhere",
                )
            )
    return failures


def _oversize_models_spill(o: Observation) -> list[Failure]:
    """A model larger than the GPUs is served by spilling, not refused or OOMed.

    The tight placement's whole promise. It is also what breaks the moment the
    launch pins a tensor split, because the engine abandons its own fit pass.
    """
    if o.weights_bytes <= o.total_free_bytes or not o.planned:
        return []
    if not o.loaded:
        return [
            Failure(
                "oversize-spills",
                o.cell_id,
                f"weights {o.weights_bytes / _GB:.1f} GiB exceed {o.total_free_bytes / _GB:.1f} "
                "GiB of VRAM and the load failed instead of spilling to system memory",
            )
        ]
    return []


def _tight_groups_let_the_engine_fit(o: Observation) -> list[Failure]:
    """A tight group must not carry --tensor-split; the engine has to fit it."""
    if o.tight and o.cards > 1 and "--tensor-split" in o.argv:
        return [
            Failure(
                "tight-group-has-no-ratio",
                o.cell_id,
                "a best-effort placement pinned a tensor split, which aborts the "
                "engine's fit pass and offloads every layer",
            )
        ]
    return []


def compare(low: Observation, high: Observation, knob: str) -> list[Failure]:
    """Metamorphic check: more room must never serve a smaller window.

    Needs no oracle, only two runs whose order is known, which is what makes it
    able to catch configurations nobody thought to write an expectation for.
    """
    if low.skipped or high.skipped:
        return []
    if not (low.planned and high.planned):
        # A refusal on the roomier side while the tighter side serves is itself wrong.
        if high.planned and not low.planned and low.refusal:
            return []
        if low.planned and not high.planned:
            return [
                Failure(
                    "monotonic-service",
                    high.cell_id,
                    f"more {knob} than {low.cell_id}, yet chat was refused",
                )
            ]
        return []
    if high.ctx < low.ctx:
        return [
            Failure(
                "monotonic-ctx",
                high.cell_id,
                f"more {knob} than {low.cell_id} but a smaller window ({high.ctx} < {low.ctx})",
            )
        ]
    return []


def judge_all(observations: Sequence[Observation]) -> list[Failure]:
    """Per-cell invariants for every observation, most severe first by rule name."""
    return [failure for observation in observations for failure in judge(observation)]
