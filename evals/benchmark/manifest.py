"""The preregistration manifest: everything frozen before any data moves.

Freezing the datasets, metrics, both arms' configs, and the held-constant
models to a file (and its fingerprint) is what stops the run from being
cherry-picked after the numbers land. The key structural guarantee validated
here is that the judge model differs from the generator model, so the answer
tier is never graded by the same model that wrote the answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.benchmark.datasets import LABEL_DERIVED, LABEL_NATIVE

LILBEE_SYSTEM = "lilbee"
RAGFLOW_SYSTEM = "ragflow"
FROZEN_TEMPERATURE = 0.0


@dataclass(frozen=True)
class ArmConfig:
    """One system under test and the configuration it runs with."""

    name: str
    system: str
    description: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """The models held constant across both arms.

    ``embedder`` is served once and shared by both arms. ``generator`` writes
    every answer; ``judge`` grades them and MUST differ from ``generator``.
    ``generator_swap`` records an alternate generator noted for a sensitivity
    check, not used in the main run.
    """

    embedder: str
    generator: str
    judge: str
    generator_swap: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset, its loader, and whether its qrels are native or derived."""

    name: str
    loader: str
    label_kind: str
    split: str = "test"


@dataclass(frozen=True)
class StatsConfig:
    """Paired-statistics configuration, frozen so CIs are reproducible."""

    bootstrap_resamples: int = 10000
    seed: int = 20260714
    alpha: float = 0.05


@dataclass(frozen=True)
class Manifest:
    """The full frozen preregistration for one benchmark run."""

    run_id: str
    arms: list[ArmConfig]
    models: ModelConfig
    datasets: list[DatasetSpec]
    metrics: list[str]
    stats: StatsConfig
    temperature: float = FROZEN_TEMPERATURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable sha256 over the canonical JSON; the preregistration's identity."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def derived_datasets(self) -> list[str]:
        return [ds.name for ds in self.datasets if ds.label_kind == LABEL_DERIVED]

    @property
    def arm_names(self) -> set[str]:
        return {arm.name for arm in self.arms}

    @property
    def dataset_names(self) -> set[str]:
        return {ds.name for ds in self.datasets}

    def require_declared_comparison(self, arm_a: str, arm_b: str, dataset: str) -> None:
        """Fail unless this manifest declares exactly this pair of arms and dataset.

        The fingerprint is the preregistration's identity; stamping it onto a
        comparison the manifest never declared attests to a study that was not
        performed. A comparison is declared only when both arms are named in the
        manifest, they are the two distinct declared arms, and the dataset is one
        the manifest lists.
        """
        undeclared_arms = {arm_a, arm_b} - self.arm_names
        if undeclared_arms:
            raise ValueError(
                f"arms {sorted(undeclared_arms)} are not declared in manifest "
                f"'{self.run_id}' (declares {sorted(self.arm_names)}); "
                "the frozen fingerprint cannot attest to this comparison"
            )
        if arm_a == arm_b:
            raise ValueError(f"a comparison needs two distinct arms, both are '{arm_a}'")
        if {arm_a, arm_b} != self.arm_names:
            raise ValueError(
                f"manifest '{self.run_id}' declares arms {sorted(self.arm_names)}, "
                f"but the comparison is between {sorted({arm_a, arm_b})}"
            )
        if dataset not in self.dataset_names:
            raise ValueError(
                f"dataset '{dataset}' is not declared in manifest '{self.run_id}' "
                f"(declares {sorted(self.dataset_names)})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        manifest = cls(
            run_id=data["run_id"],
            arms=[ArmConfig(**arm) for arm in data["arms"]],
            models=ModelConfig(**data["models"]),
            datasets=[DatasetSpec(**ds) for ds in data["datasets"]],
            metrics=list(data["metrics"]),
            stats=StatsConfig(**data.get("stats", {})),
            temperature=data.get("temperature", FROZEN_TEMPERATURE),
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load and validate a manifest from a YAML or JSON file."""
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def freeze(self, path: Path) -> str:
        """Write the canonical manifest to ``path`` and return its fingerprint."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload["fingerprint"]

    def validate(self) -> None:
        """Fail loudly on any preregistration invariant the run depends on."""
        _validate_arms(self.arms)
        _validate_models(self.models)
        _validate_datasets(self.datasets)
        if not self.metrics:
            raise ValueError("manifest lists no metrics")
        if self.temperature != FROZEN_TEMPERATURE:
            raise ValueError(f"temperature must be {FROZEN_TEMPERATURE} for a deterministic run")


KNOWN_SYSTEMS = frozenset({LILBEE_SYSTEM, RAGFLOW_SYSTEM})


def _validate_arms(arms: list[ArmConfig]) -> None:
    if len(arms) != 2:  # noqa: PLR2004 - a paired comparison has exactly two arms
        raise ValueError("a paired benchmark needs exactly two arms")
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be distinct")
    # Both a cross-system parity study (one lilbee, one ragflow) and a
    # single-system ablation (two lilbee arms at different configs) are valid
    # preregistrations; forcing one of each made the ablation impossible to
    # declare, so the run compared undeclared arms under a stamped fingerprint.
    unknown = {arm.system for arm in arms} - KNOWN_SYSTEMS
    if unknown:
        raise ValueError(
            f"unknown arm system(s) {sorted(unknown)}; each arm's system must be "
            f"one of {sorted(KNOWN_SYSTEMS)}"
        )


def _validate_models(models: ModelConfig) -> None:
    if not models.embedder:
        raise ValueError("a shared embedder model is required")
    if not models.generator:
        raise ValueError("a generator model is required")
    if models.judge == models.generator:
        raise ValueError("judge model must differ from the generator model")


def _validate_datasets(datasets: list[DatasetSpec]) -> None:
    if not datasets:
        raise ValueError("manifest lists no datasets")
    for ds in datasets:
        if ds.label_kind not in (LABEL_NATIVE, LABEL_DERIVED):
            raise ValueError(f"dataset '{ds.name}' has unknown label_kind '{ds.label_kind}'")
