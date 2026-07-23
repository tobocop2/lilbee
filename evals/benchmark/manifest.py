"""The preregistration manifest: everything frozen before any data moves.

Freezing the datasets, metrics, every arm's config, and the held-constant models
to a file and its fingerprint is what stops a run from being cherry-picked after
the numbers land. The structural guarantee that matters most is that the judge
differs from the generator, so an answer is never graded by the model that wrote
it.

Pydantic owns the shape. It was already a dependency, arriving through ragas and
instructor, while this module hand-rolled parsing, per-field type checking,
``to_dict``/``from_dict``, and its own validation functions -- all of which
pydantic does, with better errors, for free. What is left here is the part
pydantic cannot know: which combinations of otherwise-valid values are a
preregistration this study is willing to stand behind.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.benchmark.datasets import LABEL_DERIVED, LABEL_NATIVE

LILBEE_SYSTEM = "lilbee"
KNOWN_SYSTEMS = frozenset({LILBEE_SYSTEM})
FROZEN_TEMPERATURE = 0.0
MIN_ARMS = 2


class Frozen(BaseModel):
    """Immutable, and refuses a key it does not recognise.

    ``extra="forbid"`` is the point of the base class. A manifest is meant to be
    read by a person and stamped with a fingerprint, so a misspelled key that
    silently does nothing is the worst outcome: the run proceeds under a
    configuration nobody chose, and the fingerprint attests to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArmConfig(Frozen):
    """One system under test and the configuration it runs with."""

    name: str
    system: str
    description: str
    config: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(Frozen):
    """The models held constant across every arm.

    ``embedder`` is served once and shared. ``generator`` writes every answer;
    ``judge`` grades them and must differ from ``generator``. ``generator_swap``
    records an alternate generator noted for a sensitivity check, not used in
    the main run.
    """

    embedder: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    judge: str = Field(min_length=1)
    generator_swap: str = ""

    @model_validator(mode="after")
    def _judge_differs_from_generator(self) -> ModelConfig:
        if self.judge == self.generator:
            raise ValueError(
                "judge model must differ from the generator model, or answers are "
                "graded by the model that wrote them"
            )
        return self


class DatasetSpec(Frozen):
    """One dataset, its loader, and whether its qrels are native or derived.

    ``loader`` is an ir_datasets id for a native set (``beir/fiqa/test``), which
    names the corpus, the split, and the published copy in one string. The split
    is deliberately not a separate field: two places declaring it is two places
    to disagree, and the id is what selects the data.
    """

    name: str
    loader: str
    label_kind: str

    @model_validator(mode="after")
    def _known_label_kind(self) -> DatasetSpec:
        if self.label_kind not in (LABEL_NATIVE, LABEL_DERIVED):
            raise ValueError(
                f"dataset '{self.name}' has unknown label_kind '{self.label_kind}'; "
                f"expected '{LABEL_NATIVE}' or '{LABEL_DERIVED}'"
            )
        return self


class StatsConfig(Frozen):
    """Paired-statistics configuration, frozen so intervals are reproducible."""

    bootstrap_resamples: int = 10000
    seed: int = 20260714
    alpha: float = 0.05


class SystemProvenance(Frozen):
    """Which build produced the runs, and how the corpus was indexed.

    An earlier study pinned the embedder and nothing else, so its run files could
    be rescored but not reproduced: no commit, no index parameters. A reader who
    cannot rebuild the system under test is being asked to take the numbers on
    trust, which is the thing a benchmark exists to avoid.

    Every field is empty by default so a manifest frozen before this existed
    keeps its identity, but a run that leaves them empty publishes an
    unreproducible result and ``require_reproducible`` says so.
    """

    lilbee_commit: str = ""
    lilbee_version: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    reranker: str = ""
    index_built_at: str = ""

    @property
    def is_complete(self) -> bool:
        return bool(self.lilbee_commit and self.chunk_size)


# Identity fields added after manifests were already frozen. Omitting them when
# empty keeps those fingerprints valid; a populated value still changes the
# identity. Only fields introduced later belong here: excluding one that was
# already hashed would re-identify every study that recorded it.
OPTIONAL_IDENTITY_FIELDS = (
    "lilbee_commit",
    "lilbee_version",
    "reranker",
    "index_built_at",
)


def _without_empty_optionals(payload: Any) -> Any:
    """Drop optional identity fields left empty, recursively."""
    if isinstance(payload, dict):
        return {
            key: _without_empty_optionals(value)
            for key, value in payload.items()
            if not (key in OPTIONAL_IDENTITY_FIELDS and value == "")
        }
    if isinstance(payload, list):
        return [_without_empty_optionals(item) for item in payload]
    return payload


class Manifest(Frozen):
    """The full frozen preregistration for one benchmark run."""

    run_id: str
    arms: list[ArmConfig]
    models: ModelConfig
    datasets: list[DatasetSpec]
    metrics: list[str]
    stats: StatsConfig = Field(default_factory=StatsConfig)
    temperature: float = FROZEN_TEMPERATURE
    system: SystemProvenance = Field(default_factory=SystemProvenance)
    # Present on a frozen file, absent on a hand-written one. Kept out of the
    # canonical form so a manifest cannot hash its own hash.
    fingerprint_on_file: str | None = Field(default=None, alias="fingerprint")

    @model_validator(mode="after")
    def _valid_preregistration(self) -> Manifest:
        """The invariants pydantic cannot infer from the field types alone."""
        # Emptiness is checked here rather than with Field(min_length=1) so the
        # message names the thing a person has to go and fix. A manifest is read
        # by an operator on a pod, and "List should have at least 1 item after
        # validation, not 0" is not what they need to see at that moment.
        if not self.metrics:
            raise ValueError("manifest lists no metrics")
        if not self.datasets:
            raise ValueError("manifest lists no datasets")
        if len(self.arms) < MIN_ARMS:
            raise ValueError("a benchmark needs at least two arms to compare")
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be distinct")
        # Every arm is a lilbee configuration: the study grades lilbee against
        # itself (a baseline against feature variants), so an arm naming any
        # other system is a mistake the stamped fingerprint would otherwise hide.
        unknown = {arm.system for arm in self.arms} - KNOWN_SYSTEMS
        if unknown:
            raise ValueError(
                f"unknown arm system(s) {sorted(unknown)}; each arm's system must be "
                f"one of {sorted(KNOWN_SYSTEMS)}"
            )
        if self.temperature != FROZEN_TEMPERATURE:
            raise ValueError(f"temperature must be {FROZEN_TEMPERATURE} for a deterministic run")
        stored = self.fingerprint_on_file
        # Recomputing the fingerprint from whatever the file now says would make
        # any post-hoc edit self-consistent, so the one artifact whose purpose is
        # to be tamper-evident would report no tampering.
        if stored is not None and stored != self.fingerprint():
            raise ValueError(
                f"manifest fingerprint does not match its contents (file {stored[:12]}, "
                f"contents hash to {self.fingerprint()[:12]}); the frozen "
                "preregistration was edited after it was frozen"
            )
        return self

    def require_reproducible(self) -> None:
        """Fail unless the manifest records which build produced the runs."""
        if not self.system.is_complete:
            raise ValueError(
                "manifest does not record the system under test: set "
                "system.lilbee_commit and system.chunk_size so the run can be "
                "reproduced, not merely rescored"
            )

    def to_dict(self) -> dict[str, Any]:
        """The manifest's content, without the fingerprint stamped onto a file."""
        return self.model_dump(exclude={"fingerprint_on_file"})

    def fingerprint(self) -> str:
        """Stable sha256 over the canonical JSON; the preregistration's identity.

        Optional identity fields never filled in are omitted, so adding one to
        the schema does not silently re-identify every study that predates it. A
        populated value does change the fingerprint, which is the point: a run
        against a different build or index configuration is a different
        preregistration.
        """
        canonical = json.dumps(
            _without_empty_optionals(self.to_dict()), sort_keys=True, separators=(",", ":")
        )
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
        """Fail unless this manifest declares both arms and the dataset.

        The fingerprint is the preregistration's identity; stamping it onto a
        comparison the manifest never declared attests to a study that was not
        performed.
        """
        undeclared = {arm_a, arm_b} - self.arm_names
        if undeclared:
            raise ValueError(
                f"arms {sorted(undeclared)} are not declared in manifest "
                f"'{self.run_id}' (declares {sorted(self.arm_names)}); "
                "the frozen fingerprint cannot attest to this comparison"
            )
        if arm_a == arm_b:
            raise ValueError(f"a comparison needs two distinct arms, both are '{arm_a}'")
        if dataset not in self.dataset_names:
            raise ValueError(
                f"dataset '{dataset}' is not declared in manifest '{self.run_id}' "
                f"(declares {sorted(self.dataset_names)})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Build and validate from a manifest payload."""
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load and validate a manifest from a YAML or JSON file."""
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            return cls.model_validate(yaml.safe_load(text))
        return cls.model_validate(json.loads(text))

    def freeze(self, path: Path) -> str:
        """Write the canonical manifest to ``path`` and return its fingerprint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.to_dict(), "fingerprint": self.fingerprint()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload["fingerprint"]
