"""The stats command's comparison guard: it refuses undeclared pairings.

resolve_comparison is the gate that decides whether the frozen fingerprint may
be stamped onto a pair of metrics files. It reads the arms from the files' own
run_tags so a mislabelled invocation cannot slip a different study past the
manifest.
"""

import pytest
from evals.benchmark.cli import resolve_comparison
from evals.benchmark.manifest import (
    ArmConfig,
    DatasetSpec,
    Manifest,
    ModelConfig,
    StatsConfig,
)


def _manifest(arms, dataset="scifact") -> Manifest:
    return Manifest(
        run_id="run-1",
        arms=arms,
        models=ModelConfig(embedder="bge", generator="gen", judge="judge"),
        datasets=[DatasetSpec(name=dataset, loader=dataset, label_kind="native")],
        metrics=["MRR@10"],
        stats=StatsConfig(),
    )


def _file(dataset, run_tag):
    return {"dataset": dataset, "run_tag": run_tag, "per_query": {}, "aggregated": {}}


_ABLATION = [
    ArmConfig(name="dense", system="lilbee", description=""),
    ArmConfig(name="w1.0", system="lilbee", description=""),
]


def test_resolve_returns_the_dataset_and_arms_from_the_files():
    manifest = _manifest(_ABLATION)
    dataset, arm_a, arm_b = resolve_comparison(
        manifest, _file("scifact", "dense"), _file("scifact", "w1.0")
    )
    assert (dataset, arm_a, arm_b) == ("scifact", "dense", "w1.0")


def test_resolve_rejects_two_different_datasets():
    manifest = _manifest(_ABLATION)
    with pytest.raises(ValueError, match="different datasets"):
        resolve_comparison(manifest, _file("scifact", "dense"), _file("fiqa", "w1.0"))


def test_resolve_rejects_arms_the_manifest_does_not_declare():
    # A manifest frozen for a parity study cannot stamp an ablation comparison.
    parity = [
        ArmConfig(name="lilbee-parity", system="lilbee", description=""),
        ArmConfig(name="ragflow-default", system="ragflow", description=""),
    ]
    manifest = _manifest(parity)
    with pytest.raises(ValueError, match="not declared in manifest"):
        resolve_comparison(manifest, _file("scifact", "dense"), _file("scifact", "w1.0"))


def test_resolve_rejects_a_dataset_absent_from_the_manifest():
    manifest = _manifest(_ABLATION, dataset="scifact")
    with pytest.raises(ValueError, match="not declared in manifest"):
        resolve_comparison(manifest, _file("nfcorpus", "dense"), _file("nfcorpus", "w1.0"))
