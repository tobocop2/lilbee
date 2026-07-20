"""The stats command's comparison guard: it refuses undeclared pairings.

resolve_comparison is the gate that decides whether the frozen fingerprint may
be stamped onto a pair of metrics files. It reads the arms from the files' own
run_tags so a mislabelled invocation cannot slip a different study past the
manifest.
"""

import json

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


def _scored(dataset, run_tag, metrics):
    return {
        "dataset": dataset,
        "run_tag": run_tag,
        "per_query": {m: {"q1": 1.0} for m in metrics},
        "aggregated": {m: 1.0 for m in metrics},
    }


def test_stats_refuses_a_metric_the_scored_files_do_not_carry(tmp_path):
    # Absent metrics reach compare() as empty vectors and come back as a
    # measured tie (n=0, delta 0.0, p=1.0), which is data that never existed.
    import argparse

    from evals.benchmark.cli import _cmd_stats

    manifest_path = tmp_path / "m.json"
    _manifest(_ABLATION).freeze(manifest_path)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_scored("scifact", "dense", ["MRR@10"])) + "\n")
    b.write_text(json.dumps(_scored("scifact", "w1.0", [])) + "\n")
    args = argparse.Namespace(
        manifest=manifest_path, metrics_a=a, metrics_b=b, out=tmp_path / "out.jsonl"
    )
    with pytest.raises(ValueError, match="absent from the scored files"):
        _cmd_stats(args)
