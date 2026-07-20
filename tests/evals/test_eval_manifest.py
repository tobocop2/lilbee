"""Preregistration manifest: validation invariants and the comparison guard.

The manifest's whole job is to make the frozen fingerprint mean something, so
these pin down which studies it accepts and which comparisons it will let the
fingerprint be stamped onto.
"""

import json

import pytest
from evals.benchmark.manifest import (
    ArmConfig,
    DatasetSpec,
    Manifest,
    ModelConfig,
    StatsConfig,
)


def _models(**over) -> ModelConfig:
    base = {"embedder": "bge", "generator": "gen-a", "judge": "judge-b"}
    base.update(over)
    return ModelConfig(**base)


def _manifest(arms=None, datasets=None, **over) -> Manifest:
    arms = (
        arms
        if arms is not None
        else [
            ArmConfig(name="lilbee-parity", system="lilbee", description="on"),
            ArmConfig(name="ragflow-default", system="ragflow", description="default"),
        ]
    )
    datasets = (
        datasets
        if datasets is not None
        else [
            DatasetSpec(name="scifact", loader="scifact", label_kind="native"),
        ]
    )
    fields = {
        "run_id": "run-1",
        "arms": arms,
        "models": _models(),
        "datasets": datasets,
        "metrics": ["nDCG@10", "MRR@10"],
        "stats": StatsConfig(),
    }
    fields.update(over)
    return Manifest(**fields)


def test_a_valid_cross_system_manifest_validates():
    _manifest().validate()


def test_a_single_system_ablation_is_a_valid_preregistration():
    # Two lilbee arms at different configs: the ablation the harness actually ran.
    ablation = [
        ArmConfig(name="dense", system="lilbee", description="dense only"),
        ArmConfig(name="w1.0", system="lilbee", description="full weighted fusion"),
    ]
    _manifest(arms=ablation).validate()


def test_exactly_two_arms_are_required():
    one = [ArmConfig(name="solo", system="lilbee", description="")]
    with pytest.raises(ValueError, match="exactly two arms"):
        _manifest(arms=one).validate()


def test_arm_names_must_be_distinct():
    dupes = [
        ArmConfig(name="same", system="lilbee", description=""),
        ArmConfig(name="same", system="ragflow", description=""),
    ]
    with pytest.raises(ValueError, match="distinct"):
        _manifest(arms=dupes).validate()


def test_unknown_arm_system_is_rejected():
    bad = [
        ArmConfig(name="a", system="lilbee", description=""),
        ArmConfig(name="b", system="elasticsearch", description=""),
    ]
    with pytest.raises(ValueError, match="unknown arm system"):
        _manifest(arms=bad).validate()


def test_judge_must_differ_from_generator():
    with pytest.raises(ValueError, match="judge model must differ"):
        _manifest(models=_models(judge="gen-a")).validate()


def test_nonzero_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be"):
        _manifest(temperature=0.7).validate()


def test_empty_metrics_are_rejected():
    with pytest.raises(ValueError, match="no metrics"):
        _manifest(metrics=[]).validate()


def test_empty_datasets_are_rejected():
    with pytest.raises(ValueError, match="no datasets"):
        _manifest(datasets=[]).validate()


def test_unknown_dataset_label_kind_is_rejected():
    bad = [DatasetSpec(name="x", loader="x", label_kind="made-up")]
    with pytest.raises(ValueError, match="unknown label_kind"):
        _manifest(datasets=bad).validate()


def test_require_declared_comparison_accepts_the_declared_pair():
    _manifest().require_declared_comparison("lilbee-parity", "ragflow-default", "scifact")


def test_require_declared_comparison_rejects_an_undeclared_arm():
    # This is the committed bug: arms w1.0/dense were compared under a manifest
    # that declares lilbee-parity/ragflow-default.
    with pytest.raises(ValueError, match="not declared in manifest"):
        _manifest().require_declared_comparison("w1.0", "dense", "scifact")


def test_require_declared_comparison_rejects_a_partial_pair():
    # One declared arm compared against an undeclared one.
    with pytest.raises(ValueError, match="not declared in manifest"):
        _manifest().require_declared_comparison("lilbee-parity", "dense", "scifact")


def test_require_declared_comparison_rejects_an_undeclared_dataset():
    with pytest.raises(ValueError, match="dataset 'fiqa' is not declared"):
        _manifest().require_declared_comparison("lilbee-parity", "ragflow-default", "fiqa")


def test_require_declared_comparison_rejects_comparing_an_arm_with_itself():
    with pytest.raises(ValueError, match="two distinct arms"):
        _manifest().require_declared_comparison("lilbee-parity", "lilbee-parity", "scifact")


def test_fingerprint_is_stable_across_equal_manifests():
    assert _manifest().fingerprint() == _manifest().fingerprint()


def test_fingerprint_changes_when_any_frozen_field_changes():
    base = _manifest().fingerprint()
    assert _manifest(metrics=["nDCG@10"]).fingerprint() != base


def test_freeze_then_load_round_trips_and_records_the_fingerprint(tmp_path):
    manifest = _manifest()
    out = tmp_path / "frozen.json"
    fingerprint = manifest.freeze(out)
    payload = json.loads(out.read_text())
    assert payload["fingerprint"] == fingerprint
    reloaded = Manifest.from_dict({k: v for k, v in payload.items() if k != "fingerprint"})
    assert reloaded.fingerprint() == fingerprint


def test_arm_and_dataset_name_helpers():
    manifest = _manifest()
    assert manifest.arm_names == {"lilbee-parity", "ragflow-default"}
    assert manifest.dataset_names == {"scifact"}
