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


def test_at_least_two_arms_are_required():
    one = [ArmConfig(name="solo", system="lilbee", description="")]
    with pytest.raises(ValueError, match="at least two arms"):
        _manifest(arms=one).validate()


def test_an_ablation_may_declare_a_baseline_and_several_variants():
    # One baseline against four fusion weights is one study, not four studies.
    ablation = [ArmConfig(name="dense", system="lilbee", description="")] + [
        ArmConfig(name=f"w{w}", system="lilbee", description="") for w in ("0.25", "0.5", "1.0")
    ]
    manifest = _manifest(arms=ablation)
    manifest.validate()
    manifest.require_declared_comparison("dense", "w0.5", "scifact")


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


def test_loading_a_frozen_manifest_verifies_its_stored_fingerprint(tmp_path):
    out = tmp_path / "frozen.json"
    _manifest().freeze(out)
    Manifest.load(out)  # unedited: verifies and loads


def test_an_edited_frozen_manifest_is_refused(tmp_path):
    # The one artifact whose entire purpose is to be tamper-evident. Recomputing
    # the fingerprint from whatever the file now says makes any edit
    # self-consistent and silent.
    out = tmp_path / "frozen.json"
    _manifest().freeze(out)
    payload = json.loads(out.read_text())
    payload["stats"]["seed"] = 999
    out.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        Manifest.load(out)


def test_a_manifest_without_a_stored_fingerprint_still_loads(tmp_path):
    # Hand-authored source manifests carry no fingerprint until frozen.
    out = tmp_path / "source.json"
    out.write_text(json.dumps(_manifest().to_dict()))
    Manifest.load(out)


def test_an_unfilled_identity_field_does_not_change_the_fingerprint():
    # Adding a field to the schema must not silently re-identify every study
    # frozen before it existed, or the stamped fingerprints stop matching.
    base = _manifest()
    with_empty = _manifest(
        datasets=[DatasetSpec(name="scifact", loader="scifact", label_kind="native", revision="")]
    )
    assert base.fingerprint() == with_empty.fingerprint()


def test_a_populated_identity_field_does_change_the_fingerprint():
    base = _manifest()
    pinned = _manifest(
        datasets=[
            DatasetSpec(name="scifact", loader="scifact", label_kind="native", revision="v1.0.0")
        ]
    )
    assert base.fingerprint() != pinned.fingerprint()
