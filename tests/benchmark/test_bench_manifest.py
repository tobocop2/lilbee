"""Manifest validation, the judge-not-generator guard, and freeze round-trip."""

from pathlib import Path

import pytest

from evals.benchmark.manifest import Manifest

EXAMPLE = Path(__file__).resolve().parents[2] / "evals" / "benchmark" / "manifest.example.yaml"


def _manifest_dict(**overrides):
    data = {
        "run_id": "r1",
        "models": {
            "embedder": "qwen3-embedding",
            "generator": "qwen2.5-72b",
            "judge": "fable-5",
        },
        "arms": [
            {"name": "a", "system": "lilbee", "description": "d"},
            {"name": "b", "system": "ragflow", "description": "d"},
        ],
        "datasets": [{"name": "scifact", "loader": "scifact", "label_kind": "native"}],
        "metrics": ["nDCG@10"],
    }
    data.update(overrides)
    return data


def test_from_dict_accepts_a_well_formed_manifest():
    manifest = Manifest.from_dict(_manifest_dict())
    assert manifest.run_id == "r1"
    assert manifest.temperature == 0.0


def test_judge_must_differ_from_generator():
    models = {"embedder": "e", "generator": "same", "judge": "same"}
    with pytest.raises(ValueError, match="judge model must differ"):
        Manifest.from_dict(_manifest_dict(models=models))


def test_requires_exactly_two_arms():
    arms = [{"name": "a", "system": "lilbee", "description": "d"}]
    with pytest.raises(ValueError, match="exactly two arms"):
        Manifest.from_dict(_manifest_dict(arms=arms))


def test_arms_must_be_one_lilbee_one_ragflow():
    arms = [
        {"name": "a", "system": "lilbee", "description": "d"},
        {"name": "b", "system": "lilbee", "description": "d"},
    ]
    with pytest.raises(ValueError, match="one lilbee and one ragflow"):
        Manifest.from_dict(_manifest_dict(arms=arms))


def test_nonzero_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be"):
        Manifest.from_dict(_manifest_dict(temperature=0.7))


def test_unknown_label_kind_is_rejected():
    datasets = [{"name": "x", "loader": "x", "label_kind": "guessed"}]
    with pytest.raises(ValueError, match="unknown label_kind"):
        Manifest.from_dict(_manifest_dict(datasets=datasets))


def test_derived_datasets_are_listed():
    datasets = [
        {"name": "scifact", "loader": "scifact", "label_kind": "native"},
        {"name": "tat-dqa", "loader": "tatdqa", "label_kind": "derived"},
    ]
    manifest = Manifest.from_dict(_manifest_dict(datasets=datasets))
    assert manifest.derived_datasets == ["tat-dqa"]


def test_freeze_writes_a_stable_fingerprint(tmp_path):
    manifest = Manifest.from_dict(_manifest_dict())
    out = tmp_path / "frozen.json"
    fingerprint = manifest.freeze(out)
    assert fingerprint == manifest.fingerprint()
    assert f'"fingerprint": "{fingerprint}"' in out.read_text()


def test_example_manifest_loads_and_validates():
    manifest = Manifest.load(EXAMPLE)
    assert manifest.models.judge != manifest.models.generator
    assert "tat-dqa" in manifest.derived_datasets
    assert manifest.metrics == ["nDCG@10", "Recall@20", "MRR@10"]
