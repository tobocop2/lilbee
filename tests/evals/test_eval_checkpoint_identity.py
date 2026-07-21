"""A checkpoint may only resume the run that created it.

The checkpoint is keyed on the item id alone, so pointing a second arm at a file
that already holds the first arm's rows skips every item as done and produces a
complete, plausible artifact for arm B containing arm A's data. Nothing in the
file records which arm, endpoint, or configuration produced it.
"""

import pytest
from evals.retrieval.checkpoint import CheckpointMismatchError, JsonlCheckpoint


def test_a_fingerprinted_checkpoint_resumes_its_own_run(tmp_path):
    path = tmp_path / "ck.jsonl"
    JsonlCheckpoint(path, "qid", fingerprint={"arm": "A", "top_k": 20}).append({"qid": "q1"})
    resumed = JsonlCheckpoint(path, "qid", fingerprint={"arm": "A", "top_k": 20})
    assert "q1" in resumed


def test_resuming_another_arms_checkpoint_is_refused(tmp_path):
    # The copy-paste slip the two-command pod workflow invites.
    path = tmp_path / "ck.jsonl"
    JsonlCheckpoint(path, "qid", fingerprint={"arm": "A", "top_k": 20}).append({"qid": "q1"})
    with pytest.raises(CheckpointMismatchError, match="arm"):
        JsonlCheckpoint(path, "qid", fingerprint={"arm": "B", "top_k": 20})


def test_resuming_after_a_config_change_is_refused(tmp_path):
    # Same arm, different depth: the file would mix two configurations silently.
    path = tmp_path / "ck.jsonl"
    JsonlCheckpoint(path, "qid", fingerprint={"arm": "A", "top_k": 20}).append({"qid": "q1"})
    with pytest.raises(CheckpointMismatchError):
        JsonlCheckpoint(path, "qid", fingerprint={"arm": "A", "top_k": 50})


def test_the_fingerprint_is_not_returned_as_a_completed_item(tmp_path):
    path = tmp_path / "ck.jsonl"
    ck = JsonlCheckpoint(path, "qid", fingerprint={"arm": "A"})
    ck.append({"qid": "q1"})
    assert ck.done == {"q1"}
    assert len(JsonlCheckpoint(path, "qid", fingerprint={"arm": "A"}).done) == 1


def test_an_unfingerprinted_checkpoint_still_works(tmp_path):
    # Callers that genuinely have no configuration to bind stay supported.
    path = tmp_path / "ck.jsonl"
    JsonlCheckpoint(path, "qid").append({"qid": "q1"})
    assert "q1" in JsonlCheckpoint(path, "qid")


def test_a_legacy_file_without_a_fingerprint_is_refused_when_one_is_expected(tmp_path):
    # Pre-existing checkpoints carry no provenance, so they cannot be shown to
    # belong to this run and must not be silently adopted.
    path = tmp_path / "ck.jsonl"
    JsonlCheckpoint(path, "qid").append({"qid": "q1"})
    with pytest.raises(CheckpointMismatchError):
        JsonlCheckpoint(path, "qid", fingerprint={"arm": "A"})
