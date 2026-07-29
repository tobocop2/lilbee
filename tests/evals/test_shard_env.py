"""The shard arithmetic the shell scripts run on.

checkpoint.sh decides which slot to push to and how far along this host is;
restore.sh decides which slot to pull back. Both get those from shard_env.sh, so
a mistake here does not fail loudly: it pushes a shard's index over another's,
or restores the wrong slice and lets the merge combine a corpus that is quietly
missing part of itself. The last test pins the shell arithmetic against the
Python that materialises the passages, because the two have to agree about how a
corpus divides or the milestones track the wrong denominator.
"""

import pathlib
import subprocess

import pytest
from evals.infra.materialise import materialise

SHARD_ENV = pathlib.Path(__file__).resolve().parents[2] / "evals" / "infra" / "shard_env.sh"


def run_shard_env(**env: str) -> dict[str, str]:
    """Source shard_env.sh and report what it set."""
    result = subprocess.run(
        ["bash", "-c", f'set -u; . "{SHARD_ENV}"; echo "$CKPT_PATH"; echo "${{SHARD_TOTAL:-}}"'],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "shard_env.sh", result.stderr)
    path, total = result.stdout.strip().splitlines()
    return {"CKPT_PATH": path, "SHARD_TOTAL": total}


def test_a_single_host_run_keeps_the_original_checkpoint_slot(tmp_path):
    # Checkpoints written before sharding existed live at this path; changing it
    # for single-host runs would strand them.
    out = run_shard_env(CHECKPOINT_TOTAL="8841823")
    assert out["CKPT_PATH"] == "checkpoint-latest.tar"
    assert out["SHARD_TOTAL"] == "8841823"


def test_each_shard_gets_its_own_slot():
    # The defect: every shard pushed to one rolling path, so the last writer won
    # and a pod loss restored some other shard's index.
    paths = {
        run_shard_env(SHARD_INDEX=str(i), SHARD_COUNT="3", CHECKPOINT_TOTAL="900")["CKPT_PATH"]
        for i in range(3)
    }
    assert len(paths) == 3, f"shards share a checkpoint slot: {paths}"
    assert paths == {
        "shard-0of3/checkpoint-latest.tar",
        "shard-1of3/checkpoint-latest.tar",
        "shard-2of3/checkpoint-latest.tar",
    }


def test_the_shard_totals_cover_the_corpus_exactly():
    # Milestones divide by SHARD_TOTAL. If the parts did not sum to the whole,
    # some host would either never reach 100% or claim it early.
    total = 8841823
    parts = [
        int(
            run_shard_env(SHARD_INDEX=str(i), SHARD_COUNT="4", CHECKPOINT_TOTAL=str(total))[
                "SHARD_TOTAL"
            ]
        )
        for i in range(4)
    ]
    assert sum(parts) == total
    assert max(parts) - min(parts) <= 1


@pytest.mark.parametrize(
    ("index", "count"),
    [("2", "2"), ("-1", "2"), ("5", "3")],
)
def test_a_shard_index_outside_the_set_refuses(index, count):
    # The red path: prove the guard can actually fail, rather than trusting it.
    with pytest.raises(subprocess.CalledProcessError):
        run_shard_env(SHARD_INDEX=index, SHARD_COUNT=count, CHECKPOINT_TOTAL="100")


def test_a_corpus_smaller_than_the_shard_count_still_divides():
    # SMOKE_N smaller than the shard count would otherwise floor to zero and
    # abort the watcher on its first division.
    out = run_shard_env(SHARD_INDEX="3", SHARD_COUNT="4", CHECKPOINT_TOTAL="2")
    assert int(out["SHARD_TOTAL"]) >= 1


class Passage:
    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text


@pytest.mark.parametrize("count", [2, 3, 5])
def test_the_shell_total_matches_what_the_python_actually_writes(tmp_path, count):
    # checkpoint.sh's denominator and ingest.sh's output come from two different
    # implementations of the same split. If they drift, progress reporting is
    # wrong and the final milestone fires at the wrong row count.
    corpus = [Passage(f"p{i}", f"passage {i}") for i in range(1003)]
    for index in range(count):
        out = tmp_path / f"{count}_{index}"
        out.mkdir()
        written, _ = materialise(corpus, out, shard_index=index, shard_count=count, bucket_size=100)
        predicted = run_shard_env(
            SHARD_INDEX=str(index), SHARD_COUNT=str(count), CHECKPOINT_TOTAL="1003"
        )["SHARD_TOTAL"]
        assert written == int(predicted), (
            f"shard {index}/{count}: shell predicts {predicted} passages, "
            f"materialise wrote {written}"
        )
