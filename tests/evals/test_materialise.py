"""Partitioning the corpus: the union of the shards must be the single-host run.

Everything downstream assumes this. If the shards together write a different set
of files, or the same passages at different paths, the merged index is not the
index a single host would have built and no amount of care in the merge recovers
it. lilbee stores the path as the source of every chunk, so a path difference is
a data difference, not a cosmetic one.
"""

import pathlib
from dataclasses import dataclass

import pytest
from evals.infra.materialise import materialise


@dataclass
class Passage:
    doc_id: str
    text: str


def corpus(n: int, *, blanks: set[int] | None = None) -> list[Passage]:
    """n passages, with the given positions blank so the empty-skip is exercised."""
    blanks = blanks or set()
    return [Passage(f"p{i}", "" if i in blanks else f"passage {i}") for i in range(n)]


def tree(root: pathlib.Path) -> dict[str, str]:
    """Every written file as {relative path: contents}."""
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("shard_count", [2, 3, 5])
def test_the_shards_together_write_exactly_what_one_host_writes(tmp_path, shard_count):
    passages = corpus(2500)

    single = tmp_path / "single"
    single.mkdir()
    materialise(passages, single, bucket_size=100)

    merged: dict[str, str] = {}
    for index in range(shard_count):
        out = tmp_path / f"shard{index}"
        out.mkdir()
        written, scanned = materialise(
            passages, out, shard_index=index, shard_count=shard_count, bucket_size=100
        )
        assert scanned == len(passages)
        shard_tree = tree(out)
        assert len(shard_tree) == written
        overlap = merged.keys() & shard_tree.keys()
        assert not overlap, f"shards {index} and earlier both wrote {sorted(overlap)[:3]}"
        merged.update(shard_tree)

    assert merged == tree(single)


def test_a_passage_lands_in_the_same_bucket_however_many_shards_run(tmp_path):
    # The defect: the bucket came from the shard's local counter, so with two
    # shards the passage a single host puts in 00002/ landed in 00001/. The file
    # set still looked plausible; the source paths were wrong.
    passages = corpus(300)

    single = tmp_path / "single"
    single.mkdir()
    materialise(passages, single, bucket_size=100)
    single_paths = tree(single)

    out = tmp_path / "shard1"
    out.mkdir()
    materialise(passages, out, shard_index=1, shard_count=2, bucket_size=100)

    for path in tree(out):
        assert path in single_paths, f"shard wrote {path}, which a single host never writes"
    # The last shard-1 passage is global index 299, which belongs in bucket 2.
    assert "00002/p299.txt" in tree(out)


def test_blank_passages_are_skipped_before_the_index_is_assigned(tmp_path):
    # The global index counts non-empty passages only. If a blank consumed an
    # index the shards would still be disjoint but would not match a single
    # host, whose indices also skip blanks.
    passages = corpus(200, blanks={3, 7, 50})

    single = tmp_path / "single"
    single.mkdir()
    written, scanned = materialise(passages, single, bucket_size=100)
    assert written == 197
    assert scanned == 197

    merged: dict[str, str] = {}
    for index in range(2):
        out = tmp_path / f"s{index}"
        out.mkdir()
        materialise(passages, out, shard_index=index, shard_count=2, bucket_size=100)
        merged.update(tree(out))
    assert merged == tree(single)
    assert not any(name.endswith(("p3.txt", "p7.txt", "p50.txt")) for name in merged)


def test_a_smoke_cap_slices_the_same_passages_the_full_run_would(tmp_path):
    # SMOKE_N caps the global corpus, then the shards split that capped set. A
    # shard applying the cap to its own output would take a different slice.
    passages = corpus(1000)

    single = tmp_path / "single"
    single.mkdir()
    written, _ = materialise(passages, single, smoke=250, bucket_size=100)
    assert written == 250

    merged: dict[str, str] = {}
    for index in range(2):
        out = tmp_path / f"s{index}"
        out.mkdir()
        materialise(passages, out, shard_index=index, shard_count=2, smoke=250, bucket_size=100)
        merged.update(tree(out))
    assert merged == tree(single)


def test_each_shard_writes_its_arithmetic_share(tmp_path):
    # checkpoint.sh derives its milestone target from the same arithmetic, so
    # the two have to agree about how a corpus divides.
    passages = corpus(1003)
    for index in range(3):
        out = tmp_path / f"s{index}"
        out.mkdir()
        written, _ = materialise(passages, out, shard_index=index, shard_count=3, bucket_size=100)
        assert written == (1003 + 3 - 1 - index) // 3


def test_a_shard_index_outside_the_set_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"outside 0\.\.1"):
        materialise(corpus(10), tmp_path, shard_index=2, shard_count=2)
