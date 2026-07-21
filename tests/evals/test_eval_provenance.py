"""Run provenance: the physical facts a published benchmark has to state.

Whether nvidia-smi reports utilisation is the driver's business. What is tested
here is the harness' side: that a duration is measured rather than declared,
that throughput is derived from it, that a machine with no GPU is recorded as
such instead of failing, and that two machines' records accumulate rather than
overwriting each other.
"""

import json

from evals.infra.provenance import Machine, RunProvenance, Stage, describe_machine, start


def test_a_stage_measures_its_own_duration():
    # A hand-written t1-t0 drifts from the block it claims to time the moment
    # anybody edits between them; the context manager cannot.
    run = start("test")
    with run.stage("work", documents=100) as stage:
        stage.bytes_out = 2048
    assert run.stages[0].wall_seconds > 0
    assert run.stages[0].documents_per_second > 0
    assert run.stages[0].bytes_out == 2048


def test_a_stage_records_even_when_the_body_raises():
    # A crashed ingest still cost money and still took time; losing the record
    # of that is how a post-mortem starts with no numbers.
    run = start("test")
    try:
        with run.stage("boom"):
            raise RuntimeError("ingest died")
    except RuntimeError:
        pass
    assert [s.name for s in run.stages] == ["boom"]
    assert run.stages[0].wall_seconds > 0


def test_a_cpu_only_machine_is_recorded_not_refused():
    # Hydration runs on a CPU box on purpose; a record that demanded a GPU would
    # be unusable for the stage where cheapness is the point.
    machine = Machine(host="h", platform="p", python="3.13", cpu_count=8, memory_gib=32.0)
    assert "CPU-only" in machine.gpu_summary


def test_the_gpu_summary_reads_as_a_reader_wants_it():
    from evals.infra.provenance import GPU

    machine = Machine(
        host="h",
        platform="p",
        python="3.13",
        cpu_count=8,
        memory_gib=32.0,
        gpus=[
            GPU(index=i, name="A100-SXM4-80GB", memory_mib=81920, driver="550") for i in range(4)
        ],
    )
    assert machine.gpu_summary == "4x A100-SXM4-80GB (80 GiB each)"


def test_cost_follows_the_billed_rate_and_the_measured_time():
    run = RunProvenance(stage_group="g", machine=describe_machine(), hourly_rate_usd=2.0)
    run.stages.append(Stage(name="s", wall_seconds=1800.0))
    assert run.total_cost_usd == 1.0


def test_records_from_two_machines_accumulate(tmp_path):
    # Hydration and ingest run on different boxes and the report wants both, so
    # the second must not erase the first.
    path = tmp_path / "provenance.jsonl"
    for group in ("hydrate", "ingest"):
        run = RunProvenance(stage_group=group, machine=describe_machine())
        run.stages.append(Stage(name="s", wall_seconds=1.0))
        run.write(path)
    groups = [json.loads(line)["stage_group"] for line in path.read_text().splitlines()]
    assert groups == ["hydrate", "ingest"]


def test_a_stalled_stage_is_distinguishable_from_a_busy_one():
    # The whole reason utilisation is sampled: four hours of compute and four
    # hours of waiting on MooseFS look identical in a duration alone.
    busy = Stage(name="b", wall_seconds=100.0, cpu_seconds=95.0)
    stalled = Stage(name="s", wall_seconds=100.0, cpu_seconds=2.0)
    assert busy.cpu_utilisation > 0.9
    assert stalled.cpu_utilisation < 0.1
