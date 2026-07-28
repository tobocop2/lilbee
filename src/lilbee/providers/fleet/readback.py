"""What the engine actually allocated, read back from its own startup report.

Compares the planner's estimate against reality per device, and warns naming the
role, the estimate and the reality when they diverge.

The log is the only source. There is no API: ``llama_model_size`` is the whole
model's weights and nothing per device, ``llama_state_get_size`` is session
state, the per-device figures come from ``ggml_backend_buffer_get_size`` on
handles the server never exposes, and llama-server's HTTP surface carries none of
it (``/props`` is metadata, ``/metrics`` is token counters; both checked against
a running server).

The format, from upstream source:

    src/llama-model.cpp     "%s: %12s model buffer size = %8.2f MiB"
    src/llama-kv-cache.cpp  "%s: %10s KV buffer size = %8.2f MiB"
    src/llama-context.cpp   "%s: %10s compute buffer size = %8.2f MiB"

These are format strings, not a promised interface. A bundled-engine version bump
can break this: re-capture the fixture and confirm :func:`parse_device_buffers`
still finds every line. A build that stops matching is reported, not swallowed
(see :func:`check_launch`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

MIB = 1024 * 1024

# The build the checked-in fixture was captured from, named in the drift warning
# so a report says what to compare with. Tracks the fixture, not the shipped
# engine. A landmark, not a gate: drift is detected by the parse coming back
# empty on a load that finished, not by a version comparison.
VERIFIED_ENGINE_BUILD = "9310 (e2ef8fe42)"

# "load_tensors:  MTL0_Mapped model buffer size =    82.41 MiB", and its siblings.
#
# Matches the shape rather than a list of buffer kinds, so LoRA, RS and the
# DeepSeek V4 state buffer parse without being enumerated here.
#
# The "= N MiB" is load-bearing: it excludes the three lines carrying these words
# that are not allocations -- the self-check pair ("compute buffer size is N MiB,
# matches expectation" / "... does not match expectation") and ggml-opencl's
# "buffer size reduced from A to B". None uses "=".
#
# The device label is whatever the backend calls itself: CUDA0, MTL0, Vulkan1,
# CPU. A timestamp and level prefix the line under --log-file, so the match is
# not anchored to the start.
_BUFFER_RE = re.compile(r"\S+:\s+(?P<device>\S+)\s+.*?buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB")
# The engine names an mmapped weight buffer "<device>_Mapped" beside the same
# device's other buffers. Same memory, so the suffix is folded away rather than
# splitting one card's total across two keys.
_MAPPED_SUFFIX = "_Mapped"
# ggml names a row-split buffer "<backend>_Split", one allocation shared by every
# card in the split rather than a device of its own.
_SPLIT_SUFFIX = "_Split"
# Host memory rather than a GPU, in two shapes from ggml: the CPU backend's own
# buffers (CPU, CPU_Mapped, and AMX, which is a CPU extension), and every GPU
# backend's pinned-host allocator, named "<backend>_Host" by ggml-cuda, -sycl,
# -vulkan, -cann and -hip alike. Observed as CUDA_Host, Vulkan_Host, ROCm_Host.
# None of it occupies VRAM; charging it to a card reports a phantom overrun on
# every partially offloaded model.
_HOST_PREFIXES = ("CPU", "AMX")
_HOST_SUFFIX = "_Host"


def _is_host_device(device: str) -> bool:
    """Whether *device* names host memory rather than a GPU."""
    return device.startswith(_HOST_PREFIXES) or device.endswith(_HOST_SUFFIX)


def parse_device_buffers(text: str) -> dict[str, int]:
    """Bytes the engine reported allocating, per device label, from *text*.

    Sums the model, KV, compute and output buffers, which is the same total the
    estimate predicts. Empty when the text carries no buffer report: a load that
    failed before allocating, a log that has rotated past it, or an engine whose
    verbosity is below the level that prints it.
    """
    totals: dict[str, int] = {}
    for match in _BUFFER_RE.finditer(text):
        device = match.group("device")
        if device.endswith(_SPLIT_SUFFIX):
            # A row-split buffer is spread across every card in the split, so it
            # belongs to no single one. Keeping it would invent a device that the
            # per-device comparison then reports as an unplanned allocation.
            continue
        device = device.removesuffix(_MAPPED_SUFFIX)
        totals[device] = totals.get(device, 0) + int(float(match.group("mib")) * MIB)
    return totals


# Printed once the weights are in and slots are being wired up, so its presence
# separates "the report is not written yet" from "this engine writes none here".
#
# Matches only the word the engine has kept: "initializing slots" through b9665,
# "initializing, n_slots = N" from b9829. This gate arms the format-drift
# warning, so pinning either exact phrase would silence the warning on the other.
# The buffer lines held identical across all three builds; only the prose moved.
_LOAD_FINISHED_RE = re.compile(r"load_model:\s+initializing\b")
# "common_params_print_info: build 9310 (e2ef8fe42) with AppleClang ...", the
# engine's own first line. Carried into the format-drift warning so the report
# names the exact build to re-verify against.
_BUILD_RE = re.compile(r"build\s+(?P<build>\d+)\s+\((?P<commit>[0-9a-f]+)\)")


def engine_build(text: str) -> str:
    """The engine build the log was written by, or empty when it does not say."""
    match = _BUILD_RE.search(text)
    return f"{match.group('build')} ({match.group('commit')})" if match else ""


def load_finished(text: str) -> bool:
    """Whether the engine got far enough to have reported its buffers."""
    return _LOAD_FINISHED_RE.search(text) is not None


def device_footprint(text: str) -> int:
    """Total GPU bytes the engine reported, host buffers excluded."""
    return sum(
        size for device, size in parse_device_buffers(text).items() if not _is_host_device(device)
    )


def device_label(device: FleetDevice) -> str:
    """The name the engine prints for *device*, and the join between the two sides.

    ``ggml_backend_dev_name`` produces ``CUDA0`` / ``MTL0`` / ``Vulkan1``, which is
    the same token ``--device`` and ``--tensor-split`` take and the same one the
    buffer report is keyed by. Joining on it keeps the check out of the index-space
    ambiguity that ``FleetDevice.from_loader`` exists to mark.
    """
    return f"{device.backend}{device.index}"


def report_divergence(
    role: WorkerRole,
    model: str,
    estimated_bytes: int,
    actual_bytes: int,
    *,
    tolerance: float,
) -> bool:
    """Warn when the engine's real footprint diverges materially from the estimate.

    Returns whether a warning was emitted, so a caller can record that this
    instance has already been checked and not repeat it on every request.

    Both directions are worth saying. An under-estimate is how a plan that fit on
    paper OOMs, and it is the one that ends in a failed load. A large
    over-estimate is quieter but costs capacity: it is why a role gets fewer
    slots, a narrower context, or a split it did not need.
    """
    if estimated_bytes <= 0 or actual_bytes <= 0:
        return False
    ratio = actual_bytes / estimated_bytes
    if abs(ratio - 1.0) <= tolerance:
        return False
    log.warning(
        "The %s model %s allocated %.1f GiB of GPU memory but was planned for %.1f GiB "
        "(%+.0f%%). Placement decisions for this model were made on the smaller figure; "
        "if it fails to load or runs slowly, that gap is why.",
        role.value,
        model,
        actual_bytes / 1024**3,
        estimated_bytes / 1024**3,
        (ratio - 1.0) * 100,
    )
    return True


# The engine's own log, one per instance, beside the swap process's log. Named
# by model id so a role's replicas do not overwrite each other.
_ENGINE_LOG_TEMPLATE = "engine-{model_id}.log"
# Env the engine reads for its log destination and threshold. Set through the
# environment rather than argv: the launch is planned before the data directory
# holding these logs is chosen, and neither affects sizing.
#
# Both spellings, because the engine renamed them. common/arg.cpp registers
# LLAMA_ARG_LOG_FILE / LLAMA_ARG_LOG_VERBOSITY on master; builds around 9310 read
# LLAMA_LOG_FILE / LLAMA_LOG_VERBOSITY. Verified against both. An unread variable
# costs nothing; picking one produced no log at all on half the builds in use.
ENV_LOG_FILE = "LLAMA_LOG_FILE"
ENV_LOG_VERBOSITY = "LLAMA_LOG_VERBOSITY"
ENV_ARG_LOG_FILE = "LLAMA_ARG_LOG_FILE"
ENV_ARG_LOG_VERBOSITY = "LLAMA_ARG_LOG_VERBOSITY"
# Level 4 ("trace") is where the per-device buffer report appears. Measured
# against the bundled engine: the default 3 omits it entirely, and 5 adds a
# per-layer and per-slot flood for the same six lines.
LOAD_REPORT_VERBOSITY = "4"


def engine_log_path(log_dir: Path, model_id: str) -> Path:
    """Where the engine serving *model_id* writes its own log."""
    return log_dir / _ENGINE_LOG_TEMPLATE.format(model_id=model_id)


def engine_log_env(log_dir: Path, model_id: str) -> dict[str, str]:
    """Environment that makes the engine report what it allocated, and where."""
    path = str(engine_log_path(log_dir, model_id))
    return {
        ENV_LOG_FILE: path,
        ENV_ARG_LOG_FILE: path,
        ENV_LOG_VERBOSITY: LOAD_REPORT_VERBOSITY,
        ENV_ARG_LOG_VERBOSITY: LOAD_REPORT_VERBOSITY,
    }


def check_launch(
    log_dir: Path,
    model_id: str,
    role: WorkerRole,
    model: str,
    estimated_bytes: int,
    est_by_device: dict[str, int] | None = None,
    unreported_bytes: int = 0,
) -> bool:
    """Compare the engine's own report for *model_id* against the estimate.

    Checked per device when *est_by_device* says what each card was planned for,
    because per device is the only dimension the planner decides in: a split is a
    ratio, a placement is a card, and a shortfall is recorded against a role on a
    card. Two cards planned 50/50 that land 80/20 sum to exactly the planned
    total, so a scalar comparison sees nothing while card 0 is the one that runs
    out. Falls back to the total for a model the estimator could only size as one
    number.

    Three outcomes, and the third is the one that matters. The engine has no API
    for any of this: /props carries no memory keys and /metrics is token
    counters, both checked against a running server, so its log is the only
    place these numbers exist. That makes this the one part of the fleet whose
    input is a format nobody promises to keep.

    So a load that finished without a readable report is reported, not swallowed.
    Left silent it would look exactly like a correct estimate, and the check
    would quietly become decoration the first time llama.cpp renames a line or
    renumbers its verbosity levels. Loud, it names itself as the thing to fix.
    """
    try:
        text = engine_log_path(log_dir, model_id).read_text(errors="replace")
    except OSError:
        # No log at all. Usually the engine simply has not written one yet, so
        # this is silent by default. It is also exactly what a wrong environment
        # variable name looks like, which is how an earlier spelling went
        # unnoticed: the check returned False forever and read as "estimate fine".
        # report_missing_log is how a caller that knows the engine is up says so.
        return False
    per_device = {
        label: size
        for label, size in parse_device_buffers(text).items()
        if not _is_host_device(label)
    }
    actual = sum(per_device.values())
    if actual > 0 and est_by_device:
        return _report_per_device(
            role, model, _without_unreported(est_by_device, unreported_bytes), per_device
        )
    if actual <= 0:
        if load_finished(text):
            log.warning(
                "The %s engine (build %s) finished loading but reported no memory usage where "
                "lilbee reads it, so its estimate could not be checked. The engine's log format "
                "or verbosity levels have most likely changed since build %s, which lilbee's "
                "parser was written against; placement estimates are unverified until it is "
                "updated to match.",
                role.value,
                engine_build(text) or "unknown",
                VERIFIED_ENGINE_BUILD,
            )
        return False
    return report_divergence(
        role, model, estimated_bytes - unreported_bytes, actual, tolerance=_TOLERANCE
    )


def _without_unreported(est_by_device: dict[str, int], unreported: int) -> dict[str, int]:
    """*est_by_device* less the bytes the engine allocates without reporting them.

    Charged to the busiest device, which is where the planner put them: a vision
    projector loads on the main GPU rather than across a split. Comparing the
    full estimate against a report that structurally cannot contain these bytes
    warns on every correctly sized vision load.
    """
    if unreported <= 0 or not est_by_device:
        return est_by_device
    main = max(est_by_device, key=lambda label: est_by_device[label])
    adjusted = dict(est_by_device)
    adjusted[main] = max(0, adjusted[main] - unreported)
    return adjusted


# How far the engine may land from the estimate before it is worth saying. Wide
# enough that the estimator's normal error is quiet, narrow enough to catch the
# whole-slot and whole-cache mistakes this exists to surface.
_TOLERANCE = 0.25


def _report_per_device(
    role: WorkerRole,
    model: str,
    estimated: dict[str, int],
    actual: dict[str, int],
) -> bool:
    """Warn about the card that diverged worst, naming both figures.

    One warning rather than one per card: the operator needs to know the plan did
    not hold and which card to look at, and a split that skews puts every card out
    at once by construction.
    """
    worst_label, worst_gap, worst_over = "", 0.0, False
    for label in set(estimated) | set(actual):
        planned, landed = estimated.get(label, 0), actual.get(label, 0)
        gap = abs(landed - planned) / planned if planned else float(landed)
        over = landed > planned
        # An overrun outranks an equal shortfall: a card holding more than it was
        # planned for is the one that fails to load, while its partner holding
        # less is only the symptom of the same skew.
        if (over, gap) > (worst_over, worst_gap):
            worst_label, worst_gap, worst_over = label, gap, over
    if not worst_label or (estimated.get(worst_label) and worst_gap <= _TOLERANCE):
        return False
    log.warning(
        "The %s model %s did not land where it was planned: %s holds %.1f GiB but was "
        "planned for %.1f GiB. Placement, the tensor split and the context were all "
        "decided per card, so a total that looks right can still overrun one of them.",
        role.value,
        model,
        worst_label,
        actual.get(worst_label, 0) / 1024**3,
        estimated.get(worst_label, 0) / 1024**3,
    )
    return True


def report_missing_log(log_dir: Path, model_id: str, role: WorkerRole) -> bool:
    """Warn when a ready engine wrote no log where lilbee told it to.

    Separate from :func:`check_launch` because only the caller knows the engine
    finished loading; an absent file before that is ordinary. After it, the file
    should exist, and its absence means the engine never accepted the settings
    that produce it. That is a silent no-op rather than a wrong answer, which is
    the harder kind to notice, so it is stated.
    """
    if engine_log_path(log_dir, model_id).exists():
        return False
    log.warning(
        "The %s engine is running but wrote no log to %s, so its memory use could not "
        "be checked against the estimate. The engine build most likely does not read "
        "the variables lilbee sets to ask for one (%s or %s); placement estimates are "
        "unverified until that is updated.",
        role.value,
        engine_log_path(log_dir, model_id),
        ENV_LOG_FILE,
        ENV_ARG_LOG_FILE,
    )
    return True
