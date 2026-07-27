"""What the engine actually allocated, read back from its own startup report.

The plan is otherwise open-loop: every budget in :mod:`lilbee.providers.fleet.planning`
is a pure function of a snapshot taken before launch, and nothing ever checks
whether it was right. A wrong estimate surfaces as a failed request much later,
with no way to tell an under-estimate from a genuinely full card.

llama.cpp prints its per-device buffer sizes on every load, so the truth is
already in the log. Reading it back turns silent estimator drift into one warning
naming the role, the estimate and the reality, and it costs a regex over a log
tail that is already on disk.

WHY A LOG AND NOT AN API. There is no API. ``llama_model_size`` gives the whole
model's weights and nothing per device; ``llama_state_get_size`` is session
state. The per-device figures come from ``ggml_backend_buffer_get_size`` on
buffer handles the server holds and never exposes, and llama-server's HTTP
surface carries none of it either: ``/props`` is model and template metadata,
``/metrics`` is token counters, and both were checked against a running server.
The log is the only place these numbers leave the process.

THE FORMAT THIS PARSES, and where it comes from upstream:

    src/llama-model.cpp     "%s: %12s model buffer size = %8.2f MiB"
    src/llama-kv-cache.cpp  "%s: %10s KV buffer size = %8.2f MiB"
    src/llama-context.cpp   "%s: %10s compute buffer size = %8.2f MiB"

Verified against llama.cpp build 9310 (e2ef8fe42), the build the checked-in
fixture was captured from, and again on build 9665 (e3a74b299) running on two
A40s, where both a single-card and a tensor-split load produced every line with
the CUDA0/CUDA1 device labels this module joins on. These are plain format
strings in upstream source, not an interface anyone has promised to keep, so
treat a version bump of the bundled engine as a change that can break this:
re-capture the fixture and confirm :func:`parse_device_buffers` still finds every
line. A build that stops matching is reported rather than swallowed (see
:func:`check_launch`), so the failure announces itself instead of turning the
check into decoration.
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
# so a report says what to compare with. Tracks the fixture rather than the
# shipped engine, because reproducing a drift report means re-running the parser
# against that exact capture.
#
# A landmark, not a gate. Refusing on a version mismatch would be the wrong
# check: the format held unchanged from this build through 9665, confirmed on
# two A40s, so a gate would have fired on every bump while the parser was
# working. What detects drift is the parse coming back empty on a load that
# finished, which cannot happen while the format is intact and cannot be missed
# once it is not.
VERIFIED_ENGINE_BUILD = "9310 (e2ef8fe42)"

# "load_tensors:  MTL0_Mapped model buffer size =    82.41 MiB", and its siblings.
#
# Deliberately does NOT enumerate the buffer kinds. Listing them by hand meant
# reading two logs and hardcoding the four that happened to appear, which
# silently dropped LoRA (every adapter), RS (every Mamba and RWKV model) and the
# DeepSeek V4 state buffer. Upstream is free to add another tomorrow. The shape
# is what is stable: a prefix, the device, some words, "buffer size = N MiB".
#
# What the "= N MiB" requirement keeps out is the point of writing it that way.
# llama.cpp has three other lines carrying these words that are not allocations:
# the self-check pair reading "compute buffer size is N MiB, matches expectation"
# and "... of N MiB, does not match expectation", plus ggml-opencl's "buffer size
# reduced from A to B". None uses "=", so none is counted.
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
# Devices that are host memory rather than a GPU. Two shapes, both from ggml:
# the CPU backend's own buffers (CPU, CPU_Mapped), and every GPU backend's
# pinned-host allocator, which it names "<backend>_Host" (ggml-cuda.cu returns
# GGML_CUDA_NAME "_Host", and ggml-sycl, ggml-vulkan and ggml-cann do the same).
# None of it occupies VRAM, so charging it to a card reports a phantom overrun on
# every partially offloaded model. Found on real CUDA hardware, where CUDA_Host
# was being counted as a third GPU.
# AMX is a CPU extension with its own buffer type name, so it reports beside the
# CPU's and is host memory just the same.
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


# The engine says this once the weights are in and it is wiring up slots. Its
# presence means the load finished, which is what separates "the report has not
# been written yet" from "this engine does not write one where we look".
#
# Deliberately matches only the word the engine has kept. It said "initializing
# slots" through b9665 and "initializing, n_slots = N" from b9829, and this gate
# is what arms the format-drift warning: pinning the older phrase meant a newer
# engine finished loading, parsed to nothing, and reported nothing, leaving every
# placement estimate silently unverified. The buffer lines this module actually
# reads have held identical across all three builds; it is the prose around them
# that moves, so the prose is matched as loosely as it can still be meaningful.
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
# environment rather than argv because the launch is planned before the data
# directory that holds these logs is chosen, and because neither affects sizing,
# which is what the estimate-versus-launch argv parity test covers.
# Both spellings, because the engine renamed them and lilbee has to work with
# whichever build is installed. common/arg.cpp registers LLAMA_ARG_LOG_FILE and
# LLAMA_ARG_LOG_VERBOSITY on current master; builds around 9310 read the same
# settings as LLAMA_LOG_FILE and LLAMA_LOG_VERBOSITY, verified by running both
# pairs against one. An unread variable costs nothing, and picking one meant the
# readback silently produced no log at all on half the builds in the wild.
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
