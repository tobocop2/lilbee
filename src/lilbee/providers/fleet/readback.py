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

Verified against llama.cpp build 9310 (e2ef8fe42), which is the build the
checked-in fixture was captured from. These are plain format strings in upstream
source, not an interface anyone has promised to keep, so treat a version bump of
the bundled engine as a change that can break this: re-capture the fixture and
confirm :func:`parse_device_buffers` still finds every line. A build that stops
matching is reported rather than swallowed (see :func:`check_launch`), so the
failure announces itself instead of turning the check into decoration.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

MIB = 1024 * 1024

# The llama.cpp build the buffer-report format above was verified against, and
# the one the checked-in fixture came from. Named in the drift warning so a
# report says what to compare with.
VERIFIED_ENGINE_BUILD = "9310 (e2ef8fe42)"

# "load_tensors:  MTL0_Mapped model buffer size =    82.41 MiB", plus the KV,
# compute and output lines that follow under different prefixes (load_tensors,
# llama_context, llama_kv_cache, sched_reserve). The device label is whatever the
# backend calls itself: CUDA0, MTL0, Vulkan1, CPU. Sizes are always MiB.
# A timestamp and level prefix the line when the engine writes to --log-file, so
# the match is not anchored to the start.
_BUFFER_RE = re.compile(
    r"\S+:\s+(?P<device>\S+)\s+(?:model|KV|compute|output)\s+"
    r"buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB"
)
# The engine names an mmapped weight buffer "<device>_Mapped" beside the same
# device's other buffers. Same memory, so the suffix is folded away rather than
# splitting one card's total across two keys.
_MAPPED_SUFFIX = "_Mapped"
# Devices that are host memory rather than a GPU. The engine names the mmapped
# weight buffer CPU_Mapped and its scratch CPU; neither occupies VRAM, so
# charging them against a card's budget would report a phantom overrun on every
# partially offloaded model.
_HOST_DEVICES = ("CPU",)


def _is_host_device(device: str) -> bool:
    """Whether *device* names host memory rather than a GPU."""
    return device.upper().startswith(_HOST_DEVICES)


def parse_device_buffers(text: str) -> dict[str, int]:
    """Bytes the engine reported allocating, per device label, from *text*.

    Sums the model, KV, compute and output buffers, which is the same total the
    estimate predicts. Empty when the text carries no buffer report: a load that
    failed before allocating, a log that has rotated past it, or an engine whose
    verbosity is below the level that prints it.
    """
    totals: dict[str, int] = {}
    for match in _BUFFER_RE.finditer(text):
        device = match.group("device").removesuffix(_MAPPED_SUFFIX)
        totals[device] = totals.get(device, 0) + int(float(match.group("mib")) * MIB)
    return totals


# The engine says this once the weights are in and it is wiring up slots. Its
# presence means the load finished, which is what separates "the report has not
# been written yet" from "this engine does not write one where we look".
_LOAD_FINISHED_RE = re.compile(r"load_model:\s+initializing slots")
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
ENV_LOG_FILE = "LLAMA_LOG_FILE"
ENV_LOG_VERBOSITY = "LLAMA_LOG_VERBOSITY"
# Level 4 ("trace") is where the per-device buffer report appears. Measured
# against the bundled engine: the default 3 omits it entirely, and 5 adds a
# per-layer and per-slot flood for the same six lines.
LOAD_REPORT_VERBOSITY = "4"


def engine_log_path(log_dir: Path, model_id: str) -> Path:
    """Where the engine serving *model_id* writes its own log."""
    return log_dir / _ENGINE_LOG_TEMPLATE.format(model_id=model_id)


def engine_log_env(log_dir: Path, model_id: str) -> dict[str, str]:
    """Environment that makes the engine report what it allocated, and where."""
    return {
        ENV_LOG_FILE: str(engine_log_path(log_dir, model_id)),
        ENV_LOG_VERBOSITY: LOAD_REPORT_VERBOSITY,
    }


def check_launch(
    log_dir: Path, model_id: str, role: WorkerRole, model: str, estimated_bytes: int
) -> bool:
    """Compare the engine's own report for *model_id* against the estimate.

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
        # No log yet: the engine has not started writing. Nothing to say.
        return False
    actual = device_footprint(text)
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
    return report_divergence(role, model, estimated_bytes, actual, tolerance=_TOLERANCE)


# How far the engine may land from the estimate before it is worth saying. Wide
# enough that the estimator's normal error is quiet, narrow enough to catch the
# whole-slot and whole-cache mistakes this exists to surface.
_TOLERANCE = 0.25
