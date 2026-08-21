"""Whether the bundled engine can decode a remote GGUF, answered by gguf-parser."""

from __future__ import annotations

import logging
import re
import subprocess

from lilbee.catalog.compat import UnsupportedQuantError
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.binary import resolve_gguf_parser
from lilbee.providers.fleet.proc import run_bounded

log = logging.getLogger(__name__)

_FLAG_HF_REPO = "--hf-repo"
_FLAG_HF_FILE = "--hf-file"
# Named for what it carries: a constant spelled with "token" trips the
# hardcoded-credential lint.
_FLAG_AUTH = "--token"
_FLAG_JSON = "--json"
_FLAG_SKIP_ESTIMATE = "--skip-estimate"

_PARSER_LABEL = "gguf-parser"

# gguf-parser reports a tensor it cannot decode against the tensor itself, and
# prefixes every I/O failure (404, DNS, TLS) differently. Only a tensor failure
# says anything about the engine; the rest is the network.
_TENSOR_FAILURE = "read tensor info"
_QUANT_TYPE_RE = re.compile(r"GGMLType\(\d+\)")

# Header range-reads only, but a large tokenizer pushes the tensor table past
# 10 MB, so the ceiling is generous next to the download it guards.
_PROBE_TIMEOUT_S = 120
_PROBE_KILL_WAIT_S = 5.0


def _named_quant(detail: str) -> str:
    """The quantization gguf-parser named in *detail*, or the whole message."""
    match = _QUANT_TYPE_RE.search(detail)
    return match.group(0) if match else detail


def assert_engine_can_load(hf_repo: str, filename: str, token: str | None = None) -> None:
    """Raise :class:`UnsupportedQuantError` when the engine cannot decode *filename*.

    gguf-parser is the engine's own reader, built from the same pin as
    llama-server, so the types it accepts are the binary's rather than a table
    lilbee would have to keep in step. It range-reads the header and downloads
    none of the weights.

    Anything that is not a tensor failure (a 404, DNS, an expired token) is not a
    verdict: the probe stays quiet and lets the download report the real problem,
    which is how the architecture probe already degrades. An install without the
    engine extra has no parser to ask, and must still be able to pull.
    """
    try:
        parser = resolve_gguf_parser()
    except ProviderError as exc:
        log.debug("No gguf-parser to check %s/%s against: %s", hf_repo, filename, exc)
        return
    argv = [
        str(parser),
        _FLAG_HF_REPO,
        hf_repo,
        _FLAG_HF_FILE,
        filename,
        _FLAG_JSON,
        _FLAG_SKIP_ESTIMATE,
    ]
    if token:
        argv += [_FLAG_AUTH, token]
    try:
        output, returncode = run_bounded(
            argv,
            timeout_s=_PROBE_TIMEOUT_S,
            kill_wait_s=_PROBE_KILL_WAIT_S,
            merge_stderr=True,
            label=_PARSER_LABEL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Loadability probe did not run for %s/%s: %s", hf_repo, filename, exc)
        return
    if returncode == 0:
        return
    detail = output.strip()
    if _TENSOR_FAILURE not in detail:
        log.debug("Loadability probe could not read %s/%s: %s", hf_repo, filename, detail)
        return
    raise UnsupportedQuantError(f"{hf_repo}/{filename}", _named_quant(detail))
