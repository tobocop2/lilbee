"""Source-level allowlist for outbound HTTP hosts.

Scanners like CodeQL catch tainted input flowing to dangerous sinks but
cannot distinguish a legitimate API call from an exfiltration call to a
static attacker-controlled URL. This test locks down the set of hosts
that can appear anywhere in shipped source. Any new host surfaces here
and must be explicitly approved by extending ALLOWED_HOSTS in the same
commit.
"""

from __future__ import annotations

import re
from pathlib import Path

# Hosts permitted to appear in src/lilbee/ in any form (runtime call,
# docstring reference, help text, error message). Review additions with
# the same care as any outbound network dependency.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        # Runtime endpoints.
        "huggingface.co",  # model catalog + hub downloads
        "localhost",  # default Ollama / litellm backend (user-configurable)
        # Source references: docstrings, ported-from attributions, examples.
        # Present in source but not called at runtime.
        "arxiv.org",
        "docs.python.org",
        "en.wikipedia.org",
        "example.com",
        "github.com",
        # Citations for the dual-vendor Vulkan ICD workaround in gpu_select.py.
        "community.khronos.org",
        "nvidia.custhelp.com",
        "projects.blender.org",
    }
)

_URL_HOST_RE = re.compile(r"https?://([a-zA-Z0-9.-]+)")
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "lilbee"


def test_source_contains_only_allowlisted_hosts() -> None:
    found: set[str] = set()
    for py in _SRC_ROOT.rglob("*.py"):
        for match in _URL_HOST_RE.finditer(py.read_text(encoding="utf-8")):
            found.add(match.group(1).lower())
    unexpected = found - ALLOWED_HOSTS
    assert not unexpected, (
        f"New hosts in src/lilbee/: {sorted(unexpected)}. "
        "Review whether lilbee should be able to reach these endpoints. "
        f"If legitimate, extend ALLOWED_HOSTS in {Path(__file__).name}."
    )
