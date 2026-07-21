"""One message for a missing optional dependency.

Every heavy scorer and loader is imported lazily and raises the same sentence
when it is absent. Spelling that sentence out per module meant six copies of the
requirements path, which is the part that actually moves.
"""

from __future__ import annotations

REQUIREMENTS_PATH = "evals/benchmark/requirements.txt"


def install_hint(package: str, purpose: str) -> str:
    """The error text for ``package`` being absent, naming what it was needed for."""
    return (
        f"{package} is required {purpose}; install the benchmark deps: "
        f"uv pip install -r {REQUIREMENTS_PATH}"
    )
