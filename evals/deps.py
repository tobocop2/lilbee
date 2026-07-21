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


# Every package whose version can move a published number. ragas is the sharp
# case: it is a fast-moving product whose metric prompts change between
# releases, so "which ragas scored this" is a question a reader is entitled to
# ask and the requirements pin alone cannot answer for a run already finished.
SCORER_PACKAGES = (
    "ir_measures",
    "ir_datasets",
    "pytrec_eval_terrier",
    "python-terrier",
    "scipy",
    "scikit-learn",
    "ragas",
)


def scorer_versions() -> dict[str, str]:
    """The installed version of every scorer, for the run record.

    Read from installed metadata rather than the requirements file, because the
    requirements file states an intention and this states what actually ran. A
    package that is absent is recorded as absent rather than omitted, so a run
    that skipped a tier says so instead of leaving a gap the reader must guess at.
    """
    from importlib.metadata import PackageNotFoundError, version

    installed: dict[str, str] = {}
    for package in SCORER_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "not installed"
    return installed
