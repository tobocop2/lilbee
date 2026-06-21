"""Cell result aggregation and the markdown matrix report."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from harness_config import ExpectedSupport, ScenarioStatus
from scenarios import ScenarioResult

# Per-cell classification, comparing the verdict to the documented expectation.
_CLASS_PASS = "PASS"
_CLASS_REGRESSION = "REGRESSION"
_CLASS_EXPECTED_FAIL = "expected-fail"
_CLASS_NEWLY_WORKING = "newly-working"


@dataclass
class CellResult:
    family: str
    ref: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
    setup_error: str = ""
    serve_errors: str = ""
    """Worker / dispatch errors scraped from the cell's launcher-serve.log.
    Non-empty means the chat or embed worker raised an exception during the
    cell, so even an all-green substring check downgrades to FAIL.
    """
    chat_completions_ok: int = 0
    """Count of ``POST /v1/chat/completions ... 200`` lines in launcher-serve.log.
    Zero means opencode never got a successful chat back (model failed to load,
    or every turn 500'd), so the cell cannot be a real PASS regardless of pane.
    """
    expected: ExpectedSupport = ExpectedSupport.SUPPORTED

    @property
    def passed(self) -> bool:
        return (
            not self.setup_error
            and not self.serve_errors
            and self.chat_completions_ok >= len(self.scenarios)
            and bool(self.scenarios)
            and all(s.status is ScenarioStatus.PASS for s in self.scenarios)
        )

    @property
    def classification(self) -> str:
        if self.passed:
            return (
                _CLASS_PASS if self.expected is ExpectedSupport.SUPPORTED else _CLASS_NEWLY_WORKING
            )
        return (
            _CLASS_REGRESSION
            if self.expected is ExpectedSupport.SUPPORTED
            else _CLASS_EXPECTED_FAIL
        )

    @property
    def is_regression(self) -> bool:
        """A supported family that failed: the only outcome that fails the run."""
        return self.classification == _CLASS_REGRESSION


def render_report(results: list[CellResult]) -> str:
    regressions = [r for r in results if r.is_regression]
    lines = [
        "# QA Matrix Results",
        "",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Regressions (supported family failed): {len(regressions)}"
        + (f" -> {', '.join(r.family for r in regressions)}" if regressions else ""),
        "",
        "| Family | Ref | Result | Expected | Chat 200s | Scenarios |",
        "|--------|-----|--------|----------|-----------|-----------|",
    ]
    for r in results:
        if r.setup_error:
            cells = f"setup: {r.setup_error}"
        else:
            cells = ", ".join(f"{s.name.split()[0]}={s.status.value}" for s in r.scenarios)
        lines.append(
            f"| {r.family} | `{r.ref}` | {r.classification} | {r.expected.value} "
            f"| {r.chat_completions_ok} | {cells} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r.family} ({r.ref})")
        lines.append("")
        if r.setup_error:
            lines.append(f"Setup error: {r.setup_error}")
        if r.serve_errors:
            lines.append("### Worker / dispatch errors in launcher-serve.log")
            lines.append("```")
            lines.append(r.serve_errors)
            lines.append("```")
        for s in r.scenarios:
            lines.append(f"### {s.name} -> {s.status.value}")
            lines.append(f"Detail: {s.detail}")
            lines.append("```")
            lines.append(s.pane_excerpt or "(no pane captured)")
            lines.append("```")
        lines.append("")
    return "\n".join(lines)
