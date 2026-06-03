"""Cell result aggregation and the markdown matrix report."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from harness_config import ScenarioStatus
from scenarios import ScenarioResult


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

    @property
    def passed(self) -> bool:
        return (
            not self.setup_error
            and not self.serve_errors
            and self.chat_completions_ok >= len(self.scenarios)
            and bool(self.scenarios)
            and all(s.status is ScenarioStatus.PASS for s in self.scenarios)
        )


def render_report(results: list[CellResult]) -> str:
    lines = [
        "# QA Matrix Results",
        "",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Family | Ref | Status | Chat 200s | Scenarios |",
        "|--------|-----|--------|-----------|-----------|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.setup_error:
            cells = f"setup: {r.setup_error}"
        else:
            cells = ", ".join(f"{s.name.split()[0]}={s.status.value}" for s in r.scenarios)
        lines.append(f"| {r.family} | `{r.ref}` | {status} | {r.chat_completions_ok} | {cells} |")
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
