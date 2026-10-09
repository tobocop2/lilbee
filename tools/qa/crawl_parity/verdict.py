"""The verdict: every difference set against expected.toml, then PASS or FAIL."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Mode, Status

EXPECTED_FILE = Path(__file__).parent / "expected.toml"
SEGMENTS = 4
FAILING = frozenset({Status.NEW, Status.KNOWN})
ANY_PAGE = "*"


@dataclass(frozen=True)
class Expected:
    """One known difference: globs over signatures and pages, its issue, and the owner's word."""

    signature: str
    issue: str
    note: str = ""
    accepted: bool = False
    page: str = ANY_PAGE

    def matches(self, difference: Difference) -> bool:
        """Whether this entry covers *difference*."""
        return fnmatchcase(difference.signature, self.signature) and fnmatchcase(
            difference.page, self.page
        )

    def applies_to(self, ran: set[tuple[Kind, Mode]]) -> bool:
        """Whether a run that measured *ran* could have shown this entry's difference."""
        kind, mode, *_rest = self.signature.split("/", SEGMENTS - 1)
        return any(fnmatchcase(k.value, kind) and fnmatchcase(m.value, mode) for k, m in ran)


@dataclass(frozen=True)
class Finding:
    """Every difference with one signature, and where the signature stands."""

    signature: str
    status: Status
    differences: tuple[Difference, ...]
    issue: str = ""

    @property
    def pages(self) -> int:
        """On how many pages the signature shows."""
        return len({difference.page for difference in self.differences})


@dataclass(frozen=True)
class Verdict:
    """The findings of a run and the entries of expected.toml that no difference matched."""

    findings: tuple[Finding, ...]
    fixed: tuple[Expected, ...]
    not_run: tuple[Expected, ...]
    not_measured: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        """True when nothing fails and every requested yardstick gave a measurement."""
        return not self.not_measured and not any(f.status in FAILING for f in self.findings)

    def with_status(self, status: Status) -> list[Finding]:
        """The findings that stand at *status*."""
        return [finding for finding in self.findings if finding.status is status]


def load_expected(path: Path = EXPECTED_FILE) -> list[Expected]:
    """The entries of an expected.toml."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    return [
        Expected(
            signature=str(entry["signature"]),
            issue=str(entry["issue"]),
            note=str(entry.get("note", "")),
            accepted=bool(entry.get("accepted", False)),
            page=str(entry.get("page", ANY_PAGE)),
        )
        for entry in document.get("known", [])
    ]


def _status(entries: list[Expected | None]) -> Status:
    """The status of a signature from the entry of each of its differences that counts."""
    if not entries:
        return Status.NOT_COUNTED
    known = [entry for entry in entries if entry is not None]
    if len(known) < len(entries):
        return Status.NEW
    return Status.ACCEPTED if all(entry.accepted for entry in known) else Status.KNOWN


def judge(
    differences: list[Difference],
    expected: list[Expected],
    ran: set[tuple[Kind, Mode]],
    not_measured: tuple[str, ...] = (),
) -> Verdict:
    """Group differences by signature and set each difference against the expected entries.

    A signature is NEW when one of its differences that counts has no entry.
    """
    groups: dict[str, list[Difference]] = {}
    for difference in differences:
        groups.setdefault(difference.signature, []).append(difference)
    findings: list[Finding] = []
    matched: set[Expected] = set()
    for signature, group in sorted(groups.items()):
        entries = [
            next((entry for entry in expected if entry.matches(difference)), None)
            for difference in group
        ]
        used = [entry for entry in entries if entry is not None]
        matched.update(used)
        counting = [
            entry for difference, entry in zip(group, entries, strict=True) if difference.counts
        ]
        issues = ", ".join(dict.fromkeys(entry.issue for entry in used))
        findings.append(Finding(signature, _status(counting), tuple(group), issues))
    unmatched = [entry for entry in expected if entry not in matched]
    return Verdict(
        findings=tuple(findings),
        fixed=tuple(entry for entry in unmatched if entry.applies_to(ran)),
        not_run=tuple(entry for entry in unmatched if not entry.applies_to(ran)),
        not_measured=not_measured,
    )
