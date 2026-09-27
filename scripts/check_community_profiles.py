#!/usr/bin/env python3
"""Run ``lilbee profile validate`` over every file in profiles/community/.

Run by ``make lint``. The acceptance bar CONTRIBUTING.md's "Share a profile"
section states is enforced here mechanically: no built-in name, no duplicate
of an existing profile's values, a stated tested_on, and evidence for any
retrieval value that differs from Default. The parts of the bar that need a
human (the measured evidence itself, proof that ingest works on the named
corpus) stay a pull request review's job.

Exits 0 when every file in the folder is a valid community profile
(including when the folder holds none), 1 with one "path: problem" line per
problem when one is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lilbee.core.profile_files import (  # noqa: E402
    COMMUNITY_DIRNAME,
    PACKAGE_PROFILES_DIR,
    PROFILE_SUFFIX,
    ProfileFolder,
    validate_file,
)

COMMUNITY_DIR = PACKAGE_PROFILES_DIR / COMMUNITY_DIRNAME


def check(directory: Path) -> list[str]:
    """One "path: problem" line per problem in every profile file in *directory*."""
    findings = []
    for path in sorted(directory.glob(f"*{PROFILE_SUFFIX}")):
        result = validate_file(path, ProfileFolder.COMMUNITY)
        findings.extend(f"{path}: {problem}" for problem in result.problems)
    return findings


def main() -> int:
    findings = check(COMMUNITY_DIR)
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
