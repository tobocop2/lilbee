"""Read a pytest log and fail unless the run executed at least a floor of tests.

Usage: fd_tests_ran.py LOG STEP FLOOR

A trace of a run that never left collection reads zero closed descriptors. This control
turns that silent zero into a failure. Output lines start with ``FDC-TESTS``.
"""

import re
import sys

SUMMARY = re.compile(r"(?P<passed>\d+) passed")


def main() -> int:
    log, step, floor = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(log, encoding="utf-8", errors="replace") as handle:
        counts = [int(m["passed"]) for line in handle if (m := SUMMARY.search(line))]
    passed = counts[-1] if counts else 0
    verdict = "ran" if passed >= floor else "DID-NOT-RUN"
    print(f"FDC-TESTS step={step} passed={passed} floor={floor} {verdict}")
    return 0 if passed >= floor else 1


if __name__ == "__main__":
    sys.exit(main())
