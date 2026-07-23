"""Sync the marketing site's coverage numbers with the generated htmlcov report."""

import re
import sys
from pathlib import Path

HTMLCOV_INDEX = Path("htmlcov/index.html")
SITE_INDEX = Path("site/index.html")

PC_COV_RE = re.compile(r'<span class="pc_cov">(\d+(?:\.\d+)?%)</span>')
PILL_ANCHOR = '<span class="pill">100% coverage</span>'
DESC_ANCHOR = '<span class="desc">100%</span>'


def report_percent() -> str:
    match = PC_COV_RE.search(HTMLCOV_INDEX.read_text(encoding="utf-8"))
    if not match:
        sys.exit(f"error: no coverage percent in {HTMLCOV_INDEX}; the htmlcov report did not build")
    return match.group(1)


def main() -> None:
    percent = report_percent()
    html = SITE_INDEX.read_text(encoding="utf-8")
    for anchor, replacement in [
        (PILL_ANCHOR, f'<span class="pill">{percent} coverage</span>'),
        (DESC_ANCHOR, f'<span class="desc">{percent}</span>'),
    ]:
        if html.count(anchor) != 1:
            sys.exit(f"error: expected one {anchor!r} in {SITE_INDEX}; update the anchors to match the markup")
        html = html.replace(anchor, replacement)
    SITE_INDEX.write_text(html, encoding="utf-8")
    print(f"site coverage numbers set to {percent}")


if __name__ == "__main__":
    main()
