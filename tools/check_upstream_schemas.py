#!/usr/bin/env python3
"""Watch upstream HF model repos for response_schema population.

For each tool-call response schema lilbee ships locally, fetch the
representative model's ``tokenizer_config.json`` from HuggingFace Hub and
report which can be retired (upstream now populates ``response_schema``),
which are still pending, and which could not be checked (gated repo,
network error). Emits a markdown report on stdout.

The script holds no runtime hook: lilbee continues to use its local
schemas. The report's role is to surface candidates for one-by-one
manual retirement once upstream catches up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "src" / "lilbee" / "providers" / "worker" / "response_parser" / "schemas"
UPSTREAM_REPOS_FILE = SCHEMAS_DIR / "_upstream_repos.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
RESPONSE_SCHEMA_KEY = "response_schema"

log = logging.getLogger(__name__)


class RetirementStatus(StrEnum):
    """Verdict for one (family, upstream repo) check."""

    READY = "ready"
    PENDING = "pending"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FamilyCheck:
    """One family's verdict plus any human-readable detail."""

    family: str
    repo_id: str
    status: RetirementStatus
    detail: str = ""


def load_upstream_repos(path: Path) -> dict[str, str]:
    """Read the family -> repo_id mapping from disk."""
    data = json.loads(path.read_text("utf-8"))
    return {family: entry["repo"] for family, entry in data.items()}


def list_local_schemas(schemas_dir: Path) -> set[str]:
    """Family names with a JSON schema shipped (filenames, without ``.json``)."""
    return {p.stem for p in schemas_dir.glob("*.json") if not p.name.startswith("_")}


def check_drift(schemas_dir: Path, repo_map: dict[str, str]) -> tuple[set[str], set[str]]:
    """Return ``(schemas_without_repo, repos_without_schema)``."""
    local = list_local_schemas(schemas_dir)
    tracked = set(repo_map)
    return (local - tracked, tracked - local)


def check_family(family: str, repo_id: str) -> FamilyCheck:
    """Fetch one upstream ``tokenizer_config.json`` and classify retirement status."""
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename=TOKENIZER_CONFIG_FILE)
    except GatedRepoError:
        return FamilyCheck(family, repo_id, RetirementStatus.BLOCKED, "gated repo")
    except RepositoryNotFoundError:
        return FamilyCheck(family, repo_id, RetirementStatus.BLOCKED, "repo not found")
    except OSError as exc:
        return FamilyCheck(family, repo_id, RetirementStatus.BLOCKED, f"network/IO: {exc}")
    try:
        config = json.loads(Path(config_path).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        return FamilyCheck(family, repo_id, RetirementStatus.BLOCKED, f"parse: {exc}")
    if config.get(RESPONSE_SCHEMA_KEY):
        return FamilyCheck(family, repo_id, RetirementStatus.READY)
    return FamilyCheck(family, repo_id, RetirementStatus.PENDING)


def render_report(checks: list[FamilyCheck]) -> str:
    """Render a markdown retirement report grouped by status."""
    if not checks:
        return "# Response-schema retirement check\n\n_No families to check._\n"
    by_status: dict[RetirementStatus, list[FamilyCheck]] = {
        RetirementStatus.READY: [],
        RetirementStatus.PENDING: [],
        RetirementStatus.BLOCKED: [],
    }
    for check in checks:
        by_status[check.status].append(check)
    sections = [
        _render_section("Ready to retire", by_status[RetirementStatus.READY], _render_ready_block),
        _render_section("Pending upstream", by_status[RetirementStatus.PENDING], _render_pending),
        _render_section("Could not check", by_status[RetirementStatus.BLOCKED], _render_blocked),
    ]
    body = "\n".join(section for section in sections if section)
    return f"# Response-schema retirement check\n\n{body}".rstrip() + "\n"


def _render_section(title: str, checks: list[FamilyCheck], format_item: Callable) -> str:
    """Render one status section, or empty string when nothing falls in it."""
    if not checks:
        return ""
    items = "".join(format_item(check) for check in checks)
    return f"## {title}\n\n{items}"


def _render_pending(check: FamilyCheck) -> str:
    """One bullet for a family that's still pending upstream populating."""
    return f"- `{check.family}`: {check.repo_id} has no response_schema yet\n"


def _render_blocked(check: FamilyCheck) -> str:
    """One bullet for a family the watcher could not check."""
    return f"- `{check.family}`: {check.repo_id}: {check.detail}\n"


def _render_ready_block(check: FamilyCheck) -> str:
    """Render the retirement checklist for one ready family."""
    return (
        f"### `{check.family}` migrated upstream\n"
        f"\n"
        f"`{check.repo_id}` populates `tokenizer_config.json#response_schema`. "
        f"To retire the local copy:\n"
        f"\n"
        f"- [ ] Remove `src/lilbee/providers/worker/response_parser/schemas/{check.family}.json`\n"
        f"- [ ] Remove `ModelFamily.{check.family.upper()}` from `families.py` "
        f"(enum, detection-marker constants, branch in `detect_family`)\n"
        f"- [ ] Remove `ModelFamily.{check.family.upper()}` from `_SCHEMA_FILES` "
        f"in `schemas.py`\n"
        f"- [ ] Remove the `{check.family}` round-trip test in "
        f"`tests/providers/worker/response_parser/test_parse.py`\n"
        f"- [ ] Remove the `{check.family}` detection test in "
        f"`tests/providers/worker/response_parser/test_families.py`\n"
        f"- [ ] Remove the `{check.family}` row from `docs/agent-integration.md`\n"
        f"- [ ] Remove the `{check.family}` entry from "
        f"`response_parser/schemas/_upstream_repos.json`\n\n"
    )


def run(*, schemas_dir: Path, upstream_repos_file: Path) -> str:
    """Run the watcher end-to-end and return the markdown report."""
    repo_map = load_upstream_repos(upstream_repos_file)
    missing_repos, missing_schemas = check_drift(schemas_dir, repo_map)
    if missing_repos:
        raise ValueError(
            f"Schemas without an upstream repo entry: {sorted(missing_repos)}. "
            f"Add them to {upstream_repos_file}."
        )
    if missing_schemas:
        raise ValueError(
            f"Upstream repo entries with no matching local schema: "
            f"{sorted(missing_schemas)}. Remove them from {upstream_repos_file}."
        )
    checks = [check_family(family, repo_id) for family, repo_id in sorted(repo_map.items())]
    return render_report(checks)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=SCHEMAS_DIR,
        help="Override the schemas directory (default: repo schemas/).",
    )
    parser.add_argument(
        "--upstream-repos-file",
        type=Path,
        default=UPSTREAM_REPOS_FILE,
        help="Override the upstream-repos JSON (default: schemas/_upstream_repos.json).",
    )
    args = parser.parse_args(argv)
    report = run(schemas_dir=args.schemas_dir, upstream_repos_file=args.upstream_repos_file)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
