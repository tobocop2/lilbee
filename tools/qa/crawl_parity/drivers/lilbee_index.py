"""Index a directory of markdown with lilbee's own ingest and answer questions with its search."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--data", required=True, type=Path, help="lilbee data directory to index")
PARSER.add_argument("--questions", required=True, type=Path, help="JSON lines of id and text")
PARSER.add_argument("--embedding-model", required=True)
PARSER.add_argument("--top-k", required=True, type=int)
PARSER.add_argument("--out", required=True, type=Path, help="JSON result file")
ARGS = PARSER.parse_args()
os.environ.update(
    {
        "LILBEE_DATA": str(ARGS.data),
        "LILBEE_SKIP_TOML_CONFIG": "1",
        "LILBEE_NO_SPLASH": "1",
        "LILBEE_EMBEDDING_MODEL": ARGS.embedding_model,
        "LILBEE_QUERY_EXPANSION_COUNT": "0",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
)

# lilbee reads its settings from the environment at import, so these imports follow it.
from lilbee.app.services import get_services, reset_services  # noqa: E402
from lilbee.data.ingest import sync  # noqa: E402


def run() -> int:
    """Sync, search every question, and write the sources each search returned."""
    started = time.time()
    result = asyncio.run(sync(quiet=True))
    indexed = time.time()
    answers: dict[str, list[str]] = {}
    searcher = get_services().searcher
    for line in ARGS.questions.read_text(encoding="utf-8").splitlines():
        question = json.loads(line)
        hits = searcher.search(question["text"], top_k=ARGS.top_k)
        answers[question["id"]] = [hit.source for hit in hits]
    record = {
        "added": len(result.added),
        "failed": list(result.failed),
        "skipped": list(result.skipped),
        "index_seconds": indexed - started,
        "search_seconds": time.time() - indexed,
        "answers": answers,
    }
    ARGS.out.write_text(json.dumps(record), encoding="utf-8")
    reset_services()
    return 0


if __name__ == "__main__":
    sys.exit(run())
