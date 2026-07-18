"""Answer collection from a running lilbee server, checkpointed for pod restarts."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.retrieval.checkpoint import JsonlCheckpoint
from evals.retrieval.questions import Question

HEALTH_ROUTE = "/api/health"
ASK_ROUTE = "/api/ask"
ASK_TIMEOUT_SECONDS = 600.0
ANSWER_ATTEMPTS = 3
ANSWER_RETRY_DELAY_SECONDS = 5.0
WARMUP_ATTEMPTS = 180
WARMUP_POLL_SECONDS = 5.0


@dataclass
class AnswerRow:
    """One arm's answer to one question, or its hard failure."""

    qid: str
    arm: str
    answer: str
    sources: list[str]
    cited_sources: list[str]
    seconds: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerRow:
        return cls(
            qid=data["qid"],
            arm=data["arm"],
            answer=data["answer"],
            sources=list(data["sources"]),
            cited_sources=list(data["cited_sources"]),
            seconds=data["seconds"],
            error=data["error"],
        )


def make_http_client() -> httpx.Client:
    return httpx.Client(timeout=ASK_TIMEOUT_SECONDS)


def wait_for_server(
    base_url: str,
    client: httpx.Client,
    *,
    attempts: int = WARMUP_ATTEMPTS,
    poll: float = WARMUP_POLL_SECONDS,
) -> None:
    """Block until the server's health route answers 200."""
    for _ in range(attempts):
        try:
            if client.get(f"{base_url.rstrip('/')}{HEALTH_ROUTE}").status_code == httpx.codes.OK:
                return
        except httpx.HTTPError:
            pass
        time.sleep(poll)
    raise RuntimeError(f"server at {base_url} never became healthy")


def _ask(client: httpx.Client, base_url: str, question: str, top_k: int) -> AnswerRow:
    response = client.post(
        f"{base_url.rstrip('/')}{ASK_ROUTE}", json={"question": question, "top_k": top_k}
    )
    response.raise_for_status()
    body = response.json()
    return AnswerRow(
        qid="",
        arm="",
        answer=body["answer"],
        sources=[chunk["source"] for chunk in body["sources"]],
        cited_sources=[chunk["source"] for chunk in body.get("cited_sources", [])],
        seconds=0.0,
        error=None,
    )


def answer_questions(
    questions: list[Question],
    base_url: str,
    arm: str,
    out_path: Path,
    *,
    top_k: int = 0,
    attempts: int = ANSWER_ATTEMPTS,
    retry_delay: float = ANSWER_RETRY_DELAY_SECONDS,
    client: httpx.Client | None = None,
) -> list[AnswerRow]:
    """Answer every question not already checkpointed; return this run's rows."""
    http = client or make_http_client()
    checkpoint = JsonlCheckpoint(out_path, "qid")
    wait_for_server(base_url, http)
    rows: list[AnswerRow] = []
    for question in questions:
        if question.qid in checkpoint:
            continue
        started = time.monotonic()
        row: AnswerRow | None = None
        error = ""
        for attempt in range(attempts):
            try:
                row = _ask(http, base_url, question.question, top_k)
                break
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < attempts:
                    time.sleep(retry_delay)
        if row is None:
            # A question that fails all attempts is itself a result.
            row = AnswerRow(
                qid="", arm="", answer="", sources=[], cited_sources=[], seconds=0.0, error=error
            )
        row.qid = question.qid
        row.arm = arm
        row.seconds = round(time.monotonic() - started, 2)
        checkpoint.append(row.to_dict())
        rows.append(row)
        status = "ERROR" if row.error else "ok"
        print(f"[{arm}] {question.qid} {status}", flush=True)
    return rows
