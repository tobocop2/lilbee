"""Answer collection from a running lilbee server, checkpointed for pod restarts."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.retrieval.checkpoint import JsonlCheckpoint
from evals.retrieval.questions import Question

HEALTH_ROUTE = "/api/health"
ASK_ROUTE = "/api/ask"
MEMORIES_ROUTE = "/api/memories"
HTTP_NOT_FOUND = 404
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


def questions_digest(questions: list[Question]) -> str:
    """Stable digest of the question set an answers file was collected against.

    Regenerating questions with a different seed produces the same qids for
    different questions, so binding the checkpoint to the qids alone would still
    let one file mix answers to two different batteries.
    """
    material = "\x00".join(f"{q.qid}\x1f{q.question}" for q in questions)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def require_memory_disabled(base_url: str, http: httpx.Client) -> None:
    """Refuse to collect answers from a server with memory enabled.

    The ask handler extracts memories from every answered question and the
    searcher reads them back on later turns, so with memory on, question N is
    answered using state produced by questions 1..N-1, and both arms sharing a
    data dir means one arm seeds the other. The run stops being a comparison
    under identical conditions and stops being reproducible, with nothing in the
    results recording which state it ran in.

    The memory routes answer 404 when the subsystem is off, which is the default,
    so a non-404 here is the operator having switched it on.
    """
    response = http.get(f"{base_url.rstrip('/')}{MEMORIES_ROUTE}")
    if response.status_code != HTTP_NOT_FOUND:
        raise RuntimeError(
            f"{base_url} has the memory subsystem enabled. Answers would seed memories "
            "that later questions and the other arm read back, so the arms would not "
            "answer under identical conditions and the run would be order-dependent. "
            "Disable memory on both servers before collecting answers."
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
    checkpoint = JsonlCheckpoint(
        out_path,
        "qid",
        fingerprint={
            "arm": arm,
            "base_url": base_url,
            "top_k": top_k,
            "questions": questions_digest(questions),
        },
    )
    wait_for_server(base_url, http)
    require_memory_disabled(base_url, http)
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
