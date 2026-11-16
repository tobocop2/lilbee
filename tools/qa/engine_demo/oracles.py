"""What must be true of an end-to-end engine run, judged from a recorded transcript.

Pure functions over :class:`Transcript`, apart from the code that records one.
Every verdict is checked against fabricated transcripts in
tests/test_engine_demo_oracles.py.

Known limits:
- The served-by log line holds the file name of a local model path, so two models with one
  file name in different directories look the same, and a file name with a space is cut at it.
- An answer passes when one of its words starts with a word stem of a corpus file. A stem
  that a word of the question starts with does not count, unless the question names every
  stem of the file. An answer that uses a stem and says something false about it passes, as
  does a negation ("bees do not waggle").
- A refusal is caught only by its opening phrase (a short list: "I don't know", "I cannot
  find", "Sorry", "the context does not say", with an optional "Unfortunately,"). A refusal
  in other words, or one that starts after a first sentence, passes if it names a stem.
- A citation names a corpus file when its source equals the file name. lilbee reports the
  document name, not a path, so the transcript has no directory to compare; a path fails.
- A log line that only quotes the served-by text counts: the match is not anchored on the
  log-record prefix.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tools.qa.engine_demo.corpus import CORPUS, CORPUS_FACTS

from lilbee.providers.roles import WorkerRole
from lilbee.providers.serving import PRESET_VERSION, REMOTE_VERSION, UNKNOWN_VERSION, log_name

# Bumped whenever a field changes meaning; a stale file must not read as measured.
SCHEMA_VERSION = 1

# Versions a served-by line carries when no managed engine install served the request.
_NOT_AN_INSTALL = frozenset({PRESET_VERSION, REMOTE_VERSION, UNKNOWN_VERSION})

# The line routing_provider logs per request, in this field order:
# "served by engine=%s version=%s build=%s role=%s model=%s".
# The oracle compares the version only; the build names the upstream ref and is not compared.
_SERVED_BY = re.compile(r"served by engine=(\S+) version=(\S+) build=\S+ role=(\S+) model=(\S+)")

# Keys of the CLI JSON, pinned: cli/engine.py writes them from RoleView and BuildView.
ROLE_KEYS = frozenset({"role", "model", "format", "engine", "notice"})
BUILD_KEYS = frozenset({"engine", "version", "backend", "not_run_on"})
ROLES_TOP_KEYS = frozenset({"command", "roles"})
LIST_TOP_KEYS = frozenset({"command", "builds", "not_installed", "engines_folder"})
# Keys every cited source carries: clean_result writes the chunk's source and stamps its label.
CITED_SOURCE_KEYS = frozenset({"source", "label"})
ASK_KEYS = frozenset(
    {
        "command",
        "question",
        "answer",
        "sources",
        "cited_sources",
        "retrieval_query",
        "dropped_sources",
    }
)


class StepName(StrEnum):
    """The steps of one engine run."""

    ENGINE_LIST_BEFORE = "engine-list-before"
    INSTALL = "install"
    PREFER = "prefer"
    PULL = "pull"
    SET_MODEL = "set-model"
    STORED_MODELS = "stored-models"
    ENGINES_LEFT_RUNNING = "engines-left-running"
    USE = "use"
    ADD = "add"
    SYNC = "sync"
    ASK = "ask"
    CHAT = "chat"
    VISION_INGEST = "vision-ingest"
    ENGINE_ROLES = "engine-roles"
    ENGINE_LIST_AFTER = "engine-list-after"


_REQUIRED_STEPS = (StepName.ASK, StepName.ENGINE_ROLES, StepName.ENGINE_LIST_AFTER)
# Only the steps that serve a request count as proof of who served it; vision is served on ingest.
_SERVING_STEPS = (StepName.ASK, StepName.CHAT, StepName.VISION_INGEST)
# The steps whose stdout the TUI shape is read from: they must run with --json.
_JSON_STEPS = (
    StepName.ENGINE_LIST_BEFORE,
    StepName.ENGINE_LIST_AFTER,
    StepName.ENGINE_ROLES,
    StepName.ASK,
)


class Rule(StrEnum):
    """The invariant a failure names."""

    STEP_MISSING = "step-missing"
    STEP_FAILED = "step-failed"
    JSON_UNPARSEABLE = "json-unparseable"
    SERVED_BY_MISSING = "served-by-missing"
    SERVED_BY_PRESET = "served-by-preset"
    SERVED_BY_OTHER = "served-by-other"
    ROLE_UNASSIGNED = "role-unassigned"
    ROLE_WRONG_ENGINE = "role-wrong-engine"
    ASK_FAILED = "ask-failed"
    ASK_UNCITED = "ask-uncited"
    SHAPE_DRIFT = "shape-drift"
    PIN_NOT_AN_INSTALL = "pin-not-an-install"
    NO_ROLES_CLAIMED = "no-roles-claimed"
    STEP_NOT_JSON = "step-not-json"
    SERVED_BY_MODEL = "served-by-model"
    PIN_NOT_INSTALLED = "pin-not-installed"
    ROLE_CONFLICT = "role-conflict"
    CHAT_UNCITED = "chat-uncited"
    ANSWER_OFF_CORPUS = "answer-off-corpus"
    ANSWER_REFUSED = "answer-refused"
    CITATION_OFF_CORPUS = "citation-off-corpus"
    INGEST_FAILED = "ingest-failed"


@dataclass(frozen=True)
class Step:
    """One command of the run and what it printed."""

    name: StepName
    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    json_output: bool = False
    """The command ran with --json, so stdout must parse."""
    log_lines: tuple[str, ...] = ()
    """Lines the command wrote to the lilbee log."""


@dataclass(frozen=True)
class Transcript:
    """What one engine did when lilbee ran it end to end on a host."""

    engine: str
    backend: str
    pin: str
    """The version the install recorded; a served-by line must carry it."""
    claimed_roles: tuple[WorkerRole, ...]
    steps: tuple[Step, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "engine": self.engine,
            "backend": self.backend,
            "pin": self.pin,
            "claimed_roles": [str(r) for r in self.claimed_roles],
            "steps": [
                {
                    "name": str(s.name),
                    "command": list(s.command),
                    "exit_code": s.exit_code,
                    "stdout": s.stdout,
                    "json_output": s.json_output,
                    "log_lines": list(s.log_lines),
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Transcript:
        """Parse a recorded transcript, refusing another schema or an unknown field."""
        schema = payload.get("schema")
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"transcript was written by schema {schema!r}, this harness reads {SCHEMA_VERSION}"
            )
        unknown = set(payload) - {"schema", "engine", "backend", "pin", "claimed_roles", "steps"}
        if unknown:
            raise ValueError(f"transcript carries unknown fields {sorted(unknown)}")
        return cls(
            engine=payload["engine"],
            backend=payload["backend"],
            pin=payload["pin"],
            claimed_roles=tuple(WorkerRole(r) for r in payload["claimed_roles"]),
            steps=tuple(
                Step(
                    name=StepName(s["name"]),
                    command=tuple(s["command"]),
                    exit_code=s["exit_code"],
                    stdout=s["stdout"],
                    json_output=s["json_output"],
                    log_lines=tuple(s["log_lines"]),
                )
                for s in payload["steps"]
            ),
        )


@dataclass(frozen=True)
class Failure:
    """One violated invariant, named by the rule that caught it."""

    rule: Rule
    detail: str


def judge(transcript: Transcript) -> list[Failure]:
    """Every invariant one transcript violates; an empty list is a pass."""
    checks = (
        _required_steps_ran,
        _steps_exited_zero,
        _json_steps_ran_with_json,
        _json_parses,
        _pin_is_an_install,
        _pin_is_installed,
        _every_role_served_by_this_engine,
        _every_role_assigned,
        _every_ask_returned_cited_text,
        _every_chat_returned_cited_text,
        _every_answer_names_a_corpus_fact,
        _every_citation_names_a_corpus_file,
        _no_ingest_failed,
        _json_shape_is_pinned,
    )
    return [failure for check in checks for failure in check(transcript)]


def _steps(t: Transcript, *names: StepName) -> list[Step]:
    return [s for s in t.steps if s.name in names]


_ABSENT: Any = object()

# The curly single quotes (U+2019, U+2018) a model writes in place of an apostrophe.
_STRAIGHT_QUOTES = {0x2019: "'", 0x2018: "'"}

# The opening of an answer that declines to answer, matched on lower-case text with emphasis
# marks and curly quotes already normalised.
_REFUSAL_OPENING = re.compile(
    r"(?:(?:unfortunately|sadly|however|well|hmm)[,.]?\s+)?(?:"
    r"(?:i|we)\s+(?:do not|don't)\s+know\b"
    r"|(?:i|we)\s+(?:cannot|can't|could not|couldn't)\s+(?:find|answer|say|tell|determine)\b"
    r"|(?:(?:i'm|i am)\s+)?sorry\b(?!\s+for)"
    r"|there\s+(?:is|are)\s+no\s+(?:information|mention)\b"
    r"|the\s+(?:(?:provided|given|retrieved)\s+)?(?:context|documents?|sources?|text)\s+"
    r"(?:does not|do not|doesn't|don't)\s+(?:say|mention|contain|provide|include|have)\b"
    r")"
)


def _payload(step: Step) -> Any:
    """The parsed stdout of *step*, or _ABSENT when it does not parse."""
    try:
        return json.loads(step.stdout)
    except ValueError:
        return _ABSENT


def _required_steps_ran(t: Transcript) -> list[Failure]:
    return [
        Failure(Rule.STEP_MISSING, f"no {name} step in the transcript")
        for name in _REQUIRED_STEPS
        if not _steps(t, name)
    ]


def _steps_exited_zero(t: Transcript) -> list[Failure]:
    return [
        Failure(Rule.STEP_FAILED, f"{s.name} exited {s.exit_code}")
        for s in t.steps
        if s.exit_code != 0 and s.name is not StepName.ASK
    ]


def _json_steps_ran_with_json(t: Transcript) -> list[Failure]:
    return [
        Failure(Rule.STEP_NOT_JSON, f"{s.name} did not run with --json")
        for s in _steps(t, *_JSON_STEPS)
        if not s.json_output
    ]


def _json_parses(t: Transcript) -> list[Failure]:
    failures = []
    for s in t.steps:
        if not s.json_output:
            continue
        try:
            json.loads(s.stdout)
        except ValueError as exc:
            failures.append(Failure(Rule.JSON_UNPARSEABLE, f"{s.name} stdout: {exc}"))
    return failures


def _pin_is_an_install(t: Transcript) -> list[Failure]:
    if t.pin and t.pin not in _NOT_AN_INSTALL:
        return []
    return [Failure(Rule.PIN_NOT_AN_INSTALL, f"pin {t.pin!r} names no install of {t.engine}")]


def _pin_is_installed(t: Transcript) -> list[Failure]:
    """Each list after the install shows a build of this engine at the pin on this backend."""
    failures = []
    for step in _steps(t, StepName.ENGINE_LIST_AFTER):
        listing = _payload(step)
        builds = listing.get("builds") if isinstance(listing, dict) else None
        installed = any(
            isinstance(b, dict)
            and (b.get("engine"), b.get("version"), b.get("backend"))
            == (t.engine, t.pin, t.backend)
            for b in builds or []
        )
        if not installed:
            failures.append(
                Failure(
                    Rule.PIN_NOT_INSTALLED,
                    f"engine list shows no {t.engine} build at {t.pin} on {t.backend}",
                )
            )
    return failures


def _role_views(t: Transcript) -> list[list[dict[str, Any]]]:
    """The role views of each engine roles step, in step order."""
    payloads = (_payload(s) for s in _steps(t, StepName.ENGINE_ROLES))
    return [
        [v for v in p["roles"] if isinstance(v, dict)]
        for p in payloads
        if isinstance(p, dict) and isinstance(p.get("roles"), list)
    ]


def _served_lines(t: Transcript) -> list[tuple[str, str, str, str]]:
    """Every (engine, version, role, model) the serving steps log."""
    lines = (line for s in _steps(t, *_SERVING_STEPS) for line in s.log_lines)
    return [(m[1], m[2], m[3], m[4]) for m in map(_SERVED_BY.search, lines) if m]


def _every_role_served_by_this_engine(t: Transcript) -> list[Failure]:
    """Every serving line of a claimed role names this engine, this pin and the assigned model."""
    served = _served_lines(t)
    assigned = [v for views in _role_views(t) for v in views]
    failures = []
    for role in t.claimed_roles:
        lines = [(e, v, m) for e, v, r, m in served if r == role]
        if not lines:
            failures.append(Failure(Rule.SERVED_BY_MISSING, f"no served-by line for {role}"))
        elif any(v == PRESET_VERSION for _, v, _ in lines):
            failures.append(
                Failure(Rule.SERVED_BY_PRESET, f"{role} was served on the preset path: {lines}")
            )
        elif any((e, v) != (t.engine, t.pin) or v in _NOT_AN_INSTALL for e, v, _ in lines):
            failures.append(
                Failure(
                    Rule.SERVED_BY_OTHER, f"{role} was served by {lines}, not {t.engine} {t.pin}"
                )
            )
        else:
            models = {log_name(str(v.get("model"))) for v in assigned if v.get("role") == role}
            if models and any(m not in models for _, _, m in lines):
                failures.append(
                    Failure(Rule.SERVED_BY_MODEL, f"{role} served {lines}, assigned {models}")
                )
    return failures


def _every_role_assigned(t: Transcript) -> list[Failure]:
    failures = []
    if not t.claimed_roles:
        failures.append(Failure(Rule.NO_ROLES_CLAIMED, "the transcript claims no role"))
    all_views = _role_views(t)
    for role in t.claimed_roles:
        views = [[v for v in step if v.get("role") == role] for step in all_views]
        seen = [v for step in views for v in step]
        if not seen:
            failures.append(Failure(Rule.ROLE_UNASSIGNED, f"{role} has no view in engine roles"))
        if (
            any(len(step) > 1 for step in views)
            or len({(v.get("engine"), v.get("model")) for v in seen}) > 1
        ):
            failures.append(
                Failure(
                    Rule.ROLE_CONFLICT, f"engine roles shows {role} more than once, differently"
                )
            )
        for engine in {v.get("engine") for v in seen}:
            if engine is None:
                failures.append(Failure(Rule.ROLE_UNASSIGNED, f"{role} has no engine"))
            elif engine != t.engine:
                failures.append(
                    Failure(Rule.ROLE_WRONG_ENGINE, f"{role} runs on {engine}, not {t.engine}")
                )
    return failures


def _every_ask_returned_cited_text(t: Transcript) -> list[Failure]:
    failures = []
    for step in _steps(t, StepName.ASK):
        if step.exit_code != 0:
            failures.append(Failure(Rule.ASK_FAILED, f"ask exited {step.exit_code}"))
            continue
        payload = _payload(step)
        if not isinstance(payload, dict):
            continue
        if not _has_cited_text(payload):
            failures.append(
                Failure(Rule.ASK_UNCITED, "ask returned no answer text or no cited source")
            )
    return failures


def _every_chat_returned_cited_text(t: Transcript) -> list[Failure]:
    """A chat turn that exited zero answers with text and cites a source, like ask."""
    failures = []
    for step in _steps(t, StepName.CHAT):
        if step.exit_code != 0:
            continue
        payload = _payload(step)
        if not (isinstance(payload, dict) and _has_cited_text(payload)):
            failures.append(
                Failure(Rule.CHAT_UNCITED, "chat returned no answer text or no cited source")
            )
    return failures


def _has_cited_text(payload: Mapping[str, Any]) -> bool:
    answer = payload.get("answer")
    cited = payload.get("cited_sources")
    sources = payload.get("sources")
    return (
        isinstance(answer, str)
        and bool(answer.strip())
        and isinstance(sources, list)
        and bool(sources)
        and isinstance(cited, list)
        and bool(cited)
        and all(isinstance(c, dict) and set(c) >= CITED_SOURCE_KEYS for c in cited)
    )


def _answered(t: Transcript) -> list[tuple[Step, dict[str, Any]]]:
    """Each ask and chat step that exited zero with a JSON object, with that object."""
    pairs = []
    for step in _steps(t, StepName.ASK, StepName.CHAT):
        payload = _payload(step)
        if step.exit_code == 0 and isinstance(payload, dict):
            pairs.append((step, payload))
    return pairs


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def _asked(step: Step, payload: Mapping[str, Any]) -> str:
    """The question *step* sent: ask echoes it in its JSON, chat carries it in the request body."""
    if step.name is StepName.ASK:
        question = payload.get("question")
    else:
        try:
            question = json.loads(step.command[2]).get("question")
        except (IndexError, ValueError, AttributeError):
            question = None
    return question if isinstance(question, str) else ""


def _files_named(answer: str, question: str) -> set[str]:
    """The corpus files whose word stems *answer* uses.

    A stem that a word of the question starts with does not count, unless the question names
    every stem of the file: then the stems count as usual.
    """
    asked = _words(question)
    given = _words(answer)
    named = {}
    for name, stems in CORPUS_FACTS.items():
        unasked = [s for s in stems if not any(w.startswith(s) for w in asked)]
        named[name] = any(w.startswith(s) for s in unasked or stems for w in given)
    return {name for name, uses in named.items() if uses}


def _refuses(answer: str) -> bool:
    """The answer opens with a refusal, so a fact it quotes is not an answer."""
    text = answer.casefold().translate(_STRAIGHT_QUOTES)
    text = re.sub(r"[*_`~#>]", "", text)
    return _REFUSAL_OPENING.match(text.lstrip(" \t\r\n\"'-•")) is not None


def _every_answer_names_a_corpus_fact(t: Transcript) -> list[Failure]:
    """An ask or chat answer must use a stem of the corpus, and must not open with a refusal."""
    failures = []
    for step, payload in _answered(t):
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            continue
        if _refuses(answer):
            failures.append(
                Failure(Rule.ANSWER_REFUSED, f"{step.name} answered {answer[:80]!r}, a refusal")
            )
        elif not _files_named(answer, _asked(step, payload)):
            failures.append(
                Failure(
                    Rule.ANSWER_OFF_CORPUS,
                    f"{step.name} answered {answer[:80]!r}, which names no corpus fact",
                )
            )
    return failures


def _corpus_file(source: object) -> str | None:
    """The corpus file *source* names, or None when it names none; a path names none."""
    return source if isinstance(source, str) and source in CORPUS else None


def _every_citation_names_a_corpus_file(t: Transcript) -> list[Failure]:
    """Each citation is a corpus file, and one cited file is a file the answer draws on."""
    failures = []
    for step, payload in _answered(t):
        cited = payload.get("cited_sources")
        # An entry with no source is a malformed citation, which the cited-text rules report.
        entries = (
            [e for e in cited if isinstance(e, dict) and "source" in e]
            if isinstance(cited, list)
            else []
        )
        names = [_corpus_file(e["source"]) for e in entries]
        for entry, name in zip(entries, names, strict=True):
            if name is None:
                failures.append(
                    Failure(
                        Rule.CITATION_OFF_CORPUS,
                        f"{step.name} cited {entry['source']!r}, no corpus file",
                    )
                )
        answer = payload.get("answer")
        if not isinstance(answer, str) or _refuses(answer):
            continue
        named = _files_named(answer, _asked(step, payload))
        if named and names and not named & {n for n in names if n is not None}:
            failures.append(
                Failure(
                    Rule.CITATION_OFF_CORPUS,
                    f"{step.name} cited {names}, none of which holds the facts of {sorted(named)}",
                )
            )
    return failures


_COUNT_LINE = re.compile(r"^([A-Za-z ]+): (\d+)$", re.MULTILINE)
_INDEX_MISMATCH = re.compile(r"^Index mismatch:", re.MULTILINE)


def _no_ingest_failed(t: Transcript) -> list[Failure]:
    """Add and sync each ingest the corpus: a zero Failed count, nothing held out or mismatched."""
    failures = []
    for step in _steps(t, StepName.ADD, StepName.SYNC):
        counts: dict[str, list[int]] = {}
        for label, number in _COUNT_LINE.findall(step.stdout):
            counts.setdefault(label, []).append(int(number))
        if "Failed" not in counts:
            failures.append(Failure(Rule.INGEST_FAILED, f"{step.name} printed no Failed count"))
        for label in ("Failed", "Held out"):
            if any(counts.get(label, [])):
                failures.append(
                    Failure(
                        Rule.INGEST_FAILED, f"{step.name} printed {label}: {max(counts[label])}"
                    )
                )
        if _INDEX_MISMATCH.search(step.stdout):
            failures.append(Failure(Rule.INGEST_FAILED, f"{step.name} printed an Index mismatch"))
        covered = sum(sum(counts.get(label, [])) for label in ("Added", "Updated", "Unchanged"))
        if covered < len(CORPUS):
            failures.append(
                Failure(
                    Rule.INGEST_FAILED,
                    f"{step.name} covered {covered} of {len(CORPUS)} corpus files",
                )
            )
    return failures


def _keys_differ(where: str, got: object, want: frozenset[str]) -> list[Failure]:
    keys = frozenset(got) if isinstance(got, dict) else frozenset()
    if isinstance(got, dict) and keys == want:
        return []
    return [Failure(Rule.SHAPE_DRIFT, f"{where} keys {sorted(keys)}, pinned {sorted(want)}")]


def _json_shape_is_pinned(t: Transcript) -> list[Failure]:
    """Keys, not values, of the JSON the TUI reads must equal the pinned sets."""
    failures: list[Failure] = []
    for step in _steps(t, StepName.ASK):
        payload = _payload(step)
        if step.exit_code == 0 and payload is not _ABSENT:
            failures += _keys_differ("ask", payload, ASK_KEYS)
    for step in _steps(t, StepName.ENGINE_ROLES):
        roles = _payload(step)
        if roles is not _ABSENT:
            failures += _keys_differ("engine roles", roles, ROLES_TOP_KEYS)
            for view in roles.get("roles", []) if isinstance(roles, dict) else []:
                failures += _keys_differ("engine roles role", view, ROLE_KEYS)
    for step in _steps(t, StepName.ENGINE_LIST_BEFORE, StepName.ENGINE_LIST_AFTER):
        listing = _payload(step)
        if listing is not _ABSENT:
            failures += _keys_differ(str(step.name), listing, LIST_TOP_KEYS)
            for build in listing.get("builds", []) if isinstance(listing, dict) else []:
                failures += _keys_differ(f"{step.name} build", build, BUILD_KEYS)
    return failures
