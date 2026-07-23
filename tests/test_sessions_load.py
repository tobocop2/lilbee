"""Load-bearing invariants for chat sessions under volume, churn and concurrency.

The example-based suites prove each operation does the right thing once. This one
asks whether the feature holds together when a real user hammers it for weeks: a
few hundred conversations, thousands of appends, renames and deletes interleaved,
threads writing while the drawer lists, a process killed mid-write.

Every assertion here is a promise the feature makes:

* nothing written is ever lost or reordered  -- the transcript is the product
* the log is append-only                     -- earlier bytes never change
* a torn tail costs one event, never the file
* concurrent writers never corrupt or lose each other's turns
* the prompt never exceeds its token budget, however long the chat runs

Randomised, but seeded: a failure here reproduces exactly.
"""

from __future__ import annotations

import random
import threading
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.retrieval.query.compaction import (
    compaction_due,
    foldable,
    overflow,
    prompt_history,
    summary_cap,
    summary_messages,
)
from lilbee.retrieval.query.history_window import estimate_tokens
from lilbee.sessions import MessageRole, SessionMessage, SessionStore, TitleSource
from lilbee.sessions.store import SESSIONS_DIRNAME

SEED = 20260715


@pytest.fixture
def store(tmp_path, monkeypatch) -> SessionStore:
    cfg.data_dir = tmp_path / "data"
    # Neutralize os.fsync for this suite only. It writes thousands of events in a
    # tight loop, and on Windows os.fsync is a real FlushFileBuffers barrier
    # (~4ms each vs ~0.05ms on macOS), so the per-event disk barrier -- not the
    # code under test -- is what wedged these tests in Windows CI. None of the
    # invariants below depend on fsync: they hold on O_APPEND + one write per
    # event + flush + the FileLock. Production is untouched (real usage appends
    # one event per user turn, seconds apart, where the barrier is cheap and
    # buys genuine crash durability); monkeypatch reverts after each test.
    monkeypatch.setattr("lilbee.sessions.store.os.fsync", lambda _fd: None)
    return SessionStore()


def _msg(content: str, role: MessageRole = MessageRole.USER) -> SessionMessage:
    return SessionMessage(role=role, content=content)


def _path(tmp_path: Path, session_id: str) -> Path:
    return tmp_path / "data" / SESSIONS_DIRNAME / f"{session_id}.jsonl"


def test_no_turn_is_ever_lost_across_a_long_randomised_session_life(store, tmp_path) -> None:
    """Thousands of interleaved appends across many sessions all read back intact.

    The transcript IS the feature: a chat that loses a turn is worse than one that
    never saved it, because the user believes it is there.
    """
    rng = random.Random(SEED)
    expected: dict[str, list[str]] = {}
    ids = [store.create(model_ref="m", scope="both") for _ in range(40)]
    for sid in ids:
        expected[sid] = []

    for turn in range(2000):
        sid = rng.choice(ids)
        content = f"{turn}:" + "x" * rng.randint(1, 400)
        role = MessageRole.USER if turn % 2 == 0 else MessageRole.ASSISTANT
        store.add_message(sid, _msg(content, role))
        expected[sid].append(content)
        # churn the other event types alongside, exactly as the drawer/CLI do
        if rng.random() < 0.05:
            store.set_title(sid, f"renamed at {turn}", TitleSource.CUSTOM)
        if rng.random() < 0.05:
            store.set_summary(sid, f"notes at {turn}")

    for sid, contents in expected.items():
        session = store.get(sid)
        assert [m.content for m in session.messages] == contents, "a turn was lost or reordered"
        assert session.meta.message_count == len(contents)


def test_the_log_is_append_only_under_churn(store, tmp_path) -> None:
    """Every operation may only append: earlier bytes must never change.

    This is what makes the store safe to read while it is written, and what makes
    a rename cost one line instead of rewriting a conversation.
    """
    rng = random.Random(SEED)
    sid = store.create(model_ref="m", scope="both")
    path = _path(tmp_path, sid)
    seen = path.read_bytes()

    for turn in range(200):
        roll = rng.random()
        if roll < 0.6:
            store.add_message(sid, _msg(f"turn {turn}"))
        elif roll < 0.8:
            store.set_title(sid, f"title {turn}", TitleSource.CUSTOM)
        else:
            store.set_summary(sid, f"summary {turn}")
        now = path.read_bytes()
        assert now.startswith(seen), f"operation {turn} rewrote history instead of appending"
        assert len(now) > len(seen), "an operation wrote nothing"
        seen = now


def test_a_torn_final_line_costs_only_that_event(store, tmp_path) -> None:
    """A process killed mid-append must lose the half-written line and nothing else.

    The only corruption an append log can suffer, and the reason the reader skips
    a bad final line instead of refusing the file.
    """
    sid = store.create(model_ref="m", scope="both")
    for i in range(50):
        store.add_message(sid, _msg(f"turn {i}"))
    path = _path(tmp_path, sid)
    good = path.read_bytes()

    # kill the process mid-line: keep a partial JSON object at the tail
    path.write_bytes(good + b'{"type": "message", "role": "user", "cont')
    session = store.get(sid)
    assert [m.content for m in session.messages] == [f"turn {i}" for i in range(50)]
    assert session.meta.message_count == 50


def test_concurrent_writers_never_lose_or_corrupt_turns(store, tmp_path) -> None:
    """Chat writes on a worker thread while the drawer lists on the UI thread.

    Both hit the same store instance, so its meta cache and file handles are
    shared. Any turn lost here is one a user watched appear on screen.
    """
    ids = [store.create(model_ref="m", scope="both") for _ in range(8)]
    per_session = 60
    errors: list[BaseException] = []

    def write(sid: str) -> None:
        try:
            for i in range(per_session):
                store.add_message(sid, _msg(f"{sid[:4]}-{i}"))
        except BaseException as exc:  # surfaced via the assert below, not swallowed
            errors.append(exc)

    def keep_listing() -> None:
        try:
            for _ in range(200):
                store.list()
        except BaseException as exc:  # surfaced via the assert below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(sid,)) for sid in ids]
    threads.append(threading.Thread(target=keep_listing))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent access raised: {errors[:2]}"
    for sid in ids:
        session = store.get(sid)
        assert [m.content for m in session.messages] == [
            f"{sid[:4]}-{i}" for i in range(per_session)
        ], "a concurrent writer lost or interleaved a turn"


def test_listing_stays_correct_at_volume(store) -> None:
    """A heavy user's vault still lists every session, with accurate counts."""
    ids = []
    for n in range(150):
        sid = store.create(model_ref="m", scope="both")
        store.set_title(sid, f"conversation {n}", TitleSource.AUTO)
        for i in range(n % 7):
            store.add_message(sid, _msg(f"m{i}"))
        ids.append(sid)

    metas = store.list()
    assert len(metas) == 150
    by_id = {meta.id: meta for meta in metas}
    assert set(by_id) == set(ids), "a session vanished from the listing"
    for n, sid in enumerate(ids):
        assert by_id[sid].message_count == n % 7
        assert by_id[sid].title == f"conversation {n}"
    # newest-first ordering holds across the whole vault
    stamps = [meta.updated_at for meta in metas]
    assert stamps == sorted(stamps, reverse=True)


def test_the_prompt_never_exceeds_its_budget_over_a_long_conversation() -> None:
    """However long the chat runs, what goes to the model stays inside the budget.

    Exceeding it is not a degraded answer, it is a hard engine failure. The one
    documented exception is a single turn larger than the whole budget, which
    windowed_history keeps deliberately rather than send an empty prompt.
    """
    rng = random.Random(SEED)
    for ctx in (512, 2048, 8192, 32768):
        budget = int(ctx * 0.5)
        history: list[dict[str, str]] = []
        summary = ""
        for turn in range(300):
            role = "user" if turn % 2 == 0 else "assistant"
            history.append({"role": role, "content": "w " * rng.randint(5, 300)})
            # what chat does each turn: fold the overflow, then build the prompt
            reserved = sum(estimate_tokens(m) for m in summary_messages(summary))
            dropped = overflow(history, max_tokens=max(1, budget - reserved))
            if dropped:
                del history[: len(dropped)]
                summary = "notes " * rng.randint(1, 40)
            prompt = prompt_history(history, summary, max_tokens=budget)
            cost = sum(estimate_tokens(m) for m in prompt)
            # The one documented escape: windowed_history keeps the newest PAIR
            # even when it alone busts the budget, because an empty prompt is
            # useless. Carrying a summary must never widen that overrun.
            newest_pair = sum(estimate_tokens(m) for m in history[-2:])
            assert cost <= budget or newest_pair > budget, (
                f"ctx={ctx} turn={turn}: prompt of {cost} tokens exceeds the {budget} "
                f"budget while the newest pair ({newest_pair}) would have fit"
            )
            assert cost <= max(budget, newest_pair), (
                f"ctx={ctx} turn={turn}: the summary widened an overrun to {cost} tokens"
            )


def test_a_long_chat_does_not_compact_on_every_turn() -> None:
    """Compaction must buy headroom, not re-fire immediately.

    Folding only the overflow at the limit leaves the history full, so the next
    turn overflows again and every later turn pays a model call -- measured at 2
    per turn, forever. Firing at a threshold and clearing to notes plus the newest
    exchanges is what makes it affordable. A regression here is invisible in any
    single-turn test: it only shows up as the tenth turn being slow.
    """
    budget = 4096  # an 8192 context, the default floor
    history: list[dict[str, str]] = []
    summary = ""
    calls = 0
    turns = 40
    for _ in range(turns):
        history.append({"role": "user", "content": "q " * 50})
        history.append({"role": "assistant", "content": "a " * 300})
        if not compaction_due(history, summary, max_tokens=budget):
            continue
        dropped = foldable(history)
        if not dropped:
            continue
        calls += 1
        del history[: len(dropped)]
        summary = "s" * (summary_cap(8192) * 4)  # steady state: notes at the cap

    assert calls <= turns // 8, (
        f"{calls} compactions over {turns} turns: the trigger is re-firing rather "
        "than buying headroom"
    )
    assert calls > 0, "a 40-turn chat at this budget must compact at least once"


def test_the_default_path_never_calls_the_model_however_long_the_chat() -> None:
    """Compaction off has to stay free at any length, not merely at first."""
    budget = 1024
    history: list[dict[str, str]] = []
    for _ in range(200):
        history.append({"role": "user", "content": "q " * 50})
        history.append({"role": "assistant", "content": "a " * 300})
        dropped = overflow(history, max_tokens=budget)
        if dropped:
            del history[: len(dropped)]
        # the whole default path: a window and a slice, no model in sight
        assert sum(estimate_tokens(m) for m in history) <= budget or len(history) <= 2


def test_a_resumed_conversation_round_trips_exactly(store) -> None:
    """What the chat wrote is what a resume reads back, in order, with sources."""
    rng = random.Random(SEED)
    sid = store.create(model_ref="some/model.gguf", scope="both")
    written: list[tuple[str, str, tuple[str, ...]]] = []
    for i in range(300):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        content = f"{i}:" + "y" * rng.randint(1, 200)
        sources = tuple(f"doc{j}.pdf" for j in range(rng.randint(0, 3)))
        store.add_message(sid, SessionMessage(role=role, content=content, sources=sources))
        written.append((role.value, content, sources))

    session = store.get(sid)
    assert [(m.role.value, m.content, m.sources) for m in session.messages] == written
    assert session.meta.model_ref == "some/model.gguf"
