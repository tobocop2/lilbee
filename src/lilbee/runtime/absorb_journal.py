"""The journal of an absorb: a parent source taking the registered sources below it."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)

PENDING_ABSORB_FILENAME = "pending_absorb.json"
_UNREADABLE = (
    "Cannot read {path} ({error}). An add was interrupted while it moved a source into "
    "its parent folder, and lilbee cannot finish it without that file. To index the "
    "library again, delete that file and run `lilbee rebuild`, which also indexes the "
    "files you removed. Then add the folder again."
)


class AbsorbPhase(StrEnum):
    """How far the re-key of an absorb has gone."""

    LIFT = "lift"
    LAND = "land"


class AbsorbJournalError(RuntimeError):
    """Raised when the journal of an interrupted absorb cannot be read."""


@dataclass(frozen=True)
class HeldOut:
    """What a skip record says of one file: the hash it is held out at, and why."""

    hash: str
    reason: str | None
    removed: bool
    """Whether the user removed the file; an ingestion failure otherwise."""


@dataclass(frozen=True)
class AbsorbJournal:
    """One absorb: the keys that move and the registry change that follows them."""

    id: str
    moves: dict[str, str]
    """Each absorbed source's key, and the key it takes below its parent."""
    add: dict[str, str] = field(default_factory=dict)
    """Registry entries the absorb writes, label to path."""
    drop: list[str] = field(default_factory=list)
    """Registry labels the absorb removes."""
    records: dict[str, HeldOut] = field(default_factory=dict)
    """The skip record of each file the absorbed sources hold out, by the key it takes.

    From the first write of the journal to its removal, this is what those
    sources hold out; the skip record files take it when the keys have moved.
    """
    phase: AbsorbPhase = AbsorbPhase.LIFT

    def lifted(self, key: str) -> str:
        """The key *key* holds between the two phases, under a prefix no source has."""
        return f".absorb-{self.id}/{key}"

    def at(self, phase: AbsorbPhase) -> AbsorbJournal:
        """This journal at *phase*."""
        return replace(self, phase=phase)


def journal_path(data_root: Path) -> Path:
    """Where the journal of *data_root* lives."""
    return data_root / PENDING_ABSORB_FILENAME


def absorb_pending(data_root: Path) -> bool:
    """Whether an absorb on *data_root* started and did not finish."""
    return journal_path(data_root).exists()


def write_journal(data_root: Path, journal: AbsorbJournal) -> None:
    """Replace the journal atomically; a failure raises, so no key moves without it."""
    path = journal_path(data_root)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(journal), sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_journal(data_root: Path) -> AbsorbJournal | None:
    """The journal of *data_root*, or None without one; raises when it cannot be read."""
    path = journal_path(data_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AbsorbJournal(
            id=str(raw["id"]),
            moves={str(old): str(new) for old, new in raw["moves"].items()},
            add={str(label): str(target) for label, target in raw["add"].items()},
            drop=[str(label) for label in raw["drop"]],
            records={str(key): _held_out(record) for key, record in raw["records"].items()},
            phase=AbsorbPhase(raw["phase"]),
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise AbsorbJournalError(_UNREADABLE.format(path=path, error=exc)) from exc


def _held_out(raw: dict[str, object]) -> HeldOut:
    """One journal record; a malformed one raises as an unreadable journal does."""
    reason = raw["reason"]
    return HeldOut(
        hash=str(raw["hash"]),
        reason=None if reason is None else str(reason),
        removed=raw["removed"] is True,
    )


def delete_journal(data_root: Path) -> None:
    """Remove the journal; the absorb it recorded is complete."""
    journal_path(data_root).unlink(missing_ok=True)
