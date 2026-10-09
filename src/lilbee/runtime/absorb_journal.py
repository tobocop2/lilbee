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


class AbsorbPhase(StrEnum):
    """How far the re-key of an absorb has gone."""

    LIFT = "lift"
    LAND = "land"


class AbsorbJournalError(RuntimeError):
    """Raised when the journal of an interrupted absorb cannot be read."""


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
            phase=AbsorbPhase(raw["phase"]),
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise AbsorbJournalError(
            f"Cannot read {path} ({exc}). An add was interrupted while it moved a source "
            "into its parent folder, and lilbee cannot finish it without that file."
        ) from exc


def delete_journal(data_root: Path) -> None:
    """Remove the journal; the absorb it recorded is complete."""
    journal_path(data_root).unlink(missing_ok=True)
