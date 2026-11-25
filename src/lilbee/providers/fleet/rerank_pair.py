"""The token layout a cross-encoder reads: ``head query middle document tail``.

The special tokens follow llama.cpp's own pair rule (``format_prompt_rerank``):
BOS when ``add_bos_token``; EOS then SEP between the texts when ``add_eos_token``
and ``add_sep_token``; EOS at the end when ``add_eos_token``. The converter
writes the three flags from the model's Hugging Face pair template. A GGUF from
before the converter wrote ``add_sep_token`` has no such key; the absent flag
then takes the value of the Hugging Face pair processor for the vocabulary: Bert
processing (one separator) for WordPiece, Roberta processing (two) for any other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# The tokenizer.ggml.model value of a WordPiece vocabulary.
_WORDPIECE = "bert"


@dataclass(frozen=True)
class RerankPair:
    """The special token ids around the query and the document of one pair."""

    head: tuple[int, ...]
    middle: tuple[int, ...]
    tail: tuple[int, ...]

    def ids(self, query: Sequence[int], document: Sequence[int], cap: int | None) -> list[int]:
        """The pair's token ids, cut to *cap* tokens by Hugging Face's longest-first rule."""
        if cap is not None:
            room = cap - len(self.head) - len(self.middle) - len(self.tail)
            query, document = _longest_first(query, document, room)
        return [*self.head, *query, *self.middle, *document, *self.tail]

    def to_state(self) -> dict[str, list[int]]:
        """JSON-safe form for the engine state file."""
        return {"head": list(self.head), "middle": list(self.middle), "tail": list(self.tail)}

    @classmethod
    def from_state(cls, payload: Mapping[str, Sequence[int]]) -> RerankPair:
        """Rebuild a pair layout from :meth:`to_state` output."""
        return cls(
            head=tuple(int(i) for i in payload["head"]),
            middle=tuple(int(i) for i in payload["middle"]),
            tail=tuple(int(i) for i in payload["tail"]),
        )


def _longest_first(
    first: Sequence[int], second: Sequence[int], room: int
) -> tuple[Sequence[int], Sequence[int]]:
    """Cut two texts to *room* tokens together, as Hugging Face ``longest_first`` does.

    If the shorter text fits in half the room, only the longer text is cut.
    Otherwise each text keeps half, and the longer one keeps the odd token.
    """
    if len(first) + len(second) <= room:
        return first, second
    room = max(room, 0)
    short, long = sorted((len(first), len(second)))
    long = short if short > room else max(short, room - short)
    if short + long > room:
        short = room // 2
        long = short + room % 2
    if len(first) > len(second):
        return first[:long], second[:short]
    return first[:short], second[:long]


def rerank_pair_from_meta(meta: Mapping[str, str] | None) -> RerankPair | None:
    """The pair layout from a reranker GGUF's tokenizer facts, or None when they are missing."""
    if not meta:
        return None
    bos = _token_id(meta, "bos_token_id")
    sep = _token_id(meta, "sep_token_id")
    eos = _token_id(meta, "eos_token_id")
    # llama.cpp ends a text with SEP when the file names no EOS; the converter
    # names EOS as SEP when the tokenizer has no SEP of its own.
    eos = sep if eos is None else eos
    if bos is None or eos is None:
        return None
    sep = eos if sep is None else sep
    add_bos = _flag(meta, "add_bos_token", default=True)
    add_eos = _flag(meta, "add_eos_token", default=True)
    add_sep = _flag(meta, "add_sep_token", default=meta.get("tokenizer_model") != _WORDPIECE)
    ends = (eos,) if add_eos else ()
    return RerankPair(
        head=(bos,) if add_bos else (),
        middle=(*ends, sep) if add_sep else ends,
        tail=ends,
    )


def _token_id(meta: Mapping[str, str], key: str) -> int | None:
    """The token id stored under *key*, or None when the file has none."""
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _flag(meta: Mapping[str, str], key: str, *, default: bool) -> bool:
    """The boolean stored under *key*, or *default* when the file has none."""
    raw = meta.get(key)
    if raw is None:
        return default
    return raw.strip().lower() == "true"
