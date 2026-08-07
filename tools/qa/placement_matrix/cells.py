"""The matrix of placement configurations to exercise, and how to split it up.

A cell is one (model, visible cards, resident tenant, config) combination. Cells
are generated deterministically so a run can be sharded across pods and merged,
and each carries the hardware it needs so a pod runs only what it can.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

_GB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    """A model in the matrix and the placement branch it exists to reach."""

    key: str
    ref: str
    probes: str
    """Which placement branch this model exists to reach; printed by a run."""
    role: str = "chat"


# One model per decision boundary. Adding a model that reaches a branch already
# covered costs a download and proves nothing new.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="tiny",
        ref="Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        probes="single card with room to spare; multi-slot",
    ),
    ModelSpec(
        key="kv-starved",
        ref="Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q8_0.gguf",
        probes="weights fit one 24GiB card, a usable context does not: forces a split",
    ),
    ModelSpec(
        key="tight-split",
        ref="bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        probes="needs two 24GiB cards and lands in the tight group that serves anyway",
    ),
    ModelSpec(
        key="spill",
        # Sharded; the first part is the ref, siblings are resolved beside it.
        ref=("unsloth/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf"),
        probes="exceeds the box: must load by spilling to system memory, and is MoE",
    ),
    ModelSpec(
        key="embed",
        ref="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf",
        probes="co-tenant whose reservation is held back from chat",
        role="embed",
    ),
)


@dataclass(frozen=True)
class Cell:
    """One placement configuration to plan, launch and judge."""

    model: ModelSpec
    cards: int
    """How many GPUs to expose (CUDA_VISIBLE_DEVICES is masked to this many)."""
    ballast_gib: tuple[int, ...] = ()
    """VRAM a fake tenant holds on each visible card, so capacity goes uneven."""
    usable_fraction: float = 0.9
    ctx_target: int = 16384
    with_embed: bool = False
    """Place the embedding model beside chat, so chat is charged its reservation."""

    @property
    def id(self) -> str:
        """Stable identifier: the result filename, and how shards stay disjoint."""
        ballast = "-".join(str(g) for g in self.ballast_gib) or "0"
        return (
            f"{self.model.key}__c{self.cards}__b{ballast}"
            f"__f{self.usable_fraction:g}__t{self.ctx_target}"
            f"__{'embed' if self.with_embed else 'solo'}"
        )


@dataclass(frozen=True)
class Matrix:
    """The cells a run will attempt, in a deterministic order."""

    cells: tuple[Cell, ...] = field(default_factory=tuple)

    def shard(self, index: int, count: int) -> tuple[Cell, ...]:
        """Cells for shard *index* of *count*, disjoint and jointly covering."""
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard {index}/{count} is not a valid split")
        return self.cells[index::count]


def build_matrix(
    models: Sequence[ModelSpec] = DEFAULT_MODELS,
    *,
    max_cards: int = 4,
    fractions: Sequence[float] = (0.9,),
    ctx_targets: Sequence[int] = (16384,),
) -> Matrix:
    """Every combination worth planning, ordered so shards interleave evenly.

    Card counts run 1..*max_cards* because the split decision changes with the
    count, and ballast is applied to the first card only: an uneven pair is the
    case a proportional ratio exists for, and an even one never exercises it.
    """
    chat = [m for m in models if m.role == "chat"]
    has_embed = any(m.role == "embed" for m in models)
    cells: list[Cell] = []
    for model in chat:
        for cards in range(1, max_cards + 1):
            for fraction in fractions:
                for ctx_target in ctx_targets:
                    cells.append(
                        Cell(
                            model=model,
                            cards=cards,
                            usable_fraction=fraction,
                            ctx_target=ctx_target,
                        )
                    )
                    if cards > 1:
                        cells.append(
                            Cell(
                                model=model,
                                cards=cards,
                                ballast_gib=(8,) + (0,) * (cards - 1),
                                usable_fraction=fraction,
                                ctx_target=ctx_target,
                            )
                        )
                    if has_embed:
                        cells.append(
                            Cell(
                                model=model,
                                cards=cards,
                                usable_fraction=fraction,
                                ctx_target=ctx_target,
                                with_embed=True,
                            )
                        )
    return Matrix(cells=tuple(cells))


def iter_pairs(cells: Sequence[Cell]) -> Iterator[tuple[Cell, Cell]]:
    """Cells differing in exactly one knob, which is what a metamorphic check needs."""
    for left in cells:
        for right in cells:
            if left.id >= right.id or left.model.key != right.model.key:
                continue
            if left.with_embed != right.with_embed:
                continue
            differing = [
                left.cards != right.cards,
                left.ballast_gib != right.ballast_gib,
                left.usable_fraction != right.usable_fraction,
                left.ctx_target != right.ctx_target,
            ]
            if sum(differing) == 1:
                yield left, right
