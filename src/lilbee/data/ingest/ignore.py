"""``.lilbeeignore`` loading and matching for the discovery walk.

Pattern syntax and precedence are gitignore's, matched by ``pathspec`` rather
than re-implemented: negation, ``**``, dir-only trailing slashes and anchoring
are subtle enough that a hand-rolled matcher is a standing bug source.

Two layers apply. The corpus layer is a single file at the resolved data root,
so it is project-local inside a ``lilbee init`` tree and global otherwise -- the
same resolution ``--data-dir`` / ``LILBEE_DATA`` / ``.lilbee/`` already gives
every other piece of lilbee state. Tree layers are ``.lilbeeignore`` files at any
depth inside a walked root, each scoped to its own directory and below. Deeper
layers win, so a repo can re-include what the corpus layer drops.
"""

from __future__ import annotations

from pathlib import Path

from pathspec import PathSpec

IGNORE_FILENAME = ".lilbeeignore"

_SYNTAX = "gitignore"

IGNORE_TEMPLATE = """\
# Files sync keeps out of the index. Same syntax as .gitignore.
# This file covers everything you index. Put a .lilbeeignore inside a
# tree to scope patterns to it; the deeper file wins, so ! adds back.
#
# node_modules/, __pycache__, venv, build, dist, target, vendor and
# dot-directories are already skipped. Add what those miss:
#
# *.min.js
# testdata/
# fixtures/
"""


def _load_spec(path: Path) -> PathSpec | None:
    """Compile the ignore file at *path*, or None if it holds no live pattern.

    A comment or a blank line still compiles to a pattern object, one whose
    ``include`` is None because it can never match. The file ``lilbee init``
    scaffolds is entirely comments, so keeping those would charge every walked
    file a lookup against a spec with nothing in it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    spec = PathSpec.from_lines(_SYNTAX, text.splitlines())
    return spec if any(pattern.include is not None for pattern in spec.patterns) else None


class IgnoreRules:
    """Nested ``.lilbeeignore`` matching, with per-directory compilation cached.

    One instance serves a whole sync pass: the discovery walk asks about each
    entry it visits, and the pass that reconciles the index against the corpus
    asks about paths the walk pruned before reaching. Both answers come from the
    same compiled specs, so the two cannot disagree.
    """

    def __init__(self, corpus_spec: PathSpec | None = None) -> None:
        self._corpus_spec = corpus_spec
        self._specs: dict[Path, PathSpec | None] = {}
        self._chains: dict[Path, tuple[tuple[Path, PathSpec], ...]] = {}

    @classmethod
    def for_corpus(cls) -> IgnoreRules:
        """Build rules carrying the corpus-wide layer from the resolved data root."""
        from lilbee.core.config import active_config

        return cls(_load_spec(active_config().data_root / IGNORE_FILENAME))

    def _spec_for(self, directory: Path) -> PathSpec | None:
        if directory not in self._specs:
            self._specs[directory] = _load_spec(directory / IGNORE_FILENAME)
        return self._specs[directory]

    def _chain_for(self, directory: Path, base: Path) -> tuple[tuple[Path, PathSpec], ...]:
        """The ignore files covering *directory*, deepest first.

        Cached per directory and built from the parent's chain, so a tree with no
        ignore files costs one dict hit per directory and nothing per file. The
        walk up terminates on *base*, which every caller guarantees is an
        ancestor by building the path from it.
        """
        cached = self._chains.get(directory)
        if cached is not None:
            return cached
        parent = () if directory == base else self._chain_for(directory.parent, base)
        spec = self._spec_for(directory)
        chain = (((directory, spec),) if spec is not None else ()) + parent
        self._chains[directory] = chain
        return chain

    def _verdict(self, entry: Path, *, base: Path, is_dir: bool) -> bool | None:
        """Whether the layers covering *entry* exclude it, or None if none match.

        Layers are consulted deepest first and the first one that expresses an
        opinion wins, which is what makes a nested file able to re-include what
        a shallower one dropped.
        """
        for directory, spec in self._chain_for(entry.parent, base):
            relative = entry.relative_to(directory).as_posix()
            verdict = spec.check_file(relative + "/" if is_dir else relative).include
            if verdict is not None:
                return verdict
        if self._corpus_spec is not None:
            relative = entry.relative_to(base).as_posix()
            return self._corpus_spec.check_file(relative + "/" if is_dir else relative).include
        return None

    def excludes_entry(self, entry: Path, *, base: Path, is_dir: bool) -> bool:
        """Whether *entry* itself is excluded, assuming its ancestors are not.

        The discovery walk prunes top-down, so by the time it asks about an entry
        every directory above it has already been kept.
        """
        return self._verdict(entry, base=base, is_dir=is_dir) is True

    def excludes_path(self, path: Path, *, base: Path) -> bool:
        """Whether *path* or any directory between it and *base* is excluded.

        Answers for files the walk never enumerated because it pruned a parent,
        which is what lets the index be reconciled without a second walk. A file
        under an excluded directory stays excluded even if a pattern names it
        back, matching git: the walk never descends far enough to reconsider.

        *path* must sit under *base*; both come from the same resolution step,
        which builds one from the other.
        """
        parts = path.relative_to(base).parts
        current = base
        last = len(parts) - 1
        for index, part in enumerate(parts):
            current = current / part
            if self.excludes_entry(current, base=base, is_dir=index != last):
                return True
        return False
