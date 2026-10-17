"""Where a source is, and the name a person sees for it: the deepest registered root's label."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath

from .lance_helpers import escape_sql_string


def root_location(root: str) -> str:
    """The location of the registered root path *root*: its POSIX spelling."""
    return PurePath(root).as_posix()


def _escape_like_wildcards(value: str) -> str:
    """*value* with the LIKE wildcards and the escape character escaped, matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _under_sql(column: str, root: str) -> str:
    """The predicate for *column* holding *root* or a location below it."""
    literal = escape_sql_string(root)
    return f"({column} = '{literal}' OR starts_with({column}, '{literal}/'))"


class SourceLocations:
    """The rule between a stored location and its shown name under one source registry.

    A location below a registered root shows the label of the deepest such root
    plus the rest of the path; any other location shows itself.
    """

    def __init__(self, roots: Mapping[str, str]) -> None:
        self.roots = dict(roots)
        self._at = {label: root_location(root) for label, root in roots.items()}
        self._deepest_first = sorted(
            ((root, label) for label, root in self._at.items()),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    @property
    def deepest_first(self) -> list[tuple[str, str]]:
        """Each registered root's location and label, a root below another before it."""
        return list(self._deepest_first)

    def name(self, location: str) -> str:
        """The name a person sees for the source at *location*."""
        for root, label in self._deepest_first:
            if location == root or location.startswith(f"{root}/"):
                return label + location[len(root) :]
        return location

    def location(self, name: str) -> str:
        """Where the source named *name* is: its root's location plus the rest, else *name*."""
        label, _, rest = name.partition("/")
        root = self._at.get(label)
        if root is None:
            return name
        return f"{root}/{rest}" if rest else root

    def search_filter(self, term: str, path: str, title: str) -> str:
        """The predicate for a shown name in *path*, or a title in *title*, holding *term*.

        Case-insensitive. Each root's arm leaves out the roots below it, so a
        location matches by the one name it shows.
        """
        pattern = f"LIKE '%{escape_sql_string(_escape_like_wildcards(term.lower()))}%' ESCAPE '\\'"
        arms: list[str] = []
        unders: dict[str, str] = {}
        for root, label in self._deepest_first:
            under = _under_sql(path, root)
            deeper = [f"NOT {sql}" for below, sql in unders.items() if below.startswith(f"{root}/")]
            shown = f"concat('{escape_sql_string(label)}', substr({path}, {len(root) + 1}))"
            arms.append(" AND ".join([under, *deeper, f"LOWER({shown}) {pattern}"]))
            unders[root] = under
        outside = [f"NOT {sql}" for sql in unders.values()]
        arms.append(" AND ".join([*outside, f"LOWER({path}) {pattern}"]))
        arms.append(f"LOWER({title}) {pattern}")
        return "(" + " OR ".join(f"({arm})" for arm in arms) + ")"
