"""The order of the model picks of each role.

A pick whose engine needs a download puts its model in the role late, when the install
ends. The user can pick again in that time. Each pick takes a number here, and a late
pick puts its model in the role only while its number is the newest of the role.
"""

from __future__ import annotations


class PickOrder:
    """Numbers each model pick of a role and remembers the newest one.

    Only the UI thread reads and writes it, so it needs no lock.
    """

    def __init__(self) -> None:
        self._newest: dict[str, tuple[int, str]] = {}

    def take(self, key: str, ref: str) -> int:
        """Record the pick of *ref* for the role *key* as the newest; return its number."""
        number = self._newest.get(key, (0, ""))[0] + 1
        self._newest[key] = (number, ref)
        return number

    def is_newest(self, key: str, number: int) -> bool:
        """Whether no pick of the role *key* came after the pick *number*."""
        return self._newest[key][0] == number

    def newest_ref(self, key: str) -> str:
        """The model of the newest pick of the role *key*."""
        return self._newest[key][1]
