"""Per-text token cost for the retrieval prompt budget, measured by the chat engine."""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from lilbee.providers.base import ProviderError, estimate_budget_tokens

log = logging.getLogger(__name__)

# Tokenize requests in flight at once. llama-server tokenizes on its HTTP threads.
COUNT_WORKERS = 8


class TokenCost:
    """Token cost of one text: the engine's own count, never below the character rule.

    Each distinct text is counted once and cached for the life of the object, so
    a request holds one object and the cache never outlives it. The provider's
    window check costs text at the character rule, so a prompt costed lower here
    would be refused there; that rule is the floor. An engine with no tokenize
    route, or one that fails, switches every later text to the character rule.
    """

    def __init__(self, count: Callable[[str], int | None]) -> None:
        self._count = count
        self._counted: dict[str, int] = {}
        self._measuring = True

    def __call__(self, text: str) -> int:
        floor = estimate_budget_tokens(text)
        if not self._measuring:
            return floor
        if text not in self._counted:
            tokens = self._measure(text)
            if tokens is None:
                self._measuring = False
                return floor
            self._counted[text] = tokens
        return max(self._counted[text], floor)

    def warm(self, texts: Iterable[str]) -> None:
        """Count the uncounted *texts* at once, up to COUNT_WORKERS requests in flight."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._counted]
        if not self._measuring or len(todo) < 2:
            return
        with ThreadPoolExecutor(max_workers=min(COUNT_WORKERS, len(todo))) as pool:
            futures = [pool.submit(contextvars.copy_context().run, self._measure, t) for t in todo]
            results = [f.result() for f in futures]
        for text, tokens in zip(todo, results, strict=True):
            if tokens is None:
                self._measuring = False
                return
            self._counted[text] = tokens

    def _measure(self, text: str) -> int | None:
        try:
            return self._count(text)
        except ProviderError:
            log.debug("Chat engine tokenize failed; budgeting with the character rule.")
            return None
