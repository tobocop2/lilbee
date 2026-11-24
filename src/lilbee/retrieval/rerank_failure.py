"""What a user reads when the rerank step of a search fails.

A leaf module: the searcher and the reranker both need it, and the reranker
imports the query package that holds the searcher.
"""

from __future__ import annotations

from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.providers.base import ProviderError, ProviderErrorKind

# The cause and remedy a user reads for each error kind. The error text never reaches a client:
# it can hold a server address with credentials, so the full error goes to the log only.
_FAILURE_TEXTS: dict[ProviderErrorKind, tuple[str, str]] = {
    ProviderErrorKind.AUTH: (
        "The reranker refused the credentials.",
        "Check the API key of the reranker preset.",
    ),
    ProviderErrorKind.RATE_LIMIT: ("The reranker is rate limited.", "Wait, then search again."),
    ProviderErrorKind.NOT_FOUND: (
        "The reranker model or preset was not found.",
        "Check reranker_model and the preset with lilbee status.",
    ),
    ProviderErrorKind.BAD_REQUEST: (
        "The reranker refused the request.",
        "Check reranker_model with lilbee status.",
    ),
    ProviderErrorKind.CONNECTION: (
        "The reranker did not answer.",
        "Check that the reranker server runs, then search again.",
    ),
    ProviderErrorKind.SERVER: (
        "The reranker reported a server error.",
        "Search again, or check the reranker server log.",
    ),
    ProviderErrorKind.CONTEXT_OVERFLOW: (
        "A candidate is longer than the reranker window.",
        "Use a reranker with a longer window.",
    ),
    ProviderErrorKind.CAPACITY: (
        "The reranker has no free capacity.",
        "Search again when other work ends.",
    ),
    ProviderErrorKind.STARTING: (
        "The reranker is still starting.",
        "Search again in a moment.",
    ),
    ProviderErrorKind.WARM_TIMEOUT: (
        "The reranker did not finish starting in time.",
        "Run the reranker on a GPU, or clear the reranker_model setting.",
    ),
    ProviderErrorKind.ENGINE_MISSING: (
        "The reranker engine is not installed.",
        "Run lilbee status to see which engine is missing.",
    ),
}
_UNKNOWN_TEXT = (
    "The reranker raised an error.",
    "Search again, or clear the reranker_model setting. Run lilbee status to check the reranker.",
)


class RerankScoreError(ValueError):
    """A provider returned no scores, or a count that cannot pair with the candidates."""

    def __init__(self, returned: int | None, expected: int) -> None:
        got = "no scores" if returned is None else f"{returned} scores"
        self.detail = f"returned {got} for {expected} candidates"
        super().__init__(f"Reranker {self.detail}")


def rerank_failure(exc: Exception) -> HealthWarning:
    """The warning for a failed rerank: a fixed cause and remedy for the kind of *exc*."""
    if isinstance(exc, RerankScoreError):
        cause, remedy = f"The reranker {exc.detail}.", _UNKNOWN_TEXT[1]
    else:
        if isinstance(exc, ProviderError):
            kind = exc.kind
        elif isinstance(exc, OSError):
            kind = ProviderErrorKind.CONNECTION
        else:
            kind = ProviderErrorKind.UNKNOWN
        cause, remedy = _FAILURE_TEXTS.get(kind, _UNKNOWN_TEXT)
    return HealthWarning(
        code=WarningCode.RERANK_FAILED,
        message=f"Reranking failed: results are in retrieval order. {cause}",
        remedy=remedy,
    )
