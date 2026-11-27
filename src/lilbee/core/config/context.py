"""Per-scope active config: which Config a library call or a request reads.

The process-global ``cfg`` backs the CLI, TUI, and HTTP daemon. It is a handle over
the published Config: a write publishes a new Config by one reference store. A
request binds :func:`request_config_scope` after its config check, so every ``cfg``
read in the request (and in the sync or ingest it starts) reads the Config published
when it started, plus its own writes. The library API (:class:`lilbee.Lilbee`)
instead binds a caller-supplied Config for the duration of each public method via
:func:`config_scope`, and the ingest path reads it through :func:`active_config`
instead of the global. Scoping is one ContextVar, so it stays isolated to the
entering task and propagates into the ``to_ingest_thread`` workers (they copy the
calling context), without mutating the process-global cfg that other clients share.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from pydantic import ValidationError

from lilbee.core.config.enums import OcrMode
from lilbee.core.config.model import ConfigPin, active_pin, cfg, published_config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lilbee.core.config.model import Config


def active_config() -> Config:
    """Return the library call's Config if one is bound, else the process-global ``cfg``."""
    pin = active_pin.get()
    if pin is None or pin.follows_writes:
        return cfg
    return pin.config


def validate_ocr_timeout(ocr_timeout: float | None) -> None:
    """Raise ``ValueError`` unless *ocr_timeout* satisfies the config field's own bound.

    ``None`` always passes (it means "keep the current setting"). Assigns into
    a throwaway copy of the active config, so the ``ocr_timeout`` field's own
    ``validate_assignment`` rule decides, the same rule the CLI already
    triggers through a direct ``cfg.ocr_timeout = value`` assignment.
    """
    if ocr_timeout is None:
        return
    scratch = active_config().model_copy()
    try:
        scratch.ocr_timeout = ocr_timeout
    except ValidationError as exc:
        raise ValueError(f"ocr_timeout: {exc.errors()[0]['msg']}") from exc


def validate_ocr_options(ocr: OcrMode | str | None, ocr_timeout: float | None) -> OcrMode | None:
    """Return *ocr* as an ``OcrMode`` after checking both per-call OCR options.

    Raises ``ValueError`` that names the bad value. ``None`` means "keep the setting".
    """
    validate_ocr_timeout(ocr_timeout)
    return None if ocr is None else OcrMode(ocr)


@contextmanager
def _bound(pin: ConfigPin | None) -> Iterator[None]:
    token = active_pin.set(pin)
    try:
        yield
    finally:
        active_pin.reset(token)


@contextmanager
def config_scope(config: Config) -> Iterator[None]:
    """Bind *config* as the active config for the duration of the block."""
    with _bound(ConfigPin(config, follows_writes=False)):
        yield


@contextmanager
def request_config_scope() -> Iterator[None]:
    """Pin the published Config for one request; the request's own writes move the pin."""
    with _bound(ConfigPin(published_config(), follows_writes=True)):
        yield


@contextmanager
def published_config_scope() -> Iterator[None]:
    """Read and write the published Config, whatever the caller pins (a config check)."""
    with _bound(None):
        yield
