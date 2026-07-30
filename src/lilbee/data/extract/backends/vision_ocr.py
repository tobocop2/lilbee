"""lilbee's vision model exposed as a xberg custom OCR backend."""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from lilbee.data.types import MARKDOWN_MIME, OcrBackendName
from lilbee.vision import resolve_ocr_prompt

from .registry import BackendKind, XbergBinding, register_binding

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    # OcrBackend types the config as the public xberg.OcrConfig; its
    # backend_options arrives as a JSON string at runtime (see _OcrConfigView).
    from xberg import ExtractedDocument, OcrBackendType, OcrConfig

# xberg doesn't propagate contextvars into process_image, so per-request state
# travels as this token in OcrConfig.backend_options and resolves via the registry.
_REQUEST_TOKEN_KEY = "req"  # noqa: S105  # JSON key name, not a secret


@dataclass(frozen=True)
class OcrRequestContext:
    """Per-extraction state the backend needs but xberg won't carry for it."""

    on_page: Callable[[], None] | None = None
    timeout: float = 0.0


class _OcrRequestRegistry:
    """Token-keyed request contexts; lock-guarded (process_image runs on xberg threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_token: dict[str, OcrRequestContext] = {}

    def register(self, ctx: OcrRequestContext) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._by_token[token] = ctx
        return token

    def get(self, token: str | None) -> OcrRequestContext | None:
        if token is None:
            return None
        with self._lock:
            return self._by_token.get(token)

    def unregister(self, token: str) -> None:
        with self._lock:
            self._by_token.pop(token, None)


ocr_requests = _OcrRequestRegistry()


@contextmanager
def ocr_request(
    *, on_page: Callable[[], None] | None = None, timeout: float = 0.0
) -> Generator[str, None, None]:
    """Register a per-extraction context and yield its token for OcrConfig.backend_options."""
    token = ocr_requests.register(OcrRequestContext(on_page=on_page, timeout=timeout))
    try:
        yield token
    finally:
        ocr_requests.unregister(token)


def backend_options_for(token: str) -> dict[str, str]:
    """Carry a request token in OcrConfig.backend_options for process_image to read."""
    return {_REQUEST_TOKEN_KEY: token}


class _OcrConfigView:
    """Typed reader over the xberg OcrConfig passed to process_image. The native
    round-trip hands ``backend_options`` back as a JSON string, so ``request_token``
    accepts both the dict and string shapes."""

    def __init__(self, config: OcrConfig) -> None:
        self._config = config

    @property
    def vlm_prompt(self) -> str | None:
        return self._config.vlm_prompt

    @property
    def request_token(self) -> str | None:
        options = self._config.backend_options
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                return None
        token = options.get(_REQUEST_TOKEN_KEY) if isinstance(options, dict) else None
        return token if isinstance(token, str) else None


class _OcrFn(Protocol):
    # Positional-only so the provider's vision_ocr (named png_bytes) matches structurally.
    def __call__(
        self, image_bytes: bytes, model: str, prompt: str, /, *, timeout: float
    ) -> str: ...


def _lilbee_version() -> str:
    try:
        return version("lilbee")
    except PackageNotFoundError:
        return "0"


class VisionOcrBackend:
    """Routes xberg OCR calls to lilbee's vision model through the injected
    ``ocr_fn`` (single-image OCR) and ``model_ref_fn`` (the active vision model)."""

    def __init__(self, *, ocr_fn: _OcrFn, model_ref_fn: Callable[[], str]) -> None:
        self._ocr_fn = ocr_fn
        self._model_ref_fn = model_ref_fn

    def name(self) -> str:
        return OcrBackendName.LILBEE_VISION

    def version(self) -> str:
        return _lilbee_version()

    def supported_languages(self) -> list[str]:
        return []

    def supports_language(self, lang: str) -> bool:
        return True

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def backend_type(self) -> OcrBackendType:
        from xberg import OcrBackendType

        return OcrBackendType.CUSTOM

    def supports_table_detection(self) -> bool:
        return False

    def supports_document_processing(self) -> bool:
        return False

    def emits_structured_markdown(self) -> bool:
        # False keeps xberg's layout reconstruction (the validated path). The model
        # does emit markdown, so True is a valid optimization but changes OCR output.
        return False

    def process_image(self, image_bytes: bytes, config: OcrConfig) -> ExtractedDocument:
        # xberg passes a native OcrConfig and expects a native ExtractedDocument back.
        from xberg import ExtractedDocument

        view = _OcrConfigView(config)
        model = self._model_ref_fn()
        prompt = view.vlm_prompt or resolve_ocr_prompt(model)
        ctx = ocr_requests.get(view.request_token)
        text = self._ocr_fn(image_bytes, model, prompt, timeout=ctx.timeout if ctx else 0.0)
        if ctx is not None and ctx.on_page is not None:
            ctx.on_page()
        return ExtractedDocument(content=text, mime_type=MARKDOWN_MIME)

    def process_image_file(self, path: str, config: OcrConfig) -> ExtractedDocument:
        # OCR an image file by reading its bytes and delegating to process_image
        # (mirrors xberg's default OcrBackend::process_image_file).
        return self.process_image(Path(path).read_bytes(), config)

    def process_document(self, _path: str, _config: OcrConfig) -> ExtractedDocument:
        # Image-only backend: xberg only calls this when supports_document_processing()
        # is True, which it never is here, so document-level OCR is unreachable.
        raise NotImplementedError(
            "Document-level OCR is not supported by the lilbee vision backend"
        )


register_binding(
    XbergBinding(
        kind=BackendKind.OCR,
        name=OcrBackendName.LILBEE_VISION,
        enabled=lambda cfg: bool(cfg.vision_model),
        make=lambda provider, cfg: VisionOcrBackend(
            ocr_fn=provider.vision_ocr, model_ref_fn=lambda: cfg.vision_model
        ),
    )
)
