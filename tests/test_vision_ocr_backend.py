"""Tests for the lilbee vision OCR backend (xberg custom OCR plugin)."""

from __future__ import annotations

from lilbee.data.ingest.types import MARKDOWN_MIME, OcrBackendName
from lilbee.data.ingest.vision_ocr_backend import (
    VisionOcrBackend,
    backend_options_for,
    ocr_request,
    ocr_requests,
)


def _backend(ocr_fn=None, model="vendor/glm-ocr"):
    calls: list[tuple] = []

    def default_fn(image_bytes, model, prompt, *, timeout):
        calls.append((image_bytes, model, prompt, timeout))
        return "# extracted"

    be = VisionOcrBackend(ocr_fn=ocr_fn or default_fn, model_ref_fn=lambda: model)
    return be, calls


def _cfg(*, vlm_prompt=None, backend_options=None):
    """The native xberg OcrConfig object xberg hands process_image."""
    from xberg import OcrConfig

    return OcrConfig(
        backend=OcrBackendName.LILBEE_VISION,
        vlm_prompt=vlm_prompt,
        backend_options=backend_options,
    )


class TestProtocol:
    def test_name_is_enum_value(self):
        be, _ = _backend()
        assert be.name() == OcrBackendName.LILBEE_VISION == "lilbee-vision"

    def test_backend_type_is_custom(self):
        from xberg import OcrBackendType

        be, _ = _backend()
        assert be.backend_type() == OcrBackendType.CUSTOM

    def test_supports_all_languages(self):
        be, _ = _backend()
        assert be.supported_languages() == []
        assert be.supports_language("eng") is True
        assert be.supports_language("zho") is True

    def test_version_is_str(self):
        be, _ = _backend()
        assert isinstance(be.version(), str) and be.version()

    def test_version_falls_back_when_package_missing(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        from lilbee.data.ingest import vision_ocr_backend as mod

        def _raise(_name):
            raise PackageNotFoundError

        monkeypatch.setattr(mod, "version", _raise)
        be, _ = _backend()
        assert be.version() == "0"

    def test_initialize_shutdown_noop(self):
        be, _ = _backend()
        assert be.initialize() is None
        assert be.shutdown() is None

    def test_capability_flags_are_image_only(self):
        be, _ = _backend()
        assert be.supports_table_detection() is False
        assert be.supports_document_processing() is False
        assert be.emits_structured_markdown() is False

    def test_process_image_file_reads_bytes_and_delegates(self, tmp_path):
        be, calls = _backend()
        img = tmp_path / "scan.png"
        img.write_bytes(b"PNG-on-disk")
        out = be.process_image_file(str(img), _cfg())
        assert out.content == "# extracted"
        assert out.mime_type == MARKDOWN_MIME
        assert calls[0][0] == b"PNG-on-disk"

    def test_process_document_is_unsupported(self):
        import pytest

        be, _ = _backend()
        with pytest.raises(NotImplementedError):
            be.process_document("/tmp/doc.pdf", _cfg())


class TestProcessImage:
    def test_returns_extracted_document(self):
        """xberg expects a native ExtractedDocument back, with the OCR text as markdown."""
        from xberg import ExtractedDocument

        be, _ = _backend()
        out = be.process_image(b"PNG", _cfg())
        assert isinstance(out, ExtractedDocument)
        assert out.content == "# extracted"
        assert out.mime_type == MARKDOWN_MIME

    def test_passes_model_and_resolved_prompt(self):
        be, calls = _backend(model="vendor/glm-ocr")
        be.process_image(b"PNG", _cfg())
        _, model, prompt, _ = calls[0]
        assert model == "vendor/glm-ocr"
        # glm-ocr has a native prompt; resolve_ocr_prompt picks it, not the generic one.
        assert prompt == "OCR"

    def test_vlm_prompt_overrides_resolved(self):
        be, calls = _backend()
        be.process_image(b"PNG", _cfg(vlm_prompt="custom prompt"))
        assert calls[0][2] == "custom prompt"

    def test_request_context_supplies_timeout_and_fires_progress(self):
        ticks: list[int] = []
        be, calls = _backend()
        with ocr_request(on_page=lambda: ticks.append(1), timeout=12.5) as token:
            be.process_image(b"PNG", _cfg(backend_options=backend_options_for(token)))
        assert calls[0][3] == 12.5
        assert ticks == [1]

    def test_json_string_backend_options_resolve(self):
        # The native round-trip hands backend_options back as a JSON STRING
        # (alef serializes the map). The token must resolve from that shape or
        # per-page progress and the OCR timeout silently vanish on every real
        # extraction, while dict-based unit tests stay green.
        import json

        ticks: list[int] = []
        be, calls = _backend()
        with ocr_request(on_page=lambda: ticks.append(1), timeout=7.5) as token:
            be.process_image(b"PNG", _cfg(backend_options=json.dumps(backend_options_for(token))))
        assert calls[0][3] == 7.5
        assert ticks == [1]

    def test_no_context_uses_zero_timeout_and_no_tick(self):
        be, calls = _backend()
        be.process_image(b"PNG", _cfg(backend_options=backend_options_for("unknown-token")))
        assert calls[0][3] == 0.0

    def test_malformed_backend_options_ignored(self):
        # Non-JSON and valid-but-non-object backend_options both resolve to no token.
        be, calls = _backend()
        be.process_image(b"PNG", _cfg(backend_options="not-json"))
        be.process_image(b"PNG", _cfg(backend_options="123"))
        assert calls[0][3] == 0.0
        assert calls[1][3] == 0.0


class TestRegistry:
    def test_token_registered_within_scope_and_cleaned_after(self):
        with ocr_request(timeout=3.0) as token:
            ctx = ocr_requests.get(token)
            assert ctx is not None and ctx.timeout == 3.0
        assert ocr_requests.get(token) is None

    def test_get_none_token_returns_none(self):
        assert ocr_requests.get(None) is None

    def test_backend_options_round_trip(self):
        token = "abc123"
        be, calls = _backend()
        be.process_image(b"PNG", _cfg(backend_options=backend_options_for(token)))
        # token not registered -> no context, zero timeout
        assert calls[0][3] == 0.0
