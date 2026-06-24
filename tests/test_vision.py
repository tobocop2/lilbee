"""Tests for the vision-message helpers (prompt resolution + image messages)."""

from __future__ import annotations


class TestResolveOcrPrompt:
    def test_native_prompt_for_known_family(self) -> None:
        from lilbee.vision import resolve_ocr_prompt

        assert resolve_ocr_prompt("vendor/glm-ocr-1b") == "OCR"
        assert resolve_ocr_prompt("deepseek-ocr").startswith("<|grounding|>")

    def test_generic_prompt_for_unknown_model(self) -> None:
        from lilbee.vision import OCR_PROMPT, resolve_ocr_prompt

        assert resolve_ocr_prompt("vendor/qwen-vl") == OCR_PROMPT


class TestPngToDataUrl:
    def test_encodes_png_bytes(self) -> None:
        import base64

        from lilbee.vision import _png_to_data_url

        png_bytes = b"\x89PNG\r\n\x1a\n"
        result = _png_to_data_url(png_bytes)
        assert result.startswith("data:image/png;base64,")
        encoded = result.split(",", 1)[1]
        assert base64.b64decode(encoded) == png_bytes


class TestBuildVisionMessages:
    def test_builds_openai_format(self) -> None:
        from lilbee.vision import build_vision_messages

        messages = build_vision_messages("describe this", b"fake-png")
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "describe this"
