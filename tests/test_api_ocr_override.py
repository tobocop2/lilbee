"""The Python API takes a per-call OCR mode and timeout, read by the real ingest pipeline."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lilbee import Lilbee
from lilbee.core.config import validate_ocr_options
from lilbee.core.config.enums import OcrMode
from tests.test_api import _isolate_cfg, _mock_embedder  # noqa: F401  (autouse fixtures)
from tests.test_ingest import _scan_result

DEFAULT_TIMEOUT = 300.0


class _Scan:
    """One scanned PDF in a Lilbee project; xberg's extract call records the config it gets."""

    def __init__(self, bee: Lilbee) -> None:
        self.bee = bee
        self.configs: list[Any] = []
        self.timeouts: list[float] = []
        bee.config.ocr = OcrMode.AUTO
        bee.config.vision_model = ""
        bee.config.ocr_timeout = DEFAULT_TIMEOUT
        self.pdf = bee.config.documents_dir / "scan.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 scanned")

    def run(self, call: Callable[[], Any]) -> tuple[bool, bool, float]:
        """Return (force_ocr, ocr is None, request timeout) of the OCR pass the call ran."""
        from lilbee.data.extract import document

        original = document.ocr_request

        def spy(*args: Any, **kwargs: Any) -> Any:
            self.timeouts.append(kwargs["timeout"])
            return original(*args, **kwargs)

        async def fake_extract(*args: Any, **kwargs: Any) -> Any:
            self.configs.append(kwargs["config"])
            return _scan_result([None])

        with (
            patch("lilbee.data.extract.xberg.aextract_document", new=fake_extract),
            patch.object(document, "ocr_request", new=spy),
        ):
            call()
        assert self.configs
        assert self.timeouts
        # A scan is read twice: a pass without OCR, then the OCR pass. The last one is the OCR pass.
        config = self.configs[-1]
        return (bool(config.force_ocr), config.ocr is None, self.timeouts[-1])


def _scan(tmp_path: Path) -> _Scan:
    return _Scan(Lilbee(tmp_path / "proj"))


def _outside_pdf(tmp_path: Path) -> Path:
    outside = tmp_path / "outside" / "ext.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.4 scanned")
    return outside


def test_sync_without_options_reads_the_setting(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(scan.bee.sync) == (False, False, DEFAULT_TIMEOUT)


def test_sync_ocr_all_forces_ocr_on_every_page(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.sync(ocr=OcrMode.ALL)) == (True, False, DEFAULT_TIMEOUT)


def test_sync_ocr_off_reads_no_ocr(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.sync(ocr=OcrMode.OFF)) == (False, True, DEFAULT_TIMEOUT)


def test_sync_ocr_timeout_reaches_the_ocr_request(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.sync(ocr_timeout=77.0)) == (False, False, 77.0)


def test_sync_ocr_timeout_zero_means_no_limit(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.sync(ocr_timeout=0.0)) == (False, False, 0.0)


def test_sync_ocr_string_is_coerced_to_the_mode(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.sync(ocr="all")) == (True, False, DEFAULT_TIMEOUT)


def test_add_ocr_all_forces_ocr(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    outside = _outside_pdf(tmp_path)
    scan.pdf.unlink()
    assert scan.run(lambda: scan.bee.add([outside], ocr=OcrMode.ALL)) == (
        True,
        False,
        DEFAULT_TIMEOUT,
    )


def test_add_without_options_reads_the_setting(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    outside = _outside_pdf(tmp_path)
    scan.pdf.unlink()
    assert scan.run(lambda: scan.bee.add([outside])) == (False, False, DEFAULT_TIMEOUT)


def test_add_ocr_timeout_reaches_the_ocr_request(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    outside = _outside_pdf(tmp_path)
    scan.pdf.unlink()
    assert scan.run(lambda: scan.bee.add([outside], ocr_timeout=55.0)) == (False, False, 55.0)


def test_rebuild_ocr_all_forces_ocr(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.rebuild(ocr=OcrMode.ALL)) == (True, False, DEFAULT_TIMEOUT)


def test_rebuild_ocr_timeout_reaches_the_ocr_request(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    assert scan.run(lambda: scan.bee.rebuild(ocr_timeout=33.0)) == (False, False, 33.0)


def test_sync_ocr_override_does_not_change_the_config(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    first = scan.run(lambda: scan.bee.sync(ocr=OcrMode.ALL))
    scan.configs.clear()
    scan.timeouts.clear()
    scan.pdf.write_bytes(b"%PDF-1.4 scanned again")
    second = scan.run(scan.bee.sync)
    assert first[0] is True
    assert second == (False, False, DEFAULT_TIMEOUT)
    assert scan.bee.config.ocr == OcrMode.AUTO


def test_sync_refuses_an_unknown_ocr_mode_by_name(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    with pytest.raises(ValueError, match="bogus"):
        scan.bee.sync(ocr="bogus")
    assert scan.configs == []


def test_sync_refuses_a_negative_ocr_timeout_by_name(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    with pytest.raises(ValueError, match="ocr_timeout"):
        scan.bee.sync(ocr_timeout=-1.0)
    assert scan.configs == []


def test_add_refuses_a_bad_ocr_option_before_registering(tmp_path: Path) -> None:
    bee = Lilbee(tmp_path / "proj")
    outside = _outside_pdf(tmp_path)
    with pytest.raises(ValueError, match="bogus"):
        bee.add([outside], ocr="bogus")
    assert list(bee.config.linked_roots) == []


def test_rebuild_refuses_a_bad_ocr_option_by_name(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    with pytest.raises(ValueError, match="bogus"):
        scan.bee.rebuild(ocr="bogus")
    assert scan.configs == []


def test_two_threads_keep_their_own_ocr_override(tmp_path: Path) -> None:
    seen: dict[str, list[tuple[bool, bool]]] = {"a": [], "b": []}
    barrier = threading.Barrier(2)

    async def fake_extract(*args: Any, **kwargs: Any) -> Any:
        config = kwargs["config"]
        seen[threading.current_thread().name].append((bool(config.force_ocr), config.ocr is None))
        return _scan_result([None])

    def one(name: str, mode: OcrMode) -> None:
        scan = _scan(tmp_path / name)
        barrier.wait(timeout=20)
        scan.bee.sync(ocr=mode)

    with patch("lilbee.data.extract.xberg.aextract_document", new=fake_extract):
        threads = [
            threading.Thread(target=one, args=("a", OcrMode.ALL), name="a"),
            threading.Thread(target=one, args=("b", OcrMode.OFF), name="b"),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    assert {name: rows[-1] if rows else None for name, rows in seen.items()} == {
        "a": (True, False),
        "b": (False, True),
    }


def test_validate_ocr_options_returns_the_enum_for_a_string() -> None:
    assert validate_ocr_options("off", 0.0) is OcrMode.OFF


def test_validate_ocr_options_passes_none_through() -> None:
    assert validate_ocr_options(None, None) is None


def test_validate_ocr_options_names_a_bad_mode() -> None:
    with pytest.raises(ValueError, match="bogus"):
        validate_ocr_options("bogus", None)


def test_validate_ocr_options_names_a_bad_timeout() -> None:
    with pytest.raises(ValueError, match="ocr_timeout"):
        validate_ocr_options(None, -2.0)
