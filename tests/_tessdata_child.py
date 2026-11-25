"""Child process for ``test_tessdata_race``: OCR one image on the cache the parent names.

Usage: ``python -m tests._tessdata_child <png> <start-file>``. It imports everything,
touches ``<start-file>.<pid>`` to say it is ready, and waits for the start file so
every child begins its OCR in the same instant. It then prints one JSON line with
the text the OCR read.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main(png: Path, start: Path) -> None:
    from xberg import ExtractionConfig, OcrConfig

    from lilbee.data.extract import xberg as xberg_extract

    config = ExtractionConfig(ocr=OcrConfig(backend="tesseract", language=["eng"]), force_ocr=True)
    data = png.read_bytes()
    start.with_name(f"{start.name}.{os.getpid()}").touch()
    while not start.exists():
        time.sleep(0.001)
    try:
        doc = xberg_extract.extract_document(data, "image/png", filename="race.png", config=config)
        report = {"text": doc.content, "error": None}
    except Exception as exc:
        report = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report), flush=True)
    # xberg's native threads can crash the interpreter's own shutdown; the report is out.
    os._exit(0)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
