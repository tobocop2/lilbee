"""Format-specific preprocessors for structured data files.

Convert structured formats (XML, JSON, CSV) into readable prose
that embeds well for vector search. Each preprocessor takes a Path
and returns a string of human-readable text.
"""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def preprocess_xml(path: Path) -> str:
    """Convert XML to readable prose using element tags as labels."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        log.warning("Malformed XML, falling back to raw text: %s", path)
        return path.read_text(encoding="utf-8", errors="replace")
    return _walk_element(tree.getroot(), depth=0).strip()


def _walk_element(elem: ET.Element, depth: int) -> str:
    """Recursively convert an XML element tree to readable text."""
    parts: list[str] = []
    tag = elem.tag
    attrs = " ".join(f"{k}: {v}" for k, v in elem.attrib.items() if not k.startswith("{"))
    label = f"{tag} ({attrs})" if attrs else tag

    if depth == 0:
        parts.append(f"{label}\n")
    elif depth == 1:
        parts.append(f"\n{label}\n")
    else:
        indent = "  " * (depth - 1)
        parts.append(f"{indent}{label}\n")

    text = (elem.text or "").strip()
    if text:
        indent = "  " * max(depth, 1)
        parts.append(f"{indent}{text}\n")

    for child in elem:
        parts.append(_walk_element(child, depth + 1))

    tail = (elem.tail or "").strip()
    if tail:
        indent = "  " * max(depth - 1, 0)
        parts.append(f"{indent}{tail}\n")

    return "".join(parts)


def _flatten_tree(data: Any, prefix: str = "", _top: bool = True) -> Iterator[str]:
    """Walk nested dicts/lists, yielding 'dotted.path: value' lines."""
    if isinstance(data, dict):
        for i, (key, val) in enumerate(data.items()):
            path = f"{prefix}.{key}" if prefix else key
            if _top and i > 0:
                yield ""
            yield from _flatten_tree(val, path, _top=False)
    elif isinstance(data, list):
        for i, val in enumerate(data):
            path = f"{prefix}[{i}]"
            yield from _flatten_tree(val, path, _top=False)
    else:
        yield f"{prefix}: {data}"


def preprocess_csv(path: Path) -> str:
    """Convert CSV/TSV to readable 'Header: Value' per row."""
    delimiter = "\t" if path.suffix == ".tsv" else ","
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return ""
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    sections: list[str] = []
    for i, row in enumerate(reader, 1):
        lines = [f"Row {i}:"]
        for header, value in row.items():
            if header and value and value.strip():
                lines.append(f"  {header}: {value}")
        if len(lines) > 1:
            sections.append("\n".join(lines))
    return "\n\n".join(sections)


def preprocess_json(path: Path) -> str:
    """Convert JSON/JSONL to readable 'dotted.path: value' lines."""
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".jsonl":
        sections: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                sections.append(line)
                continue
            sections.append("\n".join(_flatten_tree(obj)))
        return "\n\n".join(sections)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Malformed JSON, falling back to raw text: %s", path)
        return text
    return "\n".join(_flatten_tree(data))
