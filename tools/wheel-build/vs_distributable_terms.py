"""Print the Distributable Code section of a Visual Studio licence terms .docx file.

Usage: vs_distributable_terms.py LICENSE.docx

The section starts at the paragraph "DISTRIBUTABLE CODE." and ends before the next
paragraph that opens with an upper case heading. The tool fails when it finds no
section, or a section without the distribution requirements.
"""

from __future__ import annotations

import sys
import zipfile
from xml.etree import ElementTree

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_START = "DISTRIBUTABLE CODE."
_REQUIRED = ("Distribution Requirements", "Distribution Restrictions")


def paragraphs(docx: str) -> list[str]:
    """The text of each paragraph of the document, without empty ones."""
    with zipfile.ZipFile(docx) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    texts = ("".join(node.text or "" for node in p.iter(f"{_W}t")).strip() for p in root.iter(f"{_W}p"))
    return [text for text in texts if text]


def _is_heading(text: str) -> bool:
    head = text.split(".", 1)[0]
    return len(head) > 3 and head.isupper()


def distributable_section(lines: list[str]) -> list[str]:
    """The paragraphs of the Distributable Code section, heading first."""
    starts = [i for i, line in enumerate(lines) if line.startswith(_START)]
    if not starts:
        raise ValueError("the document has no DISTRIBUTABLE CODE section")
    section = [lines[starts[0]]]
    for line in lines[starts[0] + 1 :]:
        if _is_heading(line):
            break
        section.append(line)
    body = "\n".join(section)
    for phrase in _REQUIRED:
        if phrase not in body:
            raise ValueError(f"the DISTRIBUTABLE CODE section has no {phrase!r}")
    return section


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        section = distributable_section(paragraphs(argv[1]))
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"vs_distributable_terms: {error}", file=sys.stderr)
        return 1
    sys.stdout.write("\n".join(section) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
