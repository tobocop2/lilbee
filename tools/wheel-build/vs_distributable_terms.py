"""Write or verify the Distributable Code section of a Visual Studio licence terms .docx file.

Usage:
  vs_distributable_terms.py LICENSE.docx OUTPUT
  vs_distributable_terms.py --verify LICENSE.docx WHEEL

The section starts at the paragraph "DISTRIBUTABLE CODE." and ends before the next
paragraph that opens with an upper case heading. The first form writes the section to
OUTPUT as UTF-8. It fails when it finds no section, or a section without the
distribution requirements and restrictions. The second form fails unless the notice in
WHEEL is valid UTF-8, holds the section line for line, and holds the sentence that the
licence check needs.
"""

from __future__ import annotations

import sys
import zipfile
from xml.etree import ElementTree

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_START = "DISTRIBUTABLE CODE."
_REQUIRED = ("Distribution Requirements", "Distribution Restrictions")
_NOTICE_SUFFIX = ".dist-info/licenses/NOTICE-MICROSOFT-VC-RUNTIME"
_SENTENCE = "They are distributable code"


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


def verify_notice(section: list[str], wheel: str) -> None:
    """Raise ValueError unless the wheel's notice is UTF-8 and holds *section* line for line."""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(_NOTICE_SUFFIX)]
        if len(names) != 1:
            raise ValueError(f"the wheel has {len(names)} Visual C++ runtime notices, not one")
        raw = archive.read(names[0])
    try:
        notice = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"the notice is not valid UTF-8: {error}") from error
    if _SENTENCE not in notice:
        raise ValueError(f"the notice lacks the sentence {_SENTENCE!r}")
    lines = notice.split("\n")
    held = any(lines[i : i + len(section)] == section for i in range(len(lines)))
    if not held:
        raise ValueError("the notice does not hold the licence terms line for line")


def main(argv: list[str]) -> int:
    verifying = argv[1:2] == ["--verify"]
    args = argv[2:] if verifying else argv[1:]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        section = distributable_section(paragraphs(args[0]))
        if verifying:
            verify_notice(section, args[1])
        else:
            with open(args[1], "w", encoding="utf-8", newline="\n") as out:
                out.write("\n".join(section) + "\n")
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"vs_distributable_terms: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
