"""tools/wheel-build/embedded_ui_notices.py writes the licence notices of an embedded web page."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools/wheel-build/embedded_ui_notices.py"
_spec = importlib.util.spec_from_file_location("embedded_ui_notices", _TOOL)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

_REACT = (
    "/**\n * @license React\n * react.production.min.js\n *\n * Copyright (c) Facebook, Inc.\n"
    " * This source code is licensed under the MIT license found in the\n * LICENSE file.\n */"
)
_PAGE = (
    f"<script>var a=1;{_REACT}var b=2;/*#__PURE__*/ var c=3;\n"
    "var s='/**\\n * @licstart notice\\n * Licensed under the Apache License, Version 2.0\\n */';"
    f"{_REACT}</script>"
)


def test_the_notices_are_the_licence_comments_once_each_and_not_the_code() -> None:
    found = tool.notices(_PAGE)
    assert len(found) == 2
    assert found[0] == _REACT
    assert (
        found[1]
        == "/**\n * @licstart notice\n * Licensed under the Apache License, Version 2.0\n */"
    )


def test_a_comment_over_the_size_limit_is_not_a_notice() -> None:
    page = "/* @license " + "x" * tool.MAX_NOTICE_CHARS + " */"
    assert tool.notices(page) == []


def test_main_writes_the_notices_of_a_gzip_page_and_the_mit_text(tmp_path: Path) -> None:
    bundle = tmp_path / "index.html.gz"
    bundle.write_bytes(gzip.compress(_PAGE.encode()))
    out = tmp_path / "deep" / "index.html.gz.NOTICE"
    tool.main(["tool", str(bundle), str(out)])
    text = out.read_text(encoding="utf-8")
    assert "@license React" in text
    assert "Licensed under the Apache License, Version 2.0" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "var b=2" not in text


def test_main_adds_no_mit_text_when_no_notice_names_mit(tmp_path: Path) -> None:
    bundle = tmp_path / "index.html"
    bundle.write_text("/* Licensed under the Apache License, Version 2.0 */", encoding="utf-8")
    out = tmp_path / "out.NOTICE"
    tool.main(["tool", str(bundle), str(out)])
    assert "Permission is hereby granted" not in out.read_text(encoding="utf-8")


def test_main_refuses_a_page_with_no_notice(tmp_path: Path) -> None:
    bundle = tmp_path / "index.html"
    bundle.write_text("<html></html>", encoding="utf-8")
    with pytest.raises(SystemExit, match="holds no licence notice"):
        tool.main(["tool", str(bundle), str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_main_needs_two_arguments() -> None:
    with pytest.raises(SystemExit, match=r"Usage: embedded_ui_notices\.py"):
        tool.main(["tool"])
