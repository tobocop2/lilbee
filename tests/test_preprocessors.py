"""Tests for structured format extraction (formerly custom preprocessors, now kreuzberg).

These tests verify that structured formats (XML, CSV, JSON) are correctly
classified and extracted via kreuzberg. The custom preprocessors were removed
in favor of kreuzberg's native extraction.
"""

from pathlib import Path

from lilbee.ingest import classify_file


class TestStructuredFormatClassification:
    """Verify structured formats are still recognized and classified."""

    def test_xml_classified(self) -> None:
        assert classify_file(Path("data.xml")) == "xml"

    def test_json_classified(self) -> None:
        assert classify_file(Path("data.json")) == "json"

    def test_jsonl_classified(self) -> None:
        assert classify_file(Path("data.jsonl")) == "json"

    def test_csv_classified(self) -> None:
        assert classify_file(Path("data.csv")) == "data"

    def test_tsv_classified(self) -> None:
        assert classify_file(Path("data.tsv")) == "data"

    def test_yaml_classified(self) -> None:
        assert classify_file(Path("config.yaml")) == "text"

    def test_yml_classified(self) -> None:
        assert classify_file(Path("config.yml")) == "text"


class TestXmlExtraction:
    def test_godot_class_reference(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        xml = tmp_path / "AStarGrid2D.xml"
        xml.write_text(
            '<?xml version="1.0"?>\n'
            '<class name="AStarGrid2D" inherits="RefCounted">\n'
            "  <brief_description>A* on a 2D grid.</brief_description>\n"
            "  <methods>\n"
            '    <method name="get_point_path">\n'
            '      <return type="PackedVector2Array" />\n'
            '      <param index="0" name="from_id" type="Vector2i" />\n'
            "      <description>Returns path IDs.</description>\n"
            "    </method>\n"
            "  </methods>\n"
            "</class>"
        )
        result = extract_file_sync(str(xml))
        assert "AStarGrid2D" in result.content
        assert "get_point_path" in result.content
        assert "Returns path IDs" in result.content

    def test_nested_attributes(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        xml = tmp_path / "test.xml"
        xml.write_text('<root version="2.0"><item id="1">Hello</item></root>')
        result = extract_file_sync(str(xml))
        assert "Hello" in result.content

    def test_empty_xml(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        xml = tmp_path / "empty.xml"
        xml.write_text("<root/>")
        result = extract_file_sync(str(xml))
        assert isinstance(result.content, str)

    def test_text_only_elements(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        xml = tmp_path / "text.xml"
        xml.write_text("<doc><p>First paragraph.</p><p>Second paragraph.</p></doc>")
        result = extract_file_sync(str(xml))
        assert "First paragraph" in result.content
        assert "Second paragraph" in result.content


class TestJsonExtraction:
    def test_nested_object(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "test.json"
        f.write_text('{"name": "AStarGrid2D", "methods": [{"name": "get_path"}]}')
        result = extract_file_sync(str(f))
        assert "AStarGrid2D" in result.content
        assert "get_path" in result.content

    def test_empty_json(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "empty.json"
        f.write_text("{}")
        result = extract_file_sync(str(f))
        assert isinstance(result.content, str)


class TestCsvExtraction:
    def test_standard_csv(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "test.csv"
        f.write_text("Name,Role,Department\nAlice,Engineer,Platform\nBob,Manager,Sales\n")
        result = extract_file_sync(str(f))
        assert "Alice" in result.content
        assert "Engineer" in result.content
        assert "Bob" in result.content

    def test_tsv(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "test.tsv"
        f.write_text("Name\tAge\nAlice\t30\n")
        result = extract_file_sync(str(f))
        assert "Alice" in result.content
        assert "30" in result.content

    def test_empty_file(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "empty.csv"
        f.write_text("")
        result = extract_file_sync(str(f))
        assert isinstance(result.content, str)

    def test_single_column(self, tmp_path: Path) -> None:
        from kreuzberg import extract_file_sync

        f = tmp_path / "test.csv"
        f.write_text("Name\nAlice\nBob\n")
        result = extract_file_sync(str(f))
        assert "Alice" in result.content
        assert "Bob" in result.content
