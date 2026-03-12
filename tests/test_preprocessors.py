"""Tests for format-specific preprocessors."""

from pathlib import Path

from lilbee.preprocessors import _flatten_tree, preprocess_csv, preprocess_json, preprocess_xml


class TestPreprocessXml:
    def test_godot_class_reference(self, tmp_path: Path) -> None:
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
        result = preprocess_xml(xml)
        assert "AStarGrid2D" in result
        assert "RefCounted" in result
        assert "get_point_path" in result
        assert "from_id" in result
        assert "Vector2i" in result
        assert "Returns path IDs" in result

    def test_xsi_namespace_filtered(self, tmp_path: Path) -> None:
        """XML namespace attributes are excluded from output."""
        xml = tmp_path / "ns.xml"
        xml.write_text(
            '<class name="Foo" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            ' xsi:noNamespaceSchemaLocation="../class.xsd">\n'
            "  <brief_description>Test.</brief_description>\n"
            "</class>"
        )
        result = preprocess_xml(xml)
        assert "noNamespaceSchemaLocation" not in result
        assert "Foo" in result

    def test_nested_attributes(self, tmp_path: Path) -> None:
        xml = tmp_path / "test.xml"
        xml.write_text('<root version="2.0"><item id="1">Hello</item></root>')
        result = preprocess_xml(xml)
        assert "version" in result or "2.0" in result
        assert "Hello" in result

    def test_empty_xml(self, tmp_path: Path) -> None:
        xml = tmp_path / "empty.xml"
        xml.write_text("<root/>")
        result = preprocess_xml(xml)
        assert result.strip() == "" or "root" in result

    def test_malformed_xml(self, tmp_path: Path) -> None:
        xml = tmp_path / "bad.xml"
        xml.write_text("<root><unclosed>")
        result = preprocess_xml(xml)
        assert "<root>" in result

    def test_text_only_elements(self, tmp_path: Path) -> None:
        xml = tmp_path / "text.xml"
        xml.write_text("<doc><p>First paragraph.</p><p>Second paragraph.</p></doc>")
        result = preprocess_xml(xml)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_tail_text(self, tmp_path: Path) -> None:
        xml = tmp_path / "tail.xml"
        xml.write_text("<doc><b>bold</b> and normal text</doc>")
        result = preprocess_xml(xml)
        assert "bold" in result
        assert "normal text" in result


class TestFlattenTree:
    def test_nested_dict(self) -> None:
        data = {"a": {"b": {"c": "deep"}}}
        lines = list(_flatten_tree(data))
        assert "a.b.c: deep" in lines

    def test_list_indexing(self) -> None:
        data = {"items": [{"name": "first"}, {"name": "second"}]}
        lines = list(_flatten_tree(data))
        assert "items[0].name: first" in lines
        assert "items[1].name: second" in lines

    def test_scalar_types(self) -> None:
        data = {"s": "text", "i": 42, "f": 3.14, "b": True, "n": None}
        lines = list(_flatten_tree(data))
        assert "s: text" in lines
        assert "i: 42" in lines
        assert "f: 3.14" in lines
        assert "b: True" in lines
        assert "n: None" in lines

    def test_top_level_separation(self) -> None:
        data = {"first": "a", "second": "b"}
        lines = list(_flatten_tree(data))
        assert "" in lines


class TestPreprocessJson:
    def test_nested_object(self, tmp_path: Path) -> None:
        f = tmp_path / "test.json"
        f.write_text('{"name": "AStarGrid2D", "methods": [{"name": "get_path"}]}')
        result = preprocess_json(f)
        assert "name: AStarGrid2D" in result
        assert "methods[0].name: get_path" in result

    def test_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "test.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        result = preprocess_json(f)
        assert "a: 1" in result
        assert "b: 2" in result

    def test_empty_json(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("{}")
        result = preprocess_json(f)
        assert result.strip() == ""

    def test_malformed_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        result = preprocess_json(f)
        assert "{not valid json" in result

    def test_jsonl_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "test.jsonl"
        f.write_text('{"a": 1}\n\n{"b": 2}\n')
        result = preprocess_json(f)
        assert "a: 1" in result
        assert "b: 2" in result

    def test_jsonl_malformed_line(self, tmp_path: Path) -> None:
        f = tmp_path / "test.jsonl"
        f.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
        result = preprocess_json(f)
        assert "a: 1" in result
        assert "not json" in result
        assert "b: 2" in result


class TestPreprocessCsv:
    def test_standard_csv(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("Name,Role,Department\nAlice,Engineer,Platform\nBob,Manager,Sales\n")
        result = preprocess_csv(f)
        assert "Name: Alice" in result
        assert "Role: Engineer" in result
        assert "Department: Platform" in result
        assert "Name: Bob" in result
        assert "Row 1" in result
        assert "Row 2" in result

    def test_tsv(self, tmp_path: Path) -> None:
        f = tmp_path / "test.tsv"
        f.write_text("Name\tAge\nAlice\t30\n")
        result = preprocess_csv(f)
        assert "Name: Alice" in result
        assert "Age: 30" in result

    def test_empty_cells(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("A,B\n1,\n,2\n")
        result = preprocess_csv(f)
        assert "A: 1" in result
        assert "B: 2" in result

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = preprocess_csv(f)
        assert result.strip() == ""

    def test_single_column(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("Name\nAlice\nBob\n")
        result = preprocess_csv(f)
        assert "Name: Alice" in result
        assert "Name: Bob" in result
