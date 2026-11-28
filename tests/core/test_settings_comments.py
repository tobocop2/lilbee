"""A settings write keeps the comments and the key order of the file it changes."""

import stat
import tomllib
from pathlib import Path

import pytest
from tests._private_mode import posix_only

from lilbee.core import settings

CORPUS = """\
# header one
# header two
zeta = 1  # inline note on zeta
# between the keys
alpha = "x"
mid = true

# before the profile table
[profile]
name = "p"
# inside the profile table
[profile.values]
top_k = 3
# after the profile table
"""

COMMENTS = [
    "# header one",
    "# header two",
    "# inline note on zeta",
    "# between the keys",
    "# before the profile table",
    "# inside the profile table",
    "# after the profile table",
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config.toml").write_text(CORPUS, encoding="utf-8")
    return tmp_path


def _text(root: Path) -> str:
    return (root / "config.toml").read_text(encoding="utf-8")


def _assert_comments_and_order(text: str) -> None:
    assert all(comment in text for comment in COMMENTS)
    assert text.index("zeta") < text.index("alpha") < text.index("mid")


def test_set_value_changes_one_value_and_keeps_every_comment(root: Path) -> None:
    settings.set_value(root, "alpha", "y")
    text = _text(root)
    _assert_comments_and_order(text)
    assert tomllib.loads(text)["alpha"] == "y"
    assert tomllib.loads(text)["zeta"] == 1


def test_set_value_of_a_new_key_appends_it_and_keeps_every_comment(root: Path) -> None:
    settings.set_value(root, "fresh", 7)
    text = _text(root)
    _assert_comments_and_order(text)
    parsed = tomllib.loads(text)
    assert parsed["fresh"] == 7
    assert parsed["profile"] == {"name": "p", "values": {"top_k": 3}}


def test_update_values_keeps_every_comment(root: Path) -> None:
    settings.update_values(root, {"alpha": "z", "mid": False, "extra": [1, 2]})
    text = _text(root)
    _assert_comments_and_order(text)
    parsed = tomllib.loads(text)
    assert (parsed["alpha"], parsed["mid"], parsed["extra"]) == ("z", False, [1, 2])


def test_mutate_value_keeps_every_comment(root: Path) -> None:
    result = settings.mutate_value(root, "alpha", lambda old: (old + "!", old))
    assert result == "x"
    text = _text(root)
    _assert_comments_and_order(text)
    assert tomllib.loads(text)["alpha"] == "x!"


def test_write_profile_table_keeps_every_comment_outside_the_table(root: Path) -> None:
    settings.write_profile_table(root, "q", {"top_k": 5, "chunk_size": 300}, drop=["mid"])
    text = _text(root)
    for comment in COMMENTS[:4]:
        assert comment in text
    parsed = tomllib.loads(text)
    assert parsed["profile"] == {"name": "q", "values": {"top_k": 5, "chunk_size": 300}}
    assert "mid" not in parsed


def test_a_write_of_the_value_a_key_has_leaves_the_file_bytes(root: Path) -> None:
    before = (root / "config.toml").read_bytes()
    settings.set_value(root, "alpha", "x")
    settings.update_values(root, {"zeta": 1, "mid": True})
    assert (root / "config.toml").read_bytes() == before


def test_delete_value_keeps_the_comments_and_the_other_keys(root: Path) -> None:
    settings.delete_value(root, "alpha")
    text = _text(root)
    for comment in COMMENTS:
        assert comment in text
    assert "alpha" not in tomllib.loads(text)
    assert tomllib.loads(text)["mid"] is True


def test_delete_values_drops_an_inline_comment_with_its_key(root: Path) -> None:
    settings.delete_values(root, ["zeta", "mid"])
    text = _text(root)
    assert "# inline note on zeta" not in text
    for comment in COMMENTS:
        if comment != "# inline note on zeta":
            assert comment in text
    assert set(tomllib.loads(text)) == {"alpha", "profile"}


def test_a_nested_key_that_is_no_longer_wanted_is_removed(root: Path) -> None:
    settings.save(root, {"profile": {"name": "p"}, "alpha": "x"})
    assert tomllib.loads(_text(root))["profile"] == {"name": "p"}


def test_a_table_replaces_a_scalar_of_the_same_key(root: Path) -> None:
    settings.save(root, {"alpha": {"inner": 1}})
    assert tomllib.loads(_text(root)) == {"alpha": {"inner": 1}}


def test_a_new_file_holds_the_values_with_none_dropped_and_a_path_as_text(
    tmp_path: Path,
) -> None:
    settings.save(tmp_path, {"b": 2, "a": None, "where": tmp_path / "x", "c": "t"})
    assert tomllib.loads(_text(tmp_path)) == {"b": 2, "where": str(tmp_path / "x"), "c": "t"}


def test_a_file_that_does_not_parse_is_replaced_by_the_new_values(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("a = = 1\n", encoding="utf-8")
    settings.save(tmp_path, {"b": 2})
    assert tomllib.loads(_text(tmp_path)) == {"b": 2}


def test_a_file_nested_too_deep_is_replaced_by_the_new_values(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("a = " + "[" * 3000 + "]" * 3000 + "\n", encoding="utf-8")
    settings.save(tmp_path, {"b": 2})
    assert tomllib.loads(_text(tmp_path)) == {"b": 2}


@posix_only
def test_the_written_file_is_owner_only(root: Path) -> None:
    settings.set_value(root, "alpha", "y")
    assert stat.S_IMODE((root / "config.toml").stat().st_mode) == 0o600


class TestAWriteChangesAValueOfAnotherTypeThatComparesEqual:
    """Python reads True as 1 and 1.0 as 1, so equality alone skips a typed write."""

    @pytest.mark.parametrize(
        ("held", "wanted"),
        [
            ("1", True),
            ("0", False),
            ("1", 1.0),
            ("true", 1),
            ("[1, 2]", [1.0, 2]),
            ("[1, 2]", [1, 2, 3]),
        ],
    )
    def test_the_file_takes_the_new_type(self, tmp_path: Path, held, wanted) -> None:
        settings._config_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        settings._config_path(tmp_path).write_text(f"key = {held}  # note\n", encoding="utf-8")
        settings.save(tmp_path, {"key": wanted})
        text = settings._config_path(tmp_path).read_text(encoding="utf-8")
        read = tomllib.loads(text)["key"]
        assert read == wanted
        assert repr(read) == repr(wanted)
        assert "# note" in text

    def test_an_equal_value_of_the_same_type_leaves_the_text_alone(self, tmp_path: Path) -> None:
        path = settings._config_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        before = "key = [1, 2]  # note\nother   =   'x'\n"
        path.write_text(before, encoding="utf-8")
        settings.save(tmp_path, {"key": [1, 2], "other": "x"})
        assert path.read_text(encoding="utf-8") == before


class TestAnEmptyTableIsWritten:
    """An empty table holds its key; a table that only holds subtables is not written then."""

    @pytest.mark.parametrize(
        "held",
        ["[a.b]\nx = 1\n", "[a]\n[a.b]\nx = 1\n", "[a]\ny = 2\n"],
    )
    def test_over_a_nested_table(self, tmp_path: Path, held: str) -> None:
        (tmp_path / "config.toml").write_text(held, encoding="utf-8")
        settings.save(tmp_path, {"a": {}})
        assert tomllib.loads(_text(tmp_path)) == {"a": {}}

    def test_inside_a_table(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[a.b.c]\nx = 1\n", encoding="utf-8")
        settings.save(tmp_path, {"a": {"b": {}}})
        assert tomllib.loads(_text(tmp_path)) == {"a": {"b": {}}}

    def test_over_an_empty_table_leaves_the_file_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[a]  # note\n", encoding="utf-8")
        before = (tmp_path / "config.toml").read_bytes()
        settings.save(tmp_path, {"a": {}})
        assert (tmp_path / "config.toml").read_bytes() == before


class TestTheFileKeepsItsLineEnds:
    """A write keeps the line end the file uses; a file with mixed line ends gets LF."""

    def test_a_write_of_the_value_a_key_has_leaves_crlf_bytes(self, tmp_path: Path) -> None:
        before = CORPUS.replace("\n", "\r\n").encode()
        (tmp_path / "config.toml").write_bytes(before)
        settings.set_value(tmp_path, "alpha", "x")
        assert (tmp_path / "config.toml").read_bytes() == before

    def test_a_changed_and_a_new_key_use_crlf(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_bytes(CORPUS.replace("\n", "\r\n").encode())
        settings.update_values(tmp_path, {"alpha": "y", "new_key": 2})
        raw = (tmp_path / "config.toml").read_bytes()
        assert raw.count(b"\n") == raw.count(b"\r\n")
        assert b'alpha = "y"\r\n' in raw
        assert b"new_key = 2\r\n" in raw
        _assert_comments_and_order(raw.decode())

    def test_a_file_with_mixed_line_ends_gets_lf(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_bytes(b"a = 1\r\nb = 2\n")
        settings.set_value(tmp_path, "c", 3)
        assert (tmp_path / "config.toml").read_bytes() == b"a = 1\nb = 2\nc = 3\n"

    def test_a_new_file_gets_lf(self, tmp_path: Path) -> None:
        settings.set_value(tmp_path, "c", 3)
        assert (tmp_path / "config.toml").read_bytes() == b"c = 3\n"
